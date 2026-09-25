"""Bounded offline Codex resume probe against a loopback-only HTTP stub.

This diagnostic uses only the two frozen study session IDs. The stub reads and
discards request bytes, recording only method, URL path, and byte count, then
returns HTTP 400. It never contacts Ashkelon or an external model endpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit


REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / 'research/bridge-v2/addressed-feedback-codex-gpt6-sol'
SESSION_IDS = {
    'contingent': '01a0d910-5f8e-7a11-8a66-21a07378e777',
    'yoked': '01a0d910-6cb9-7762-8d8f-5c3ef7577c42',
}
MODEL = 'gpt-6-sol'
TIMEOUT_SECONDS = 10.0
MAX_INVOCATIONS = 4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def _study_prompts() -> dict[str, str]:
    prompts = {}
    with (STUDY / 'turns.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            if row.get('turn') == 2 and row.get('mode') in SESSION_IDS:
                prompts[row['mode']] = row['prompt']
    if set(prompts) != set(SESSION_IDS):
        raise RuntimeError('the two frozen turn-2 prompts were not both found')
    return prompts


def _rollout_path(session_id: str) -> Path:
    sessions = Path.home() / '.codex' / 'sessions'
    found = [p for p in sessions.rglob(f'*{session_id}.jsonl') if p.is_file()]
    if len(found) != 1:
        raise RuntimeError(f'expected one rollout file for the requested session; found {len(found)}')
    return found[0]


def _codex_install() -> tuple[Path, Path]:
    launcher = shutil.which('codex')
    if launcher is None:
        raise FileNotFoundError('codex CLI not found on PATH')
    launcher_path = Path(launcher).resolve()
    package_root = launcher_path.parent.parent
    native_candidates = list((package_root / 'node_modules/@openai/codex-darwin-arm64/vendor').glob('*/bin/codex'))
    if len(native_candidates) != 1:
        raise RuntimeError('expected one installed arm64 Codex native executable')
    return launcher_path, native_candidates[0].resolve()


class StubState:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests: list[dict] = []


def _stub_server(state: StubState) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, _format, *_args):
            return

        def _discard_request_body(self) -> int:
            count = 0
            transfer = self.headers.get('Transfer-Encoding', '').lower()
            if 'chunked' in transfer:
                while True:
                    line = self.rfile.readline(128)
                    if not line:
                        break
                    try:
                        size = int(line.split(b';', 1)[0].strip(), 16)
                    except ValueError:
                        break
                    if size == 0:
                        # Consume optional trailers through the terminating CRLF.
                        while self.rfile.readline(8192) not in (b'\r\n', b'\n', b''):
                            pass
                        break
                    remaining = size
                    while remaining:
                        block = self.rfile.read(min(65536, remaining))
                        if not block:
                            return count
                        count += len(block)
                        remaining -= len(block)
                    self.rfile.read(2)
            else:
                try:
                    remaining = int(self.headers.get('Content-Length', '0'))
                except ValueError:
                    remaining = 0
                while remaining > 0:
                    block = self.rfile.read(min(65536, remaining))
                    if not block:
                        break
                    count += len(block)
                    remaining -= len(block)
            return count

        def _respond(self):
            body_bytes = self._discard_request_body()
            record = {
                'method': self.command,
                'path': urlsplit(self.path).path,
                'body_bytes': body_bytes,
                'received': True,
                'response_status': 400,
            }
            with state.lock:
                state.requests.append(record)
            response = b'{"error":{"message":"offline diagnostic stub"}}'
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(response)
            self.close_connection = True

        do_POST = _respond
        do_GET = _respond
        do_PUT = _respond
        do_DELETE = _respond

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    return server


def _process_tree_rss(root_pid: int) -> tuple[list[dict], int]:
    """Return PIDs and RSS only; never inspect process command lines or environments."""
    try:
        raw = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss='], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return [], 0
    rows = {}
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            pid, ppid, rss = map(int, fields)
        except ValueError:
            continue
        rows[pid] = {'pid': pid, 'ppid': ppid, 'rss_kib': rss}
    members = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, row in rows.items():
            if row['ppid'] in members and pid not in members:
                members.add(pid)
                changed = True
    selected = [rows[pid] for pid in sorted(members) if pid in rows]
    return selected, sum(row['rss_kib'] for row in selected)


def _safe_child_env() -> dict[str, str]:
    env = os.environ.copy()
    # Keep HOME and CODEX_HOME exactly as inherited; remove provider keys and
    # proxy overrides so requests cannot leave loopback or use saved API keys.
    for key in (
        'OPENAI_API_KEY', 'CODEX_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_ORG_ID',
        'OPENAI_PROJECT_ID', 'AZURE_OPENAI_API_KEY', 'ANTHROPIC_API_KEY',
        'ANTHROPIC_BASE_URL', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
        'http_proxy', 'https_proxy', 'all_proxy', 'NO_PROXY', 'no_proxy',
    ):
        env.pop(key, None)
    env['NO_PROXY'] = '127.0.0.1,localhost'
    env['no_proxy'] = env['NO_PROXY']
    return env


def _command(executable: Path, base_url: str, session_id: str, prompt: str, output_path: Path) -> list[str]:
    provider = ('{name="ashkelon",base_url=' + json.dumps(base_url)
                + ',wire_api="responses",requires_openai_auth=false}')
    return [
        str(executable), 'exec', '--ignore-user-config', '-m', MODEL,
        '-c', 'project_doc_max_bytes=0',
        '-c', f'model_providers.ashkelon={provider}',
        '-c', 'model_provider="ashkelon"',
        'resume', '--skip-git-repo-check', '--json', '-o', str(output_path), session_id, prompt,
    ]


def _run_one(invocation: int, mode: str, session_id: str, prompt: str,
             executable: Path, out: Path, timeout: float) -> dict:
    state = StubState()
    server = _stub_server(state)
    thread = threading.Thread(target=server.serve_forever, name=f'local-stub-{invocation}', daemon=True)
    thread.start()
    # Verify bind address is loopback before starting Codex.
    host, port = server.server_address
    if host != '127.0.0.1':
        server.shutdown()
        raise RuntimeError('offline stub did not bind to loopback')
    endpoint = f'http://127.0.0.1:{port}'
    tmp = Path(tempfile.mkdtemp(prefix=f'codex-resume-offline-{invocation}-'))
    os.chmod(tmp, 0o700)
    stdout_path, stderr_path = tmp / 'stdout.bin', tmp / 'stderr.bin'
    answer_path = tmp / 'answer.txt'
    argv = _command(executable, endpoint, session_id, prompt, answer_path)
    argv_sha = hashlib.sha256(json.dumps(argv, separators=(',', ':')).encode()).hexdigest()
    process = None
    samples = []
    started = utc_now()
    timed_out = False
    try:
        with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
            process = subprocess.Popen(
                argv, cwd=tmp, env=_safe_child_env(), stdout=stdout, stderr=stderr,
                start_new_session=True,
            )
            deadline = time.monotonic() + timeout
            while process.poll() is None and time.monotonic() < deadline:
                tree, total_rss = _process_tree_rss(process.pid)
                samples.append({'elapsed_seconds': round(timeout - max(0.0, deadline - time.monotonic()), 3),
                                'pids': tree, 'total_rss_kib': total_rss})
                time.sleep(0.1)
            if process.poll() is None:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=1.0)
        exit_code = process.returncode
        stdout_bytes = stdout_path.stat().st_size
        stderr_bytes = stderr_path.stat().st_size
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
    requests = list(state.requests)
    seen_pids = sorted({entry['pid'] for sample in samples for entry in sample['pids']})
    peak_rss_kib = max((sample['total_rss_kib'] for sample in samples), default=0)
    # Remove any captured CLI output; only byte counts are retained.
    shutil.rmtree(tmp, ignore_errors=True)
    return {
        'invocation': invocation,
        'mode': mode,
        'session_id': session_id,
        'executable': str(executable),
        'executable_sha256': sha256(executable),
        'argv_sha256': argv_sha,
        'stub_base_url': endpoint,
        'started_at_utc': started,
        'finished_at_utc': utc_now(),
        'timeout_seconds': timeout,
        'timed_out': timed_out,
        'pid': process.pid if process else None,
        'exit_code': exit_code if process else None,
        'termination_signal': -exit_code if process and exit_code < 0 else None,
        'stdout_bytes': stdout_bytes if process else None,
        'stderr_bytes': stderr_bytes if process else None,
        'observed_process_ids': seen_pids,
        'peak_process_tree_rss_kib': peak_rss_kib,
        'rss_samples': samples,
        'stub_requests': requests,
        'http_request_received': bool(requests),
    }


def _copy_private_rollouts(out: Path) -> list[dict]:
    target = out / 'preprobe_rollouts'
    target.mkdir(mode=0o700)
    records = []
    for mode, session_id in SESSION_IDS.items():
        src = _rollout_path(session_id)
        digest = sha256(src)
        record_types = Counter()
        compaction_records = 0
        with src.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record_type = row.get('type')
                payload_type = (row.get('payload') or {}).get('type')
                if isinstance(record_type, str):
                    record_types[record_type] += 1
                if record_type == 'compaction' or payload_type == 'compaction':
                    compaction_records += 1
        dest = target / f'{mode}-{session_id}.jsonl'
        shutil.copyfile(src, dest)
        os.chmod(dest, 0o600)
        records.append({'mode': mode, 'session_id': session_id,
                        'source_path': str(src), 'source_sha256_before_probe': digest,
                        'private_copy': str(dest), 'private_copy_sha256': sha256(dest),
                        'source_size_bytes': src.stat().st_size,
                        'source_mtime_utc': datetime.fromtimestamp(src.stat().st_mtime, timezone.utc).isoformat(),
                        'record_types': dict(record_types),
                        'compaction_record_count': compaction_records})
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(out, 0o700)
    launcher, native = _codex_install()
    prompts = _study_prompts()
    preprobe = _copy_private_rollouts(out)
    _snapshot_runner(out)
    invocations = []
    sessions = [('contingent', launcher), ('yoked', launcher)]
    for index, (mode, executable) in enumerate(sessions, start=1):
        # Each invocation creates its own loopback-only stub and exits at most 10s later.
        row = _run_one(index, mode, SESSION_IDS[mode], prompts[mode], executable, out, TIMEOUT_SECONDS)
        invocations.append(row)
        _write_results(out, invocations, preprobe, launcher, native)
    if all(row.get('termination_signal') == signal.SIGKILL for row in invocations):
        # Authorized bounded comparison: same contingent resume via native binary.
        row = _run_one(3, 'contingent', SESSION_IDS['contingent'], prompts['contingent'],
                       native, out, TIMEOUT_SECONDS)
        invocations.append(row)
        _write_results(out, invocations, preprobe, launcher, native)
    print(json.dumps({'output_dir': str(out), 'invocations': len(invocations),
                      'results_path': str(out / 'results.json')}, sort_keys=True))
    return 0


def _write_results(out: Path, invocations: list[dict], preprobe: list[dict],
                   launcher: Path, native: Path) -> None:
    source = Path(__file__).resolve()
    snapshot_dir = out / 'source_snapshot'
    snapshot_dir.mkdir(mode=0o700, exist_ok=True)
    snapshot = snapshot_dir / source.name
    if not snapshot.exists():
        shutil.copyfile(source, snapshot)
        os.chmod(snapshot, 0o600)
    revision = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    metadata = {
        'schema': 'codex-resume-offline-probe-v1',
        'probe_started_before_first_invocation': invocations[0]['started_at_utc'],
        'recorded_at_utc': utc_now(),
        'repo_revision': revision,
        'runner_source_sha256': sha256(source),
        'source_snapshot_sha256': sha256(snapshot),
        'codex_cli_version': 'codex-cli 0.157.0',
        'codex_launcher': {'path': str(launcher), 'sha256': sha256(launcher)},
        'codex_native': {'path': str(native), 'sha256': sha256(native)},
        'session_rollouts_preprobe': preprobe,
        'invocation_count': len(invocations),
        'invocations': invocations,
        'network_boundary': 'Codex model provider base URL is an ephemeral 127.0.0.1 stub; no Ashkelon or external model endpoint is configured',
        'request_capture': 'method, URL path without query, body byte count, received flag, stub status only; headers/body are never retained',
    }
    (out / 'results.json').write_text(json.dumps(metadata, indent=2) + '\n')


def _snapshot_runner(out: Path) -> None:
    source = Path(__file__).resolve()
    snapshot_dir = out / 'source_snapshot'
    snapshot_dir.mkdir(mode=0o700)
    snapshot = snapshot_dir / source.name
    shutil.copyfile(source, snapshot)
    os.chmod(snapshot, 0o600)


if __name__ == '__main__':
    raise SystemExit(main())
