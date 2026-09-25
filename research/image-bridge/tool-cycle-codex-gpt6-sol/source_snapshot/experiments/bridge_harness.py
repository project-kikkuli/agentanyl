"""Small local harness for controlled Agentanyl bridge experiments.

It starts the installed or explicitly selected Ashkelon relay, runs one Claude Code or Codex turn at a
time, and snapshots the corresponding hook, provider call, and controller trace.
Provider failures are surfaced immediately and are never retried.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ASHKELON = REPO / '.build' / 'ashkelon' / 'target' / 'release' / 'ashkelon'
_PROVIDER_LOCK = threading.Lock()
_SESSION_RE = re.compile(r'session id: ([0-9a-f-]{36})', re.I)
_ANSI_RE = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
_CREDENTIAL_RE = re.compile(r'(?i)\b(authorization|api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*([^\s,;]+)')
_BEARER_RE = re.compile(r'(?i)\bBearer\s+[^\s,;]+')
_API_TOKEN_RE = re.compile(r'\b(?:sk-ant-|sk-)\w{8,}\b')


@dataclass
class TurnResult:
    answer: str
    session_id: str
    model: str
    http_status: int | None
    call_id: str | None
    pings_injected: list[str] | None
    token_usage: dict
    hook_status: str | None
    controller_trace: dict | None
    observed_model: str | None = None
    ashkelon_hook_statuses: list[str] | None = None
    trace_candidate_ids: list[str] | None = None
    selected_trace_prompt_matches: bool | None = None
    provider_calls: list[dict] | None = None
    ashkelon_binary: dict | None = None


def resolve_ashkelon_path(explicit=None, environ=None) -> Path:
    """Resolve explicit override, AGENTANYL_ASHKELON, or the installed repository default."""
    if explicit is not None:
        raw_path = explicit
    else:
        env = os.environ if environ is None else environ
        raw_path = env.get('AGENTANYL_ASHKELON') or DEFAULT_ASHKELON
    return Path(raw_path).expanduser().resolve()


def ashkelon_binary_metadata(path: Path) -> dict:
    """Identify the executable used by a run, independently of a mutable tag or symlink."""
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open('rb') as binary:
        for chunk in iter(lambda: binary.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def _jsonl(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(errors='replace').splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except json.JSONDecodeError:
                continue
    return rows


def _session_matches(row: dict, session_id: str) -> bool:
    session = row.get('session')
    if isinstance(session, dict):
        session = session.get('session')
    return session == session_id or (isinstance(session, str) and session.endswith(':' + session_id))


class BridgeHarness:
    """Run isolated turns through the local relay.

    The criteria file is read by the hook in place; the SQLite database and
    all Ashkelon configuration, state, and logs live in a private temp folder.
    """

    def __init__(self, criteria_path, db_path, model: str, *, port: int = 18520,
                 ashkelon_path=None, hook_timeout: float = 45,
                 startup_timeout: float = 10, provider_timeout: float = 180,
                 allow_shell_tool: bool = False,
                 tool_call_hook_command: list[str] | None = None):
        self.criteria_path = Path(criteria_path).expanduser().resolve()
        self.db_path = Path(db_path).expanduser().resolve()
        self.model = model
        self.port = int(port)
        self.ashkelon_path = resolve_ashkelon_path(ashkelon_path)
        self.ashkelon_binary = None
        self.hook_timeout = hook_timeout
        self.startup_timeout = startup_timeout
        self.provider_timeout = provider_timeout
        self.allow_shell_tool = bool(allow_shell_tool)
        if tool_call_hook_command is not None and not all(isinstance(item, str) for item in tool_call_hook_command):
            raise TypeError('tool_call_hook_command must be a list of strings')
        self.tool_call_hook_command = tool_call_hook_command
        self._tmp = None
        self._server = None
        self._active = False

    def __enter__(self):
        if self._active:
            raise RuntimeError('harness is already active')
        if not self.criteria_path.is_file():
            raise FileNotFoundError('criteria JSON file does not exist')
        if not self.ashkelon_path.is_file():
            raise FileNotFoundError(
                f'Ashkelon binary does not exist at {self.ashkelon_path}; '
                'run ./install-ashkelon.sh or set AGENTANYL_ASHKELON'
            )
        self.ashkelon_binary = ashkelon_binary_metadata(self.ashkelon_path)
        self._tmp = tempfile.TemporaryDirectory(prefix='agentanyl-bridge-')
        self.root = Path(self._tmp.name)
        self.cwd = self.root / 'empty-cwd'
        self.cwd.mkdir()
        self.log_dir = self.root / 'logs'
        self.state_dir = self.root / 'state'
        self.log_dir.mkdir(mode=0o700)
        self.state_dir.mkdir(mode=0o700)
        self.config_path = self.root / 'ashkelon.toml'
        setup = subprocess.run([
            sys.executable, '-m', 'agentanyl.setup', '--criteria', str(self.criteria_path),
            '--output', str(self.config_path), '--db', str(self.db_path),
        ], cwd=REPO, text=True, capture_output=True)
        if setup.returncode:
            self.close()
            raise RuntimeError(f'agentanyl.setup failed with exit code {setup.returncode}')
        config = self.config_path.read_text()
        prefix = (f'listen = {json.dumps(f"127.0.0.1:{self.port}")}\n'
                  f'log_dir = {json.dumps(str(self.log_dir))}\n'
                  f'state_dir = {json.dumps(str(self.state_dir))}\n')
        self.config_path.write_text(prefix + config)
        if self.tool_call_hook_command:
            command = ', '.join(json.dumps(item) for item in self.tool_call_hook_command)
            with self.config_path.open('a') as sink:
                sink.write('\n[[hooks]]\nname = "tool-cycle-image"\non = ["tool_call"]\n')
                sink.write(f'command = [{command}]\ntimeout_secs = 15\n')
        self.base_url = f'http://127.0.0.1:{self.port}'
        self._server = subprocess.Popen([str(self.ashkelon_path), 'serve', '--config', str(self.config_path)],
                                        cwd=self.cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._server.poll() is not None:
                self.close()
                raise RuntimeError(f'Ashkelon exited during startup (code {self._server.returncode})')
            try:
                with socket.create_connection(('127.0.0.1', self.port), timeout=0.2):
                    self._active = True
                    return self
            except OSError:
                time.sleep(0.05)
        self.close()
        raise TimeoutError('Ashkelon did not start before the startup deadline')

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        if self._server is not None and self._server.poll() is None:
            self._server.terminate()
            try:
                self._server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server.kill()
                self._server.wait()
        self._server = None
        self._active = False
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def _log_paths(self, prefix: str) -> list[Path]:
        return sorted(self.log_dir.glob(prefix + '-*.jsonl'))

    def _matching_hook_rows(self, session_id: str, event: str = 'turn_end') -> list[dict]:
        return [row for row in _jsonl(self._log_paths('hooks'))
                if _session_matches(row, session_id) and row.get('event') == event]

    def _matching_hook_count(self, session_id: str, event: str = 'turn_end') -> int:
        return len(self._matching_hook_rows(session_id, event))

    def _wait_hook(self, session_id: str, baseline: int, event: str = 'turn_end'):
        deadline = time.monotonic() + self.hook_timeout
        while time.monotonic() < deadline:
            matches = self._matching_hook_rows(session_id, event)
            if len(matches) > baseline:
                # Hook logs do not include provider call/event IDs. A CLI call
                # may contain multiple completed responses, so these statuses
                # are diagnostics and cannot be attributed by ordinal alone.
                return [row.get('status') for row in matches[baseline:]]
            time.sleep(0.05)
        raise TimeoutError(f'no new {event} hook completion recorded for session {session_id}')

    def _codex_args(self, prompt: str, answer_path: Path, session_id: str | None):
        provider = '{name="ashkelon",base_url=' + json.dumps(self.base_url + '/chatgpt/backend-api/codex') + ',wire_api="responses",requires_openai_auth=true}'
        args = ['codex', 'exec', '--ignore-user-config', '-m', self.model,
                '-c', 'project_doc_max_bytes=0',
                '-c', f'model_providers.ashkelon={provider}', '-c', 'model_provider="ashkelon"']
        if self.allow_shell_tool:
            args += ['--sandbox', 'workspace-write']
        if session_id:
            args += ['resume', '--skip-git-repo-check', '--json', '-o', str(answer_path), session_id, prompt]
        else:
            args += ['--skip-git-repo-check', '--json', '-o', str(answer_path), prompt]
        return args

    def _claude_args(self, prompt: str, session_id: str | None):
        args = ['claude', '--safe-mode', '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}',
                '--disable-slash-commands', '--tools', 'Bash' if self.allow_shell_tool else '', '--model', self.model,
                '--output-format', 'json']
        if self.allow_shell_tool:
            args += ['--allowedTools', 'Bash(printf *)']
        if session_id:
            args += ['--resume', session_id]
        else:
            args += ['--session-id', str(uuid.uuid4())]
        args += ['-p', prompt]
        return args

    def turn(self, provider: str, prompt: str, session_id: str | None = None, *,
             require_agentanyl_trace: bool = True) -> TurnResult:
        if not self._active:
            raise RuntimeError('enter the harness context before calling turn')
        provider = provider.lower()
        if provider not in ('claude', 'codex'):
            raise ValueError('provider must be claude or codex')
        if not isinstance(prompt, str):
            raise TypeError('prompt must be a string')
        with _PROVIDER_LOCK:
            call_rows = _jsonl(self._log_paths('calls'))
            wire = 'openai_responses' if provider == 'codex' else 'anthropic_messages'
            baseline_calls = sum(1 for row in call_rows if _session_matches(row, session_id) and row.get('wire') == wire) if session_id else 0
            before_hooks = self._matching_hook_count(session_id) if session_id else 0
            before_traces = self._trace_count(session_id) if session_id else 0
            answer_path = self.root / f'answer-{uuid.uuid4()}.txt'
            env = os.environ.copy()
            if provider == 'claude':
                env['ANTHROPIC_BASE_URL'] = self.base_url + '/anthropic'
                args = self._claude_args(prompt, session_id)
            else:
                args = self._codex_args(prompt, answer_path, session_id)
            # Each turn runs in a fresh empty cwd, retaining only its own CLI session ID.
            try:
                result = subprocess.run(args, cwd=self.cwd, env=env, text=True, capture_output=True,
                                        timeout=self.provider_timeout)
            except subprocess.TimeoutExpired as exc:
                stderr = exc.stderr.decode(errors='replace') if isinstance(exc.stderr, bytes) else (exc.stderr or '')
                diagnostic = self._safe_diagnostic(stderr, env)
                raise TimeoutError(f'{provider} exceeded provider timeout; sanitized stderr: {diagnostic}') from None
            if result.returncode:
                diagnostic = self._safe_diagnostic(result.stderr, env)
                raise RuntimeError(f'{provider} exited with code {result.returncode}; provider call was not retried; '
                                   f'sanitized stderr: {diagnostic}')
            try:
                answer, returned_session, usage = self._decode_provider(provider, result, answer_path, session_id)
            except RuntimeError as exc:
                diagnostic = self._safe_diagnostic(result.stderr, env)
                raise RuntimeError(f'{exc}; sanitized stderr: {diagnostic}') from None
            if session_id and returned_session != session_id:
                diagnostic = self._safe_diagnostic(result.stderr, env)
                raise RuntimeError(f'{provider} resumed as a different session; provider call was not retried; '
                                   f'sanitized stderr: {diagnostic}')
            hook_statuses = self._wait_hook(returned_session, before_hooks)
            all_calls = _jsonl(self._log_paths('calls'))
            matched_calls = [row for row in all_calls if _session_matches(row, returned_session) and row.get('wire') == wire]
            if session_id and len(matched_calls) <= baseline_calls:
                raise RuntimeError('provider completed without a new matching relay call record')
            if not matched_calls:
                raise RuntimeError('provider completed without a matching relay call record')
            call = matched_calls[-1] if matched_calls else {}
            new_calls = matched_calls[baseline_calls:] if session_id else matched_calls
            call_summaries = [{key: row.get(key) for key in
                               ('call_id', 'wire', 'status', 'model', 'tool_calls', 'turn_end', 'pings_injected')}
                              for row in new_calls]
            trace, trace_candidate_ids, trace_prompt_matches = self._matching_trace(
                returned_session, before_traces, answer, prompt)
            if trace is None and require_agentanyl_trace:
                raise RuntimeError('provider completed without a controller trace matching its final answer')
            # The Agentanyl trace is tied to the exact observed final answer;
            # the Ashkelon hook log has no event ID to associate raw statuses.
            logical_hook_status = trace['delivery'].get('status') if trace else None
            return TurnResult(answer=answer, session_id=returned_session, model=self.model,
                              http_status=call.get('status'), call_id=call.get('call_id'),
                              pings_injected=call.get('pings_injected'), token_usage=usage or call.get('usage', {}),
                              hook_status=logical_hook_status, controller_trace=trace,
                              observed_model=call.get('model'), ashkelon_hook_statuses=hook_statuses,
                              trace_candidate_ids=trace_candidate_ids,
                              selected_trace_prompt_matches=trace_prompt_matches,
                              provider_calls=call_summaries,
                              ashkelon_binary=self.ashkelon_binary)

    @staticmethod
    def _safe_diagnostic(stderr: str, env: dict[str, str]) -> str:
        text = _ANSI_RE.sub('', stderr or '')
        for key, value in env.items():
            if len(value) >= 6 and any(part in key.upper() for part in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
                text = text.replace(value, '[redacted]')
        text = _BEARER_RE.sub('Bearer [redacted]', text)
        text = _API_TOKEN_RE.sub('[redacted]', text)
        text = _CREDENTIAL_RE.sub(lambda match: f'{match.group(1)}=[redacted]', text)
        text = text.strip()
        return text[-1200:] if text else '(empty)'

    @staticmethod
    def _decode_provider(provider: str, completed: subprocess.CompletedProcess, answer_path: Path, prior_session: str | None):
        if provider == 'codex':
            match = _SESSION_RE.search(completed.stderr + '\n' + completed.stdout)
            # --json emits session_meta events; support these without depending on CLI prose.
            events = []
            for line in completed.stdout.splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            for event in events:
                if event.get('type') == 'thread.started':
                    match_id = event.get('thread_id')
                    if match_id:
                        session_id = str(match_id)
                        break
            else:
                session_id = match.group(1) if match else None
            if not session_id or not answer_path.exists():
                raise RuntimeError('Codex completed without a session ID or answer file')
            usage = {}
            for event in events:
                if event.get('type') == 'turn.completed':
                    usage = event.get('usage', {}) or {}
            return answer_path.read_text().strip(), session_id, usage
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            raise RuntimeError('Claude completed with invalid JSON output') from None
        session_id = payload.get('session_id')
        if not session_id:
            raise RuntimeError('Claude completed without a session ID')
        return str(payload.get('result', '')), session_id, payload.get('usage', {}) or {}

    def _trace_rows(self, session_id: str):
        import sqlite3
        if not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as db:
            rows = list(db.execute('SELECT event_id, request, response, decision, previous, current, delivery '
                                   'FROM trace WHERE session LIKE ? ORDER BY ts', ('%' + session_id,)))
        return rows

    def _trace_count(self, session_id: str):
        return len(self._trace_rows(session_id))

    def _matching_trace(self, session_id: str, baseline: int, answer: str, prompt: str):
        rows = self._trace_rows(session_id)
        candidates = []
        for row in rows[baseline:]:
            event_id, request, response, decision, previous, current, delivery = row
            request_obj = json.loads(request) if isinstance(request, str) else request
            state = request_obj.get('state', {})
            candidates.append({
                'row': row,
                'answer_matches': state.get('observation', '').strip() == answer.strip(),
                'prompt_matches': state.get('prompt') == prompt,
            })
        answer_candidates = [candidate for candidate in candidates if candidate['answer_matches']]
        candidate_ids = [candidate['row'][0] for candidate in candidates]
        if not answer_candidates:
            return None, candidate_ids, None
        selected = answer_candidates[-1]
        event_id, request, response, decision, previous, current, delivery = selected['row']
        trace = {'event_id': event_id, 'request': json.loads(request), 'response': json.loads(response),
                 'decision': decision, 'previous': json.loads(previous), 'current': json.loads(current),
                 'delivery': json.loads(delivery)}
        return trace, candidate_ids, selected['prompt_matches']
