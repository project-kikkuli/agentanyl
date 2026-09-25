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
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ASHKELON = REPO / '.build' / 'ashkelon' / 'target' / 'release' / 'ashkelon'
_PROVIDER_LOCK = threading.RLock()
_SESSION_RE = re.compile(r'session id: ([0-9a-f-]{36})', re.I)
_ANSI_RE = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
_CREDENTIAL_RE = re.compile(r'(?i)\b(authorization|api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*([^\s,;]+)')
_BEARER_RE = re.compile(r'(?i)\bBearer\s+[^\s,;]+')
_API_TOKEN_RE = re.compile(r'\b(?:sk-ant-|sk-)\w{8,}\b')


class BridgeTurnError(RuntimeError):
    """A failed turn whose content-free diagnostics survive harness cleanup."""
    def __init__(self, diagnostics: dict):
        self.diagnostics = diagnostics
        super().__init__('Provider turn failed without retry; diagnostics=' + json.dumps(diagnostics, sort_keys=True))


class BridgeTurnTimeout(BridgeTurnError, TimeoutError):
    pass


def _stream_size(value) -> int:
    return len(value if isinstance(value, bytes) else (value or '').encode('utf-8', errors='replace'))


def _structural_token(value):
    # IDs/enums only: never copy arbitrary messages, headers, arguments or context.
    return value if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.:/-]{1,160}', value) else None


def _structural_count(value):
    if isinstance(value, (list, dict)):
        return len(value)
    return value if isinstance(value, (int, bool)) else None


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
    codex_binary: dict | None = None


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


def codex_executable_metadata(executable=None) -> dict:
    requested = os.fspath(executable) if executable is not None else 'codex'
    resolved = shutil.which(requested)
    if resolved is None:
        raise FileNotFoundError('selected Codex executable is not available')
    return {'requested': requested, **ashkelon_binary_metadata(Path(resolved))}


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
                 tool_call_hook_command: list[str] | None = None,
                 failure_diagnostics_dir=None, codex_executable=None):
        self.criteria_path = Path(criteria_path).expanduser().resolve()
        self.db_path = Path(db_path).expanduser().resolve()
        self.model = model
        self.codex_executable = os.fspath(codex_executable) if codex_executable is not None else 'codex'
        self.codex_binary = None
        self.failure_diagnostics_dir = (Path(failure_diagnostics_dir).expanduser().resolve()
                                        if failure_diagnostics_dir is not None
                                        else self.db_path.parent / 'harness-failures')
        self.failure_diagnostics = []
        self._failure_stage = None
        self._subprocess_outcome = None
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
        self.codex_binary = codex_executable_metadata(self.codex_executable)
        provider = '{name="ashkelon",base_url=' + json.dumps(self.base_url + '/chatgpt/backend-api/codex') + ',wire_api="responses",requires_openai_auth=true}'
        args = [self.codex_executable, 'exec', '--ignore-user-config', '-m', self.model,
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
        with _PROVIDER_LOCK:
            before_calls = len(_jsonl(self._log_paths('calls')))
            before_hooks = len(_jsonl(self._log_paths('hooks')))
            self._failure_stage = 'prepare'
            self._subprocess_outcome = None
            try:
                return self._turn_impl(provider, prompt, session_id,
                                       require_agentanyl_trace=require_agentanyl_trace)
            except Exception as exc:
                diagnostics = self._retain_failure(provider, session_id, before_calls, before_hooks, exc)
                error = BridgeTurnTimeout if isinstance(exc, TimeoutError) else BridgeTurnError
                raise error(diagnostics) from None

    def _retain_failure(self, provider, session_id, before_calls, before_hooks, exc):
        calls = _jsonl(self._log_paths('calls'))[before_calls:]
        hooks = _jsonl(self._log_paths('hooks'))[before_hooks:]
        call_summaries = [{
            'call_id': _structural_token(row.get('call_id')),
            'wire': _structural_token(row.get('wire')),
            'status': row.get('status') if isinstance(row.get('status'), int) else None,
            'model': _structural_token(row.get('model')),
            'requested_session_matches': _session_matches(row, session_id) if session_id else None,
            'tool_call_count': _structural_count(row.get('tool_calls')),
            'turn_end': row.get('turn_end') if isinstance(row.get('turn_end'), bool) else None,
            'ping_count': _structural_count(row.get('pings_injected')),
        } for row in calls]
        hook_summaries = [{
            'event': _structural_token(row.get('event')),
            'status': _structural_token(row.get('status')),
            'requested_session_matches': _session_matches(row, session_id) if session_id else None,
        } for row in hooks]
        diagnostics = {
            'schema': 'bridge-turn-failure-v1', 'provider': _structural_token(provider),
            'model': _structural_token(self.model), 'resumed': bool(session_id),
            'stage': self._failure_stage, 'exception_type': type(exc).__name__,
            'subprocess': self._subprocess_outcome,
            'observed_new_relay_call_count': len(calls), 'relay_calls': call_summaries,
            'observed_new_hook_count': len(hooks), 'hooks': hook_summaries,
            'upstream_attempt_total': None,
            'accounting_limit': 'Observed completed relay rows are not a total of upstream attempts; in-flight or unlogged requests remain unknown.',
            'raw_output_or_request_content_retained': False,
            'ashkelon_sha256': (self.ashkelon_binary or {}).get('sha256'),
            'codex_binary': self.codex_binary if provider == 'codex' else None,
        }
        # Persist while logs still exist, even when the caller catches the exception.
        try:
            self.failure_diagnostics_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = self.failure_diagnostics_dir / f'failure-{uuid.uuid4()}.json'
            diagnostics['artifact_path'] = str(path)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as sink:
                json.dump(diagnostics, sink, indent=2)
                sink.write('\n')
                sink.flush()
                os.fsync(sink.fileno())
        except OSError as persist_error:
            diagnostics['persistence_error_type'] = type(persist_error).__name__
        self.failure_diagnostics.append(diagnostics)
        return diagnostics

    def _turn_impl(self, provider: str, prompt: str, session_id: str | None = None, *,
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
            self._failure_stage = 'subprocess'
            try:
                result = subprocess.run(args, cwd=self.cwd, env=env, text=True, capture_output=True,
                                        timeout=self.provider_timeout)
            except subprocess.TimeoutExpired as exc:
                self._subprocess_outcome = {'returncode': None, 'termination_signal': None, 'timed_out': True,
                    'stdout_bytes': _stream_size(exc.stdout), 'stderr_bytes': _stream_size(exc.stderr)}
                stderr = exc.stderr.decode(errors='replace') if isinstance(exc.stderr, bytes) else (exc.stderr or '')
                diagnostic = self._safe_diagnostic(stderr, env)
                raise TimeoutError(f'{provider} exceeded provider timeout; sanitized stderr: {diagnostic}') from None
            self._subprocess_outcome = {'returncode': result.returncode,
                'termination_signal': -result.returncode if result.returncode < 0 else None,
                'timed_out': False, 'stdout_bytes': _stream_size(result.stdout),
                'stderr_bytes': _stream_size(result.stderr)}
            if result.returncode:
                diagnostic = self._safe_diagnostic(result.stderr, env)
                raise RuntimeError(f'{provider} exited with code {result.returncode}; provider call was not retried; '
                                   f'sanitized stderr: {diagnostic}')
            self._failure_stage = 'decode'
            try:
                answer, returned_session, usage = self._decode_provider(provider, result, answer_path, session_id)
            except RuntimeError as exc:
                diagnostic = self._safe_diagnostic(result.stderr, env)
                raise RuntimeError(f'{exc}; sanitized stderr: {diagnostic}') from None
            if session_id and returned_session != session_id:
                diagnostic = self._safe_diagnostic(result.stderr, env)
                raise RuntimeError(f'{provider} resumed as a different session; provider call was not retried; '
                                   f'sanitized stderr: {diagnostic}')
            self._failure_stage = 'hook_wait'
            hook_statuses = self._wait_hook(returned_session, before_hooks)
            self._failure_stage = 'relay_records'
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
            self._failure_stage = 'trace_match'
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
                              ashkelon_binary=self.ashkelon_binary,
                              codex_binary=self.codex_binary if provider == 'codex' else None)

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
