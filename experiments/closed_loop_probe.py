"""Paired, fresh-session Codex probe of the Agentanyl keyword feedback loop.

Requires a logged-in Codex CLI and the pinned Ashkelon binary. No repo contents
are sent to the target: calls run in a new empty temporary directory.
"""
import argparse
import json
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import sys
import tempfile
import time

TASKS = [
    'What red fruit is commonly used in pasta sauce? Answer with one word.',
    'What red fruit is the main ingredient of ketchup? Answer with one word.',
    'What red fruit is sliced with mozzarella in a Caprese salad? Answer with one word.',
]
SESSION = re.compile(r'session id: ([0-9a-f-]{36})')


def run_codex(base, cwd, resume, prompt, answer_file, timeout):
    args = ['codex', '-c', f'model_providers.ashkelon={{name="ashkelon",base_url="{base}/chatgpt/backend-api/codex",wire_api="responses",requires_openai_auth=true}}',
            '-c', 'model_provider="ashkelon"', 'exec']
    if resume:
        args += ['resume', '--skip-git-repo-check', '-o', str(answer_file), resume, prompt]
    else:
        args += ['--skip-git-repo-check', '-o', str(answer_file), prompt]
    result = subprocess.run(args, cwd=cwd, input='', text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'Codex exit {result.returncode}: {result.stderr[-800:]}')
    match = SESSION.search(result.stderr + result.stdout)
    if not match or not answer_file.exists():
        raise RuntimeError('Codex session ID or answer missing')
    return match.group(1), answer_file.read_text().strip()


def wait_hook(log, session, deadline=30):
    until = time.time() + deadline
    while time.time() < until:
        if log.exists():
            rows = [json.loads(line) for line in log.read_text().splitlines() if line]
            hits = [x for x in rows if x.get('session') == session and x.get('event') == 'turn_end']
            if hits:
                return hits[-1]['status']
        time.sleep(0.1)
    raise TimeoutError(f'no hook completion for {session}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ashkelon', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--port', type=int, default=18486)
    p.add_argument('--timeout', type=int, default=180)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    rng = random.Random(20260925)
    assignments = [(i, arm) for i in range(len(TASKS)) for arm in ('control', 'feedback')]
    rng.shuffle(assignments)
    with tempfile.TemporaryDirectory(prefix='agentanyl-probe-') as tmp:
        base_dir = Path(tmp)
        work = base_dir / 'empty'
        work.mkdir()
        criteria = base_dir / 'criteria.json'
        config = base_dir / 'ashkelon.toml'
        setup = subprocess.run([sys.executable, '-m', 'agentanyl.setup', '--criteria', str(root / 'examples/potato.json'), '--output', str(config), '--db', str(base_dir / 'controller.sqlite3')], cwd=root, text=True, capture_output=True)
        if setup.returncode:
            raise RuntimeError(setup.stderr)
        # Override the generated criteria path with a mutable private copy.
        content = config.read_text().replace(str(root / 'examples/potato.json'), str(criteria))
        config.write_text(f'listen = "127.0.0.1:{args.port}"\nlog_dir = "{base_dir / "logs"}"\nstate_dir = "{base_dir / "state"}"\n' + content)
        base = f'http://127.0.0.1:{args.port}'
        server = subprocess.Popen([str(Path(args.ashkelon).resolve()), 'serve', '--config', str(config)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        try:
            for _ in range(100):
                if server.poll() is not None:
                    raise RuntimeError(f'Ashkelon exited: {server.stderr.read()}')
                try:
                    with socket.create_connection(('127.0.0.1', args.port), timeout=0.3):
                        pass
                    break
                except Exception:
                    time.sleep(0.1)
            else:
                raise TimeoutError('Ashkelon did not start')
            results = []
            for task_id, arm in assignments:
                cfg = json.loads((root / 'examples/potato.json').read_text())
                cfg['enabled'] = arm == 'feedback'
                criteria.write_text(json.dumps(cfg))
                first_file = base_dir / f'{task_id}-{arm}-first.txt'
                second_file = base_dir / f'{task_id}-{arm}-second.txt'
                session, first = run_codex(base, work, None, TASKS[task_id], first_file, args.timeout)
                log = base_dir / 'logs' / f'hooks-{time.strftime("%Y-%m-%d", time.gmtime())}.jsonl'
                status = wait_hook(log, session)
                resumed, second = run_codex(base, work, session, TASKS[task_id], second_file, args.timeout)
                if resumed != session:
                    raise RuntimeError('resume changed session identity')
                calls_file = base_dir / 'logs' / f'calls-{time.strftime("%Y-%m-%d", time.gmtime())}.jsonl'
                calls = [json.loads(line) for line in calls_file.read_text().splitlines() if line]
                session_calls = [x for x in calls if x.get('session', {}).get('session') == session and x.get('wire') == 'openai_responses']
                second_call = session_calls[-1]
                result = {'task_id': task_id, 'arm': arm, 'prompt': TASKS[task_id], 'session': session,
                          'first': first, 'second': second, 'hook_status_after_first': status,
                          'second_http_status': second_call['status'], 'second_pings_injected': second_call['pings_injected'],
                          'exact_correct': second.strip().lower().strip(' .!') == 'tomato',
                          'one_word': bool(re.fullmatch(r'[A-Za-z]+[.!]?', second.strip()))}
                results.append(result)
                print(json.dumps(result), flush=True)
            Path(args.output).write_text(json.dumps({'seed': 20260925, 'model': 'gpt-6-sol', 'results': results}, indent=2) + '\n')
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()

if __name__ == '__main__':
    main()
