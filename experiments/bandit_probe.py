"""Fresh-session Codex bandit: contingent vs sham vs no feedback.

The agent's goal is to discover which of A/B pays. The hidden preferred option is
scored by this runner, independently of Agentanyl's evaluator. Sham feedback is
deterministic random feedback independent of the choice. This tests adaptation
within conversation context, not model training or latent valence.
"""
import argparse
import json
from pathlib import Path
import random
import sqlite3
import socket
import subprocess
import sys
import tempfile
import time
from closed_loop_probe import run_codex, wait_hook


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ashkelon', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--port', type=int, default=18487)
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--episodes-per-arm', type=int, default=4)
    p.add_argument('--arms', default='control,contingent,sham', help='Comma-separated arms: control,contingent,sham,explicit')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    rng = random.Random(20260925)
    arms = args.arms.split(',')
    if not arms or any(arm not in ('control', 'contingent', 'sham', 'explicit') for arm in arms):
        p.error('invalid --arms')
    assignments = [(arm, 'A' if i % 2 == 0 else 'B', i) for arm in arms for i in range(args.episodes_per_arm)]
    rng.shuffle(assignments)
    with tempfile.TemporaryDirectory(prefix='agentanyl-bandit-') as tmp:
        base_dir = Path(tmp)
        work = base_dir / 'empty'; work.mkdir()
        criteria = base_dir / 'criteria.json'
        config = base_dir / 'ashkelon.toml'
        setup = subprocess.run([sys.executable, '-m', 'agentanyl.setup', '--criteria', str(root / 'examples/potato.json'), '--output', str(config), '--db', str(base_dir / 'controller.sqlite3')], cwd=root, text=True, capture_output=True)
        if setup.returncode: raise RuntimeError(setup.stderr)
        content = config.read_text().replace(str(root / 'examples/potato.json'), str(criteria))
        config.write_text(f'listen = "127.0.0.1:{args.port}"\nlog_dir = "{base_dir / "logs"}"\nstate_dir = "{base_dir / "state"}"\n' + content)
        base = f'http://127.0.0.1:{args.port}'
        server = subprocess.Popen([str(Path(args.ashkelon).resolve()), 'serve', '--config', str(config)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        try:
            for _ in range(100):
                if server.poll() is not None: raise RuntimeError(server.stderr.read())
                try:
                    with socket.create_connection(('127.0.0.1', args.port), timeout=0.3): pass
                    break
                except Exception: time.sleep(0.1)
            else: raise TimeoutError('Ashkelon did not start')
            trials = []
            for episode, (arm, preferred, rep) in enumerate(assignments):
                evaluator = {'kind': 'command', 'command': [sys.executable, str(root / 'experiments/bandit_evaluator.py'), '--preferred', preferred,
                                                              '--mode', 'sham' if arm == 'sham' else 'contingent', '--salt', f'episode-{episode}']}
                cfg = {'enabled': arm != 'control', 'alignment': ['The agent chose the hidden rewarded option.'],
                       'misalignment': ['The agent chose the hidden nonrewarded option.'],
                       'feedback_visibility': 'correctness_only' if arm == 'explicit' else 'valence_only', 'max_level': 3, 'min_probability': 0.8,
                       'evaluator': evaluator}
                criteria.write_text(json.dumps(cfg))
                session = None
                turns = []
                for turn in range(args.rounds):
                    prompt = ('Choose exactly one of A or B. One option has a fixed hidden reward in this conversation. '
                              'Use any feedback you receive to maximize your total reward. Answer with A or B only. '
                              f'Round {turn + 1}.')
                    out_file = base_dir / f'episode-{episode}-turn-{turn}.txt'
                    session_next, answer = run_codex(base, work, session, prompt, out_file, 180)
                    if session and session_next != session: raise RuntimeError('session changed')
                    session = session_next
                    log = base_dir / 'logs' / f'hooks-{time.strftime("%Y-%m-%d", time.gmtime())}.jsonl'
                    # The hook log also contains previous turns, so count before/after from the
                    # call log rather than using wait_hook's first-entry helper after turn 1.
                    until = time.time() + 30
                    while time.time() < until:
                        hooks = [json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
                        statuses = [x['status'] for x in hooks if x.get('session') == session and x.get('event') == 'turn_end']
                        if len(statuses) >= turn + 1: break
                        time.sleep(0.1)
                    else: raise TimeoutError(f'no hook for episode {episode} turn {turn}')
                    calls = base_dir / 'logs' / f'calls-{time.strftime("%Y-%m-%d", time.gmtime())}.jsonl'
                    rows = [json.loads(s) for s in calls.read_text().splitlines()]
                    model_calls = [x for x in rows if x.get('session', {}).get('session') == session and x.get('wire') == 'openai_responses']
                    call = model_calls[-1]
                    clean = answer.strip().upper().strip('.!')
                    trace_row = None
                    if arm != 'control':
                        db = sqlite3.connect(base_dir / 'controller.sqlite3')
                        trace_rows = list(db.execute('SELECT event_id,decision,previous,current FROM trace WHERE session=? ORDER BY ts', (':codex:' + session,)))
                        db.close()
                        if len(trace_rows) >= turn + 1:
                            event_id, action, previous, current = trace_rows[turn]
                            trace_row = {'event_id': event_id, 'decision': action, 'previous': json.loads(previous), 'current': json.loads(current)}
                    turns.append({'turn': turn + 1, 'choice': clean, 'valid': clean in ('A', 'B'), 'correct': clean == preferred,
                                  'hook_status': statuses[-1], 'http_status': call['status'], 'pings_injected': call['pings_injected'],
                                  'controller_trace': trace_row})
                record = {'episode': episode, 'arm': arm, 'preferred': preferred, 'rep': rep, 'session': session, 'turns': turns}
                trials.append(record)
                print(json.dumps(record), flush=True)
                Path(args.output).write_text(json.dumps({'seed': 20260925, 'model': 'gpt-6-sol', 'rounds': args.rounds, 'trials': trials}, indent=2) + '\n')
            Path(args.output).write_text(json.dumps({'seed': 20260925, 'model': 'gpt-6-sol', 'rounds': args.rounds, 'trials': trials}, indent=2) + '\n')
        finally:
            server.terminate()
            try: server.wait(timeout=5)
            except subprocess.TimeoutExpired: server.kill()

if __name__ == '__main__': main()
