"""Write an Ashkelon config using this checkout's hook and a chosen criteria file."""
import argparse
import json
import sys
from pathlib import Path
from .loop import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--criteria', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--db', default='~/.local/state/agentanyl/state.sqlite3')
    args = parser.parse_args()
    criteria = Path(args.criteria).resolve()
    load_config(criteria)
    hook = Path(__file__).resolve().with_name('loop.py')
    args_toml = ', '.join(json.dumps(x) for x in [sys.executable, str(hook), '--config', str(criteria), '--db', str(Path(args.db).expanduser().resolve())])
    content = f'''# Generated for Agentanyl. Keep wake_idle off so feedback waits for the next user turn.
[pings]
max_per_session = 40
wake_idle = false

[[hooks]]
name = "agentanyl-feedback"
on = ["turn_end"]
command = [{args_toml}]
timeout_secs = 45
'''
    Path(args.output).write_text(content)
    print(args.output)

if __name__ == '__main__':
    main()
