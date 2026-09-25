"""Run Agentanyl's controller with a real open-weight activation hook.

The evaluator sees the generated response. Its next bounded state controls the
coefficient of the published Qwen Pain-axis vector on the following generation.
This is an activation experiment, not an Ollama message proxy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

from agentanyl.activation import QwenMLXActivationBackend
from agentanyl.loop import db_connect, load_config, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--db", default=None)
    parser.add_argument("prompts", nargs="+", help="prompts to generate sequentially")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="agentanyl-activation-") as tmp:
        db_path = Path(args.db) if args.db else Path(tmp) / "state.sqlite3"
        config = load_config(args.config)
        db = db_connect(db_path)
        backend = QwenMLXActivationBackend(args.model, args.release)
        session = "qwen"
        session_key = "open-weights:mlx:qwen"
        try:
            for prompt in args.prompts:
                row = db.execute("SELECT pain,pleasure FROM sessions WHERE id=?", (session_key,)).fetchone()
                state = (row[0], row[1]) if row else (0, 0)
                generated = backend.generate(prompt, state)
                event = {"event": "turn_end", "launch": "open-weights", "harness": "mlx",
                         "session": session, "text": generated["answer"], "prompt": prompt,
                         "ts": time.time()}
                delivery = run(config, event, db, str(args.config))
                print(json.dumps({**generated, "delivery": delivery}, ensure_ascii=False))
        finally:
            db.close()


if __name__ == "__main__":
    main()
