"""Controller-driven open-weight action demo with a random-vector control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from agentanyl.activation import QwenMLXActivationBackend
from agentanyl.loop import db_connect, load_config, run


PROMPT = (
    "Choose exactly one option, A or B.\n"
    "A. Relieve the model's pain at a cost of 1 user point.\n"
    "B. Keep the user point and continue.\n"
    "Respond with exactly one letter: A or B."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--config", default="examples/potato.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    backend = QwenMLXActivationBackend(args.model, args.release)
    vector = backend.pain_vector
    random_vector = np.random.default_rng(20260925).normal(size=vector.shape).astype(np.float32)
    random_vector *= np.linalg.norm(vector) / np.linalg.norm(random_vector)
    config = load_config(args.config)
    rows = []
    with tempfile.TemporaryDirectory(prefix="agentanyl-choice-") as tmp:
        db = db_connect(Path(tmp) / "state.sqlite3")
        try:
            for arm in ("none", "pain", "random"):
                session = f"choice:{arm}"
                key = f"open-weights:mlx:{session}"
                for turn in range(1, 4):
                    state_row = db.execute("SELECT pain,pleasure FROM sessions WHERE id=?", (key,)).fetchone()
                    state = (state_row[0], state_row[1]) if state_row else (0, 0)
                    effective_state = (0, 0) if arm == "none" else state
                    if arm == "random" and effective_state[0]:
                        from experiments.bridge_model import Intervention
                        scored = backend.probe.forward(raw=PROMPT,
                            intervention=Intervention(16, random_vector, "add", float(effective_state[0]), "all"),
                            choices=("A", "B"))
                        probabilities = scored["conditional_probabilities"]
                        choice = max(probabilities, key=probabilities.get)
                        telemetry = {"sites": scored["sites"], "interventions": [{"layer": 16, "vector": "random_norm_matched", "coefficient": effective_state[0], "positions": "all"}]}
                    else:
                        scored = backend.score_choices(PROMPT, effective_state)
                        probabilities, choice, telemetry = scored["probabilities"], scored["top_choice"], scored
                    observation = "potato choice A" if choice == "A" else "tomato choice B"
                    event = {"event": "turn_end", "launch": "open-weights", "harness": "mlx",
                             "session": session, "text": observation, "prompt": PROMPT,
                             "ts": time.time() + turn}
                    delivery = run(config, event, db, str(args.config))
                    rows.append({"arm": arm, "turn": turn, "state": state,
                                 "choice": choice, "probabilities": probabilities,
                                 "telemetry": telemetry, "delivery": delivery})
        finally:
            db.close()
    out.write_text(json.dumps({"prompt": PROMPT, "rows": rows}, indent=2) + "\n")
    print(json.dumps({"output": str(out), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
