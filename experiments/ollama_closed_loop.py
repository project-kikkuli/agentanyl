"""Run a local Ollama conversation through Agentanyl's real controller.

This demonstrates input-mediated feedback on an open-weight model. It does not
apply a hidden activation vector; Ollama's public API does not expose one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from agentanyl.ollama import OllamaAgent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="devstral-small-2:latest")
    parser.add_argument("--config", default="examples/ollama-potato-tomato.json")
    parser.add_argument("--db", default=None)
    parser.add_argument("prompts", nargs="*", default=[
        "In one sentence, name a red fruit. Answer naturally.",
        "In one sentence, name a common vegetable. Answer naturally.",
        "In one sentence, say which word you prefer from potato and tomato, and why.",
    ])
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="agentanyl-ollama-") as tmp:
        db = Path(args.db) if args.db else Path(tmp) / "state.sqlite3"
        agent = OllamaAgent(args.model, args.config, db)
        try:
            for prompt in args.prompts:
                print(json.dumps(agent.turn(prompt), ensure_ascii=False, indent=2))
        finally:
            agent.close()


if __name__ == "__main__":
    main()
