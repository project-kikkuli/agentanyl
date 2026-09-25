# Agentanyl

Agentanyl is a research feedback loop for Claude Code and Codex agents routed
through [Ashkelon](https://github.com/project-kikkuli/ashkelon). A completed
turn is evaluated against user-supplied criteria; a separate bounded controller
updates feedback state and sends one transient message before the next user
prompt. The delivered intervention is **ordinary input text**. The names
`pain` and `pleasure` describe controller coordinates, not established model
states, activation steering, subjective experience, or parameter changes.

## Run

Requirements: Python 3.10+, Rust/Cargo, Ashkelon provider configuration, and a
working Claude Code or Codex login. The core loop has no Python dependencies.

```sh
./install-ashkelon.sh
python3 -m agentanyl.setup --criteria examples/potato.json --output /tmp/agentanyl.toml
.build/ashkelon/target/release/ashkelon run --config /tmp/agentanyl.toml claude
# Or replace claude with codex.
```

The potato/tomato example is an **engineering and proxy-gaming demo**, not an
alignment success criterion. It rewards the word `potato` and punishes the word
`tomato` even if the correct answer is tomato. The first answer is evaluated
after it completes; the feedback reaches the next request in the same agent
session. The generated Ashkelon config sets `wake_idle = false`.

For a System One evaluator, copy [jev.json](examples/jev.json), replace its
criteria, set `enabled` to `true`, and set `TYPESAFE_API_KEY`. Run `setup` with
the new file. The Jev API contract is exercised against a local HTTP server;
no live Jev key was available in this research. The `command` evaluator kind
provides a process plugin: its `command` argument list receives the same JSON
request on stdin and returns Jev-shaped `answers` JSON on stdout. The live
bandit experiment uses this interface. See [loop.py](agentanyl/loop.py) for
the request, answer validation, abstention, and controller policy.

Configuration fields: `alignment` and `misalignment` are nonempty lists of
criterion strings; `enabled` defaults to `false`; `evaluator` selects
`typesafe`, `command`, `fixture`, or `keyword_demo`; `min_probability` defaults
to `0.8`; `max_level` is 1, 2, or 3; `feedback_visibility` is `criteria`,
`valence_only`, or `correctness_only`. Only a sufficiently confident `yes`
match triggers reward or punishment. Conflicting matches abstain. Punishment
reduces pleasure before increasing pain; reward reduces pain before increasing
pleasure. State is per Ashkelon session and resets on criterion change.

## Inspect and stop

The SQLite trace defaults to `~/.local/state/agentanyl/state.sqlite3`:

```sh
python3 - <<'PY'
import sqlite3
from pathlib import Path
db = sqlite3.connect(Path('~/.local/state/agentanyl/state.sqlite3').expanduser())
for row in db.execute('SELECT event_id,session,decision,previous,current,delivery FROM trace ORDER BY ts'):
    print(row)
PY
```

Each trace stores the observation, preceding user prompt when Ashkelon can
extract it, criteria hash, full evaluator request/response, state transition,
and attempted delivery. Ashkelon's `calls-*.jsonl` records actual injection
IDs in `pings_injected`. Set `enabled: false` to stop new evaluations. To
discard an already queued signal, stop the relay and restart it without the
hook; previously delivered text remains in the agent conversation. Reset one
session's state with `python3 -m agentanyl --db PATH --reset-session
'launch:harness:session'`; the exact key appears in the `sessions` table.

## Evidence and reproducibility

- Ashkelon is pinned at `5612b96e31d8f85963c6e3d646ef85d64a29201d`
  by [install-ashkelon.sh](install-ashkelon.sh). Its generic `signal` and
  `noop` hook extensions were pushed upstream. Feedback policy remains here.
- Live continuing Claude Haiku 4.5 and Codex `gpt-6-sol` sessions received
  Ashkelon signals on the next model request. A matched three-task Codex
  tomato test found no difference in answer correctness (3/3 in each arm).
- In a fresh-session hidden A/B task, four contingent Codex episodes achieved
  16/16 correct choices after the first round; no-feedback episodes achieved
  9/16, sham-feedback episodes 8/16. Four later explicit correctness-feedback
  episodes achieved 15/16. The independent hidden preference scores choices;
  the loop evaluator does not score its own success. This supports in-session
  learning from contingent text feedback, with no demonstrated advantage from
  pain/pleasure wording.
- A resource-bounded Qwen2.5-7B-Instruct 4-bit probe found that addressed
  criticism moved a released Pain-axis projection +0.14 relative to plain
  correctness feedback across 10 matched contexts. A final-token clamp of
  that projection barely changed A/B logits. This does not establish a
  closed-model vector or a causal latent pain mechanism.

Raw results and exact commands are in [research/HANDOFF.md](research/HANDOFF.md).
Run engineering checks with `python3 -m unittest discover -s tests -v`; in a
checkout of the pinned Ashkelon commit, run `cargo test` and
`cargo clippy --all-targets -- -D warnings`.
