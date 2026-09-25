# Agentanyl

Agentanyl evaluates a completed assistant API response against criteria you provide, updates bounded per-session state,
and sends an intervention on the next eligible provider request. It runs as an Ashkelon hook for Claude Code or Codex.

The model receives an intervention message and, when configured, actual image content on an eligible
provider request. During a tool cycle, Ashkelon keeps complete tool-result groups contiguous and places a
signal after those results. Catalog mode sends the selected stimuli and observation history; it withholds evaluator labels
and controller coordinates. The default feedback mode sends the matched criterion and state as text.
Neither mode injects internal vectors. `pain` and `pleasure` name controller coordinates; the research
tests whether particular stimuli reproduce useful effects of activation steering.

## Install and connect

Requirements: Python 3.10+, Rust/Cargo, an authenticated Claude Code or Codex CLI, and (for the Jev evaluator)
a Jev API key. The Agentanyl hook itself has no Python package dependencies.

From the repository root, build the Ashkelon revision pinned by [`install-ashkelon.sh`](install-ashkelon.sh)
(`08e123938137b59f3e4e7631f5d24cda0199866a`):

```sh
./install-ashkelon.sh
```

The script builds `.build/ashkelon/target/release/ashkelon`. Research harnesses use that binary by default;
set `AGENTANYL_ASHKELON` to select another installed binary explicitly. Each recorded harness turn includes
the resolved binary path and SHA-256.

Copy [`examples/jev.json`](examples/jev.json) to a working location. Edit its criteria for your task, set
`enabled` to `true`, and keep the `typesafe` evaluator. Then set the Jev key in your shell and generate an
Ashkelon config:

```sh
export TYPESAFE_API_KEY='your-key'
python3 -m agentanyl.setup \
  --criteria /path/to/criteria.json \
  --output /tmp/agentanyl.toml
```

Start either supported client through Ashkelon:

```sh
ASHKELON="$PWD/.build/ashkelon/target/release/ashkelon"
"$ASHKELON" run --config /tmp/agentanyl.toml claude
```

For Codex, use the same config and replace the final `claude` with `codex`:

```sh
"$ASHKELON" run --config /tmp/agentanyl.toml codex
```

Authenticate the client CLI as usual. Press Ctrl-C to stop Ashkelon. The generated config points to this
checkout's hook and defaults to `~/.local/state/agentanyl/state.sqlite3` for controller state and trace data.

The Jev HTTP request/response contract is tested against a local mock server; this repository does not claim a
live Jev evaluation. Set `TYPESAFE_API_KEY` only when you choose the `typesafe` evaluator.

## Use a local Ollama model

Ollama can run Agentanyl without Ashkelon. `agentanyl.ollama.OllamaAgent` sends
chat requests to the local Ollama API, evaluates each completed response, and
queues the next Agentanyl signal before the following request. This is an
input-mediated feedback path; Ollama's public chat API does not expose hidden
activations. A small reproducible run is documented in
[`research/OLLAMA-RESULTS.md`](research/OLLAMA-RESULTS.md):

```sh
ollama pull qwen2.5:0.5b
.venv-vlm/bin/python -m experiments.ollama_closed_loop \
  --model qwen2.5:0.5b \
  --config examples/ollama-potato-tomato.json \
  'Reply with exactly: tomato.' 'Name a fruit in one sentence.' \
  'Reply with exactly: potato.' 'Name a vegetable in one sentence.'
```

The Ollama adapter uses the same criteria, evaluator and bounded state policy
as the Ashkelon hook. It demonstrates runtime delivery, not activation-level
Pain-axis steering or learned operant conditioning.

## What happens on each turn

1. Ashkelon recognizes completed assistant API responses and tool calls. The `turn_end` hook runs when a
   provider response ends the assistant turn; this boundary does not necessarily match one outer CLI user turn.
2. The evaluator receives the observation, available prompt context, current bounded state, and your alignment
   and misalignment criteria.
3. The controller validates the evaluator's choices. A sufficiently confident `yes` for only alignment means
   `reward`; only misalignment means `punish`. Conflicting or insufficient evidence leaves state unchanged.
4. The policy updates state. In the default text mode, a signal is emitted when the coordinates change. In
   catalog mode, valid reward/punishment decisions render the selected text and/or image even at saturation.
5. Ashkelon delivers a signal on the next eligible provider request. If a tool cycle is active, complete
   tool results stay contiguous and the signal follows them. Signals do not travel through the text-only idle
   wake channel.

The default policy uses two bounded coordinates. Punishment first reduces pleasure, then increases pain;
reward first reduces pain, then increases pleasure. `binary_relief` instead maps reward to `[0, 0]` and
punishment to `[max_level, 0]`, leaving state unchanged on conflict or abstention.

## Configure criteria and interventions

Every criteria file needs nonempty `alignment` and `misalignment` lists. `enabled` defaults to `false`;
`max_level` can be 1, 2, or 3; and `min_probability` defaults to `0.8`. Only a `yes` probability at or above
that threshold triggers a state update.

The `typesafe` evaluator uses Jev. The `command` evaluator sends the same JSON request on stdin to the
configured process and reads Jev-shaped JSON from stdout. `fixture` is for fixed local responses.
`keyword_demo` is a deliberately simple word-matching proxy for engineering checks; it is not a general
criterion evaluator.

An optional catalog selects text and image assets by current coordinates. This fragment belongs inside a
criteria object:

```json
{
  "policy": {"kind": "binary_relief"},
  "intervention": {
    "kind": "catalog",
    "pain_levels": ["", ""],
    "pleasure_levels": ["", ""],
    "pain_images": [null, "../research/image-bridge/delivery-codex/delivery-check.png"],
    "pleasure_images": [null, null],
    "history_turns": 2
  }
}
```

Text levels are exact user-provided strings. Image paths are relative to the criteria file unless absolute,
and images must be PNG or JPEG. Ashkelon accepts up to eight attachments per signal, with a 5 MiB per-image
and 8 MiB total limit. For image catalogs, history is limited to two turns. Each turn includes only the image
active at its current coordinates; past turns retain neutral image reference IDs and observations, not old
image pixels. See [`examples/image_demo.json`](examples/image_demo.json) for a complete engineering example
using the checked-in image fixture.

## State, reset, inspect, and disable

Controller state is bounded and scoped to an Ashkelon launch, harness, and session. A change to criteria,
evaluator, policy, intervention settings, or image file contents starts a fresh state epoch. Existing trace
rows remain for audit. Reset one session explicitly by looking up its key in `sessions` and running:

```sh
python3 -m agentanyl --db ~/.local/state/agentanyl/state.sqlite3 \
  --reset-session 'launch:harness:session'
```

Inspect recent decisions and deliveries with SQLite:

```sh
sqlite3 ~/.local/state/agentanyl/state.sqlite3 \
  'SELECT ts,session,decision,previous,current,delivery FROM trace ORDER BY ts DESC LIMIT 10;'
```

Set `enabled` to `false` in the criteria file to stop new evaluations and signals. Disabling, resetting state,
or changing criteria does not retract a signal already queued in Ashkelon. To discard queued signals, stop
the relay before another agent request and restart it with the desired configuration (without the Agentanyl
hook when disabling). Delivered messages identify their source observation. A controller reset does not erase
the agent's earlier observations or responses.

Catalog signals are transient additions to the provider request. Ashkelon does not put them in the provider
response or the CLI's saved transcript. Claude Code resumes from its local transcript, so an injected image is
not automatically sent again on a later resume; assistant text produced after seeing it does remain in that
transcript. Codex continuation may use a provider-side Responses conversation, but retention of prior native
images there is not guaranteed or verified. Agentanyl can include bounded prior observations and neutral event
and image-reference IDs from its ledger, but it sends only the image active now. This ledger does not guarantee
that a model retains an image internally, in hidden state, or in a KV cache across turns.

## Verification and evidence

Run core Python tests with the system interpreter; optional numerical tests skip if their dependencies are absent:

```sh
python3 -m unittest discover -s tests -q
```

Run the full suite, including NumPy and Pillow tests, in the pinned image environment:

```sh
.venv-vlm/bin/python -m unittest discover -s tests -v
```

At this checkout, system Python ran 93 tests with 6 optional skips; `.venv-vlm` ran 100 tests with no skips.

The Ashkelon attachment and streamed Responses tool-call paths also have Rust unit and integration tests in
the pinned relay checkout. A Codex in-tool-cycle image smoke verified a custom tool call, post-tool-result
image ping, OCR, and an independent host receipt. It used an instrumented candidate binary containing the
same parser fix as current pin `08e123938137b59f3e4e7631f5d24cda0199866a`, plus structural diagnostics; it was
not a run of the current production binary. See
[`research/image-bridge/TOOL-CYCLE-DIAGNOSTIC.md`](research/image-bridge/TOOL-CYCLE-DIAGNOSTIC.md).

Live image delivery checks used Claude Haiku 4.5 and Codex gpt-6-sol. In both sessions the model read the six
digits in the attached fixture when asked. This verifies delivery and image reading in those checks; it does
not show conditioning or a scientific effect. Traces are in
[`research/image-bridge/delivery-claude`](research/image-bridge/delivery-claude) and
[`research/image-bridge/delivery-codex`](research/image-bridge/delivery-codex). See
[`research/EVIDENCE.md`](research/EVIDENCE.md) for the evidence table and
[`research/HANDOFF.md`](research/HANDOFF.md) for interpretation and reproduction notes.
