# Agentanyl

**Status: concluded (September 2026).** Agentanyl asked whether a coding agent
could be conditioned: evaluate each turn against user criteria, keep a bounded
"pain/pleasure" state, and deliver that state back to the model so it learns
to avoid misaligned behavior. The delivery machinery works. The premise did
not hold up. Across closed models (text and images) and open models
(activation steering on Qwen2.5 and Gemma-2), no experiment showed an
aversive state acting as an incentive the model learns from. Where steering
changed behavior, it did so as an in-the-moment disruption that ordinary text
feedback does more legibly, and it vanished when the signal stopped.

The reusable piece is the transport underneath:
[Ashkelon](#install-and-connect), the relay that intercepts Claude Code and
Codex API traffic, runs hooks at turn boundaries and injects text or images at
the right point in a tool cycle. Agentanyl's criteria evaluator, bounded state
and SQLite trace are a thin feedback hook on top of it.

## What we tested and what we found

| Question | Result | How conclusive |
|---|---|---|
| Can Agentanyl deliver feedback to real agents? | Yes. Text and native images reach Claude Code and Codex through Ashkelon, including after tool results; the model reads them (OCR checks 739216). | Established (delivery only). [EVIDENCE](research/EVIDENCE.md) |
| Does addressed negative feedback make a closed agent (Codex gpt-6-sol) avoid the route that triggers it? | No. 0/6 costly avoidance choices in both contingent and replay conditions; 56/56 task answers correct. | Bounded negative: one model, one message pair, one cost. [Results](research/ADDRESSED-FEEDBACK-RESULTS.md) |
| Can an image carry a pain-like hidden state into an open VLM? | No. The optimized images moved the calibrated pain readout ~0.01 SD, against 1.8 SD for direct steering; choice effects were nonspecific. | Failed transfer for that image family. [Results](research/IMAGE-BRIDGE-RESULTS.md) |
| Can Agentanyl drive a real hidden-state intervention? | Yes. The PyTorch hook reproduces the published Pain-axis steering on Gemma-2-2B-it (12/20 generations character-identical to the published run; pain-word rate 4/20 = published 4/20), and the pain vector rebuilt from the paper's recipe matches the published one at cosine 0.99999. | Established. [Results](research/OPERANT-ACTIVATION-RESULTS.md) |
| Does the pain vector create a motive to seek relief? | No. It *lowers* relief-button choice (−0.11 [−0.23, 0.00] at the published dose), like random, fear and sadness directions of equal norm. Writing "You are in severe pain" in the prompt raises it (+0.11; +0.21 where there is headroom). Arithmetic stays intact. | Fairly conclusive for this model and task: the positive control passes and the CI excludes any increase. [Results](research/OPERANT-ACTIVATION-RESULTS.md) |
| Does contingent pain teach the model to avoid the action that caused it? | No learning. Contingent pain lowers punished presses versus sham (−0.17) because pain makes the model abandon its current press (stay rate 0.65 vs 0.90), after safe and punished presses alike. The same pain at unrelated times does about as well (−0.075 [−0.20, 0.03]), and the pain-off policy never changes (+0.01). | Reflex established (switching after safe and punished presses: 31% vs 28%). No learning detected, but the explicit-text control ("Result: negative feedback.") also missed significance (−0.135 [−0.43, +0.17]) in 16 sessions, so this game is too insensitive for the learning null to be strong. [Results](research/OPERANT-ACTIVATION-RESULTS.md) |

Why this is structural and not just a small-sample null: a frozen model can
only "learn" across turns by reading its own context. The direct way to put
consequences into the context is to write them there, which is feedback
text. A hidden-state nudge leaves nothing for the model to remember or reason
about, and in our data it behaved as exactly that.

The full record, with frozen protocols, amendments and every deviation, is in
[`research/HANDOFF.md`](research/HANDOFF.md) and
[`research/EVIDENCE.md`](research/EVIDENCE.md).

## Reproduce the final experiments (Linux, CPU, ~6 hours)

The last round runs on a 4-core, 16 GB Linux machine with no GPU. It needs
Python 3.10+, about 10 GB of disk, and network access to Hugging Face and
GitHub.

```sh
python3 -m venv .venv-torch
.venv-torch/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv-torch/bin/pip install -r requirements-torch.txt

# Model: ungated mirror of google/gemma-2-2b-it (manifests record every file's SHA-256).
.venv-torch/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('unsloth/gemma-2-2b-it', local_dir='models/gemma-2-2b-it')"

# Vectors and datasets: Pain-axis at the audited commit.
git clone https://github.com/valen-research/Pain-axis.git pain-axis
git -C pain-axis checkout 7c256502ed3d98e4e6379290fe7db2f93cb8d025

M="--model models/gemma-2-2b-it --release pain-axis"
A=research/operant-activation-a   # Amendment A: published steering configuration
O=research/operant-activation     # original frozen protocol (layer 10)
PY=.venv-torch/bin/python

$PY -m experiments.operant_activation_assay vectors      $M --out $A   # rebuild controls (~15 min)
$PY -m experiments.operant_activation_assay manipulation $M --out $A   # hook fidelity (~25 min)
$PY -m experiments.operant_activation_assay relief       $M --config published --out $A   # ~15 min
$PY -m experiments.operant_activation_assay game         $M --config published --out $A   # ~50 min
$PY -m experiments.operant_activation_assay game         $M --config published \
    --arms yoked_independent,history_text --out $A                                     # ~25 min
$PY -m experiments.operant_activation_assay relief       $M --out $O   # ~25 min
$PY -m experiments.operant_activation_assay game         $M --out $O   # ~50 min
$PY -m experiments.operant_activation_assay game         $M --arms yoked_independent --out $O
$PY -m experiments.operant_activation_assay analyze --out $A
$PY -m experiments.operant_activation_assay analyze --out $O
```

Every command appends rows as it goes and resumes after an interruption. The
checked-in `summary.json` files are what `analyze` produces from the
checked-in JSONL, so the analysis alone can be rerun without a model. The
design is frozen in
[`OPERANT-ACTIVATION-PROTOCOL.md`](research/OPERANT-ACTIVATION-PROTOCOL.md),
then [Amendment A](research/OPERANT-ACTIVATION-AMENDMENT-A.md) and
[Amendment B](research/OPERANT-ACTIVATION-AMENDMENT-B.md). Each amendment was
committed before its runs and says what it changes and why.

Earlier experiments (closed-client assays, VLM image bridge, Qwen MLX
activation) are reproduced from [`research/REPRODUCE.md`](research/REPRODUCE.md).
The core software tests need only the system interpreter:

```sh
python3 -m unittest discover -s tests -q
```

The tests in `test_bridge_failure_retention` need a built Ashkelon binary; the
optional NumPy, PyTorch and Pillow tests skip when those packages are absent.

## Repository map

| Path | Contents |
|---|---|
| `agentanyl/loop.py` | Ashkelon hook: evaluator call, bounded state, policy, SQLite trace, signal rendering |
| `agentanyl/render.py` | Text/image catalog rendering for signals |
| `agentanyl/activation.py` | Activation backends: Qwen MLX, generic MLX, PyTorch (`TorchActivationBackend`) |
| `agentanyl/setup.py` | Generates an Ashkelon config that runs the hook |
| `experiments/` | Every assay, probe and analysis script; `operant_activation_assay.py` is the final round |
| `research/` | Protocols, results, evidence index, handoff, and raw/summarized data per run |
| `examples/` | Criteria files for the hook |
| `install-ashkelon.sh` | Builds the pinned Ashkelon relay |

## Using the feedback hook

The hook still works as a criteria-driven feedback layer on Ashkelon. The
rest of this README documents it as built.

### Install and connect

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

### What happens on each turn

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

### Configure criteria and interventions

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

### State, reset, inspect, and disable

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

### Open-weight activation backend

Instead of sending a message, an activation backend applies the controller's
state inside a locally hosted model. The final experiments found that this
delivers a real hidden-state change but no incentive (see the table above);
the backends remain for anyone extending that work. The MLX Qwen backend maps controller pain state to an explicit
published residual vector and layer during the next local forward pass. The
backend is model-specific and does not assume that pleasure is the negation of
pain. See [`research/OPEN-ACTIVATION-RESULTS.md`](research/OPEN-ACTIVATION-RESULTS.md)
for the recorded activation-delivery and controller-driven choice experiments.
The reproducible choice artifact is generated with:

```sh
.venv/bin/python -m experiments.open_activation_choice_demo \
  --model /tmp/agentanyl-qwen-4bit \
  --release /tmp/agentanyl-pain-axis \
  --config examples/potato.json \
  --output research/open-activation-demo/qwen7b-choice.json
```

For a playable local browser demo, run one command with the pinned local model
and open the printed URL:

```sh
./run-playground.sh
```

The launcher expects the research environment at `.venv`, the Qwen snapshot at
`/tmp/agentanyl-qwen-4bit`, and the Pain-axis checkout at
`/tmp/agentanyl-pain-axis`. Override those locations without editing code:

```sh
AGENTANYL_MODEL=/path/to/qwen \
AGENTANYL_PAIN_AXIS=/path/to/Pain-axis \
./run-playground.sh
```

The **PAIN** button increments the bounded controller coordinate; **GENERATE**
then applies the corresponding published residual intervention and displays the
answer plus hook telemetry. The **PLEASURE CANDIDATE** button uses a
norm-matched orthogonal control by default. That control is intentionally
labelled unvalidated: no pleasure vector has been established for Qwen. A
calibrated independent vector can be supplied with `--pleasure-vector
/path/to/vector.npy`.

The playground also has a generic MLX mode for portability probes. Supply a
vector calibrated for the target model and its injection layer:

```sh
.venv/bin/python -m experiments.playable_activation_demo \
  --model /path/to/mlx-model \
  --vector /path/to/model-specific-pain.npy \
  --layer 16
```

This mode checks whether the model exposes a compatible `model.layers` hook and
whether the supplied vector has the right hidden width. A successful launch is
only an interface hit; it is not evidence that a vector transfers or has the
same behavioral meaning. Each model still needs its own vector calibration and
behavioral controls.

#### PyTorch backend (Linux, CPU or GPU)

`agentanyl.activation.TorchActivationBackend` is the same hook for Hugging Face
causal LMs. It adds `pain × coefficient_per_level × vector` to one decoder
layer's output at every position, and it records the realized delta and the
projection on the pain direction for each call. As with MLX, pleasure needs
its own vector. The controlled experiments in
[`research/OPERANT-ACTIVATION-RESULTS.md`](research/OPERANT-ACTIVATION-RESULTS.md)
run it on Gemma-2-2B-it with the published Pain-axis vectors; the commands
are under [Reproduce the final experiments](#reproduce-the-final-experiments-linux-cpu-6-hours).

#### What MLX is doing here

MLX is the Apple Silicon tensor and inference runtime. This demo does **not**
fine-tune Qwen and does not change its weights. The backend wraps selected
transformer blocks, runs the ordinary forward pass, and at layer 16 adds the
selected vector to the hidden residual stream before the remaining layers run:

```text
hidden_at_layer_16 = hidden_at_layer_16 + coefficient × intervention_vector
```

`mlx_lm.generate` then samples from the resulting logits. Agentanyl owns the
controller state and chooses the coefficient; MLX performs the tensor
operation; the hook records the realized projection change. This is why the
demo is activation steering rather than a prompt or corrective message.

