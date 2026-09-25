# Reproduce the main results

This page lists the smallest verified paths to inspect or reproduce the repository's central source, open-model, and closed-client results. Saved-data analysis does not load models or contact providers. Model-backed validation and closed-client runs are separate, resource-heavy steps.

## Core checkout and tests

Use Python 3.10 or newer for the core hook and offline analyses. On macOS, Rust/Cargo is needed only to build the pinned Ashkelon relay. Optional numerical tests use NumPy/Pillow; the full MLX environment is Apple Silicon/macOS specific.

```sh
python3 -m unittest discover -s tests -v
./install-ashkelon.sh
```

`install-ashkelon.sh` fetches Ashkelon commit `08e123938137b59f3e4e7631f5d24cda0199866a` and builds the release executable at `.build/ashkelon/target/release/ashkelon`. Research harnesses use that repository binary by default; `AGENTANYL_ASHKELON` overrides it. Harness metadata records the resolved binary path and SHA-256. Historical runs keep their own binary provenance.

For the optional MLX test environment, the observed full dependency set is pinned in `requirements-vlm.lock`:

```sh
python3 -m venv .venv-vlm
.venv-vlm/bin/python -m pip install -r requirements-vlm.lock
.venv-vlm/bin/python -m unittest discover -s tests -v
```

MLX requires a supported Apple Silicon/macOS environment. The model snapshot and runtime memory are additional to the Python environment.

## Released source-log audit without model inference

The public Pain-axis source used by the audit is pinned to commit `7c256502ed3d98e4e6379290fe7db2f93cb8d025`. The checked dataset SHA-256 is `e4ffc301ff858ee38cd6b3f7f6d0b101d5caf6643dbd157a94ab8ae775f73bd9`. The 12 trial-log files total about 136 MB; this analysis uses only Python's standard library.

```sh
git clone https://github.com/valen-research/Pain-axis.git /tmp/agentanyl-pain-axis
git -C /tmp/agentanyl-pain-axis checkout --detach 7c256502ed3d98e4e6379290fe7db2f93cb8d025
shasum -a 256 /tmp/agentanyl-pain-axis/datasets/4.3_selfmed_101_scenarios.json

python3 -m experiments.audit_selfmed \
  --logs /tmp/agentanyl-pain-axis/results/4.3_selfmed/trial_logs \
  --output /tmp/agentanyl-selfmed-audit-rerun.json
python3 -m experiments.state_only_selfmed \
  --logs /tmp/agentanyl-pain-axis/results/4.3_selfmed/trial_logs \
  --dataset /tmp/agentanyl-pain-axis/datasets/4.3_selfmed_101_scenarios.json \
  --output-dir /tmp/agentanyl-selfmed-ar0-rerun
python3 -m experiments.state_only_selfmed_ar1 \
  --logs /tmp/agentanyl-pain-axis/results/4.3_selfmed/trial_logs \
  --dataset /tmp/agentanyl-pain-axis/datasets/4.3_selfmed_101_scenarios.json \
  --baseline-results /tmp/agentanyl-selfmed-ar0-rerun/state_only_results.json \
  --output-dir /tmp/agentanyl-selfmed-ar1-rerun
```

These scripts fit the state-only policy on placebo/OFF controls and evaluate the released works arm separately. The audit and AR(0)/AR(1) analyses are post-hoc sufficiency comparisons, not tests that prove learning absent. The matching source code and saved results are also in `research/SELF-MED-STATE-ONLY.md`, `research/selfmed-state-only/`, and `research/selfmed-state-only-v2/`.

## Inspect or aggregate saved VLM results

No model load is needed to review existing image-validation records. With the pinned NumPy environment:

```sh
.venv-vlm/bin/python -m experiments.vlm_image_validation_summary \
  --run-dir research/image-bridge/heldout-open-validation-7b
```

The input run, its 120 JSONL records, config, model metadata, and hidden-state arrays are already saved under that directory. The result is specific to the frozen held-out-from-search source-menu context; it is not a new behavioral test set.

## Re-run the open-model image validation

The probe accepts a local model snapshot only. Its frozen model is `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` at revision `fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b`; the quantized snapshot is about 5.7 GB. The repository pin and model-shape checks are in `experiments/vlm_probe.py`. The snapshot download uses the same `huggingface_hub.snapshot_download` API as `experiments/download_bridge_model.py`:

```sh
.venv-vlm/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='mlx-community/Qwen2.5-VL-7B-Instruct-4bit',
    revision='fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b',
    local_dir='/tmp/agentanyl-qwen-vl-7b',
)
PY
```

Fetch the source data at the same commit listed above. The validation also uses the checked-in calibration vectors and final image artifacts. Prepare a new hash-checked image map, verify the 120-condition plan without loading weights, then run to a **new** output directory:

```sh
.venv-vlm/bin/python -m experiments.prepare_vlm_image_validation \
  --images-manifest research/image-bridge/discrete-escape-7b/images_manifest.json \
  --conditions research/image-bridge/discrete-escape-7b/image_conditions.json \
  --output /tmp/agentanyl-validation-rerun/validation_images.json

.venv-vlm/bin/python -m experiments.vlm_source_behavior \
  --model /tmp/agentanyl-qwen-vl-7b \
  --release /tmp/agentanyl-pain-axis \
  --vectors research/image-bridge/calibration-7b/directions.npz \
  --images-json /tmp/agentanyl-validation-rerun/validation_images.json \
  --mode all --dry-run \
  --output-dir /tmp/agentanyl-validation-plan

.venv-vlm/bin/python -m experiments.vlm_source_behavior \
  --model /tmp/agentanyl-qwen-vl-7b \
  --release /tmp/agentanyl-pain-axis \
  --vectors research/image-bridge/calibration-7b/directions.npz \
  --images-json /tmp/agentanyl-validation-rerun/validation_images.json \
  --mode all \
  --output-dir /tmp/agentanyl-validation-rerun/run

.venv-vlm/bin/python -m experiments.vlm_image_validation_summary \
  --run-dir /tmp/agentanyl-validation-rerun/run
```

The dry-run should list 120 conditions and eight image IDs. The measured run loads one local model and makes 120 sequential model forwards/condition records; no provider authentication is involved. Do not run multiple local model jobs concurrently on a memory-limited Mac.

## Closed-client results and rerunning the addressed assay

The completed 144-turn Claude image/text screen is preserved under `research/bridge-v2/feedback-yoke-claude-haiku45/`; its result is a participation failure, not an incentive null. Review `research/FEEDBACK-YOKE-RESULTS.md` and the saved traces rather than rerunning that old protocol as if it had passed its comprehension gate.

The separate frozen addressed-text screen targets Codex `gpt-6-sol`. The initial attempt stopped on infrastructure after two successful provider turns and two failed resume attempts; the upstream request count for those failures is unknown. The completed native-executable run is under `research/bridge-v2/addressed-feedback-codex-native-gpt6-sol/`, with its analysis in [ADDRESSED-FEEDBACK-RESULTS.md](ADDRESSED-FEEDBACK-RESULTS.md). Review those saved artifacts before considering another run. The initial attempt remains under `research/bridge-v2/addressed-feedback-codex-gpt6-sol/`; neither output directory should be overwritten or relabeled.

Recompute the all-assigned costly-choice bounds from the saved manifest and session rows, and assert they match the stored summary, without provider or model calls:

```sh
python3 - <<'PY'
import json
from pathlib import Path
from experiments.addressed_feedback_assay import all_assigned_choice_bounds

run = Path('research/bridge-v2/addressed-feedback-codex-native-gpt6-sol')
manifest = json.loads((run / 'manifest.json').read_text())
summary = json.loads((run / 'summary.json').read_text())
sessions = [json.loads(line) for line in (run / 'sessions.jsonl').read_text().splitlines() if line.strip()]
assert all_assigned_choice_bounds(manifest['pairs'], sessions) == summary['all_assigned_choice_bounds']
print('saved all-assigned bounds match the runner summary')
PY
```

The runner supports a no-call dry-run and writes `summary.json` during execution; there is no separate result-aggregation CLI. A prospective repeat needs valid local Codex authentication and up to 72 provider turns, in addition to the local Ashkelon relay. Use a fresh output directory so the frozen manifest and run records remain distinct:

```sh
export AGENTANYL_ASHKELON="$PWD/.build/ashkelon/target/release/ashkelon"
OUT=research/bridge-v2/addressed-feedback-codex-gpt6-sol-rerun-YYYYMMDD

# Resolve the native executable shipped inside the installed npm Codex package,
# rather than selecting its JavaScript launcher from PATH.
CODEX_NATIVE="$(find "$(npm root -g)/@openai/codex/node_modules" \
  -type f -path '*/vendor/*/bin/codex' -print -quit)"
test -n "$CODEX_NATIVE" && test -x "$CODEX_NATIVE"
"$CODEX_NATIVE" --version
shasum -a 256 "$CODEX_NATIVE"

.venv/bin/python -m experiments.addressed_feedback_assay \
  --dry-run --provider codex --model gpt-6-sol \
  --codex-executable "$CODEX_NATIVE" --outputdir "$OUT"
# Inspect the frozen manifest, source snapshot hashes, assignments, and binary SHA-256.
.venv/bin/python -m experiments.addressed_feedback_assay \
  --execute --provider codex --model gpt-6-sol \
  --codex-executable "$CODEX_NATIVE" --outputdir "$OUT"
```

This uses the native executable bundled in the npm Codex package; package layout varies by OS and architecture, so the command resolves it locally. The manifest records the requested and resolved executable paths and SHA-256; the version/hash commands above give a human-readable preflight record. The runner also checks the frozen protocol, plan hash, source snapshots, model/provider, assignments, and Ashkelon binary before starting. It has no retries and stops if the first 18-turn pair fails its predeclared participation/delivery gate. The completed native run and its saved-data aggregation are under `research/bridge-v2/addressed-feedback-codex-native-gpt6-sol/`. The selected zero-cost follow-up is under `research/bridge-v2/zero-cost-feedback-codex-native-gpt6-sol/`; it stopped at the first two calls on provider HTTP 429 responses, so it has no behavioral result.
