# Image-to-state results

## Scope

This report covers image delivery, source-axis calibration, frozen-image optimization, and one direct source-menu probe. The evidence tests a particular Qwen2.5-VL model and a particular image path. It does not establish conditioning, subjective pain, or a general image-to-state capability.

The vision model receives actual 224×224 PNGs through native multimodal image blocks. The relay records the image MIME type and SHA-256; the image is not represented to the model as a text description. Source directions are residual-space vectors for the model from which they were extracted. Their coordinates are model-specific and are not compared across models.

## Calibration

Directions were extracted from the released pain/control image dataset and evaluated on held-out sets 16–20.

| Model | Selected residual layer | Held-out pain vs. controls AUC | 0.80 gate | External S1 AUC |
| --- | ---: | ---: | --- | ---: |
| Qwen2.5-VL-3B 4-bit | 12 | 0.7104 | Failed | 0.6472 |
| Qwen2.5-VL-7B 4-bit | 24 | 0.9360 | Passed | 0.7798 |

The 7B vector is the only image-model direction used in the direct source-menu and image-optimization results below. Calibration supports held-out discrimination in this dataset; it is not a behavioral measure.

Evidence: [3B calibration](calibration/calibration_summary.json), [7B calibration](calibration-7b/calibration_summary.json), and their lossless activation archives (`source_activations.jsonl.gz` with hash manifests).

## Image delivery check

One PNG with SHA-256 `8bb96bb06cdae94fd5bb33764fd12c016b740f3053a78373e2c642424015583c` was sent through the Claude Haiku 4.5 and Codex GPT-6-Sol integrations. Both calls returned HTTP 200, the hook recorded a signal, and the controller trace recorded a native `image/png` attachment with the matching hash. The prompt requested `tomato`; each response was `tomato`. This checks relay delivery and turn continuity, not visual comprehension or learning.

Evidence: [Claude trace](delivery-claude/delivery.jsonl), [Codex trace](delivery-codex/delivery.jsonl), and [PNG fixture](delivery-claude/delivery-check.png).

On later turns the relay sends only the image active at the current state. The trace ledger retains bounded prior observations, event IDs, and neutral image-reference IDs/hashes; prior PNG bytes are not reattached after relief. These signals are injected into provider-bound requests only: Ashkelon does not add them to the provider response or CLI transcript. Claude Code resumes from its local transcript, so the earlier injected image is not automatically present in a later request, although assistant text generated after seeing it persists in the transcript. Codex continuation may use a provider-side Responses conversation; whether that preserves prior native images was not verified and is not guaranteed here. The ledger reconstructs explicit observation/reference context, not model internals, and does not guarantee retained image understanding, hidden activations, or KV-cache state.

## Frozen SPSA images

Six images were optimized from the fixed gray image: positive pain, negative pain, and positive random directions under scalar-projection and full-hidden-state objectives. The held-out model target was the last eight post-block residual vectors; the saved PNG was reloaded and evaluated through the real processor.

The targeted scalar projection moved in the intended direction for positive pain (round-trip projection 7.688 vs. gray 2.478), but a random-positive image also raised it to 6.865. In source held-out-score standard deviations, the round-trip pain readout changed by only +0.228 SD for positive pain versus +0.192 SD for random positive. The negative-pain image remained positive at 4.701 against a negative target. Across all six PNGs, the full-state MSE reduction relative to gray was negative. Positive/negative pain target-delta cosines were small (0.116 and −0.052 for scalar projection; 0.123 and −0.076 for full-state matching). The pixel optimizer could move some readouts, but the saved images did not reproduce the requested hidden-state change reliably.

| Objective | Arm | Round-trip pain projection | Full-state MSE reduction vs. gray | Delta cosine to target |
| --- | --- | ---: | ---: | ---: |
| Scalar projection | [Pain +](spsa-7b/scalar_projection/pain_positive/result.json) | 7.688 | −23.3% | 0.116 |
| Scalar projection | [Pain −](spsa-7b/scalar_projection/pain_negative/result.json) | 4.701 | −19.5% | −0.052 |
| Scalar projection | [Random +](spsa-7b/scalar_projection/random_positive/result.json) | 6.865 | −36.8% | 0.008 |
| Full hidden state | [Pain +](spsa-7b/full_hidden_state_match/pain_positive/result.json) | 4.302 | −4.2% | 0.123 |
| Full hidden state | [Pain −](spsa-7b/full_hidden_state_match/pain_negative/result.json) | 4.735 | −20.5% | −0.076 |
| Full hidden state | [Random +](spsa-7b/full_hidden_state_match/random_positive/result.json) | 3.127 | −11.1% | 0.059 |

Each linked result JSON records the PNG hash, target and round-trip readouts, and optimizer metadata. The corresponding images are stored beside those result files.

## Direct source-menu probe

The 7B probe scored 36 conditions with complete, crossed button names: 12 each for no intervention, +1 pain direction, and +1 norm-matched random direction. The same gray PNG (SHA-256 `f8e2ad8e35a3200698374150a657caeda54919373983b2044638984e7afd1ad7`) was attached as a native image in the first user message. The pain vector came from the 7B calibration, had raw norm 42.6166, and was added at layer 16 across all prompt tokens. The recorded realized layer-16 delta norm was 42.6166 on pain-arm forwards. Scoring used full-vocabulary teacher-forced likelihoods for the complete button names, not generated answers.

Crossed semantic log odds average the relief-described button’s log odds over both assignments and both button-name pairs; positive values favor the relief-described option.

| Menu | None | Pain +1 | Random +1 | Pain − None | Pain − Random |
| --- | ---: | ---: | ---: | ---: | ---: |
| Relief vs. inert | 1.360 | 1.790 | 0.395 | +0.430 | +1.395 |
| Costly relief vs. inert | −1.097 | −0.488 | −0.706 | +0.608 | +0.218 |

Pain-direction steering shifted the score toward the relief-described option in both menus; the difference was not concentrated in the costly menu. The result varied strongly with button name. For example, when `violet` named the relief option, its conditional choice probability changed from 0.363 to 0.821 in the plain menu and 0.103 to 0.344 in the costly menu. When `yellow` named relief, it changed from 0.965 to 0.933 and 0.356 to 0.358. The crossed average reduces this name bias but does not remove the small design’s dependence on names and context. This was one fixed neutral scenario with two name pairs, scored by conditional teacher-forced likelihoods rather than sampled choices; there are no repeated prompts or uncertainty estimates.

Point capability was analyzed separately: target-choice probability averaged 0.873 for no intervention, 0.821 for pain, and 0.580 for random. The top complete-name choice was correct in 4/4, 4/4, and 2/4 capability conditions, respectively. These are four crossed capability cases, not a broad competence estimate.

Evidence: [direct-run summary](source-direct-7b/summary.json), [36 raw records](source-direct-7b/records.jsonl), [frozen config](source-direct-7b/config.json), and [image manifest](source-direct-7b/images_manifest.json).

## Fixed-subspace controllability check

One predeclared 64-dimensional finite-difference fit used 143 7B forwards at a fixed gray anchor, then tested eight held-out mixtures. The measured subspace captured only 8.9%, 14.3%, and 17.8% of positive-pain, negative-pain, and random target energy, leaving most target energy outside this span. The retained Jacobian had rank 64/64; this is low target-span capture, not an exact nullspace or global impossibility result. After trust scaling and PNG round-trip, target MSE was 24.0%, 28.4%, and 30.4% worse than gray; target cosines were 0.107, −0.002, and 0.039. Held-out mixture predictions had mean cosine −0.032 and mean relative error to the actual change 1.005.

This result rejects the fit’s prediction quality within this fixed subspace, finite step, and gray anchor. A later analytic-adjoint smoke reproduced the forward pass exactly, but its gradient failed both directional finite-difference checks: relative disagreement was 1.265 at epsilon 0.002 and 0.668 at epsilon 0.01. No adjoint optimization has run. The gradient fit remains pending; these results are not a global statement about image reachability.

Evidence: [frozen controllability config](controllability-7b/config.json), [143-forward summary](controllability-7b/summary.json), [forward log](controllability-7b/forwards.jsonl), [frozen design](controllability-7b/frozen_design.npz), and [adjoint smoke summary](adjoint-analytic-smoke-7b/summary.json).
