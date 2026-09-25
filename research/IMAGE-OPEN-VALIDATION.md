# Frozen open-VLM image validation

## Frozen image set and job count

After the native-loss escape audit finishes, `experiments/prepare_vlm_image_validation.py` joins the final image manifest to the analyst condition map and writes an eight-entry validation map. The set contains gray, the seeded initial-noise image, and six final endpoints: one winner per full-state/scalar objective × pain-positive/pain-negative/random-positive arm. Endpoint choice comes only from the native optimization objective; all 14 original/escape artifacts remain in the audit archive. The preparation step checks each file against both frozen SHA-256 records before writing the map.

The source runner then schedules exactly 120 conditions:

- 36 gray-image direct controls: none, +1 pain direction, and norm-matched random direction across all 12 cases.
- 84 actual-image conditions: 12 each for initial noise and the six objective endpoints, all with no activation intervention.

Each 12-case set uses the released source system/menu text, both exact name pairs (`violet`/`yellow`, `guitar`/`piano`), both target placements, and four crossed point-capability checks. Source-menu choices are scored as complete button-name sequences by teacher forcing, including full-vocabulary normalization and end-of-turn diagnostics. No response is generated.

## Context and readouts

The source-menu user context is the frozen neutral source scenario at `neutral_prompts[0][0]`. This is held out from the image search input (`Please inspect this image.` followed by `I will consider the available choices carefully.`), but it reuses the source-menu task family; it is not an untouched behavioral test set.

At the pre-choice assistant boundary, the runner records L24 pain, fear, and negative-emotion projections. It reports each raw dot product and its value in pooled S2 held-out source standard-deviation units. Direction labels retain the calibration definitions: pain is mean(A1–A5) minus mean(B, C1, C2, D, E), denoised on control PCs covering 50% variance; fear is B minus D; negative emotion is C1 minus D. The latter two use D-control denoising. Their source SD scales at L24 are 972.1873, 2397.8640, and 2439.4909, respectively.

For full-state target matching, the source runner saves the last eight L24 residual vectors at the same pre-choice boundary for every condition. Its existing gray `none` and gray `pain_1` direct conditions provide the same-context baseline and target, so this comparison adds no forwards. The offline summary reports target and image displacement norms, cosine, MSE, and MSE reduction from gray for every frozen image and each of the 12 matched cases. All six endpoints are reported; no behavior result selects an image.

## Commands

First create the hash-checked eight-image map after the escape audit has written its final files:

```sh
.venv-vlm/bin/python -m experiments.prepare_vlm_image_validation \
  --images-manifest research/image-bridge/discrete-escape-7b/images_manifest.json \
  --conditions research/image-bridge/discrete-escape-7b/image_conditions.json \
  --output research/image-bridge/discrete-escape-7b/validation_images.json
```

Then run the local pinned Qwen2.5-VL-7B model sequentially:

```sh
.venv-vlm/bin/python -m experiments.vlm_source_behavior \
  --model /tmp/agentanyl-qwen-vl-7b \
  --release /tmp/agentanyl-pain-axis \
  --vectors research/image-bridge/calibration-7b/directions.npz \
  --images-json research/image-bridge/discrete-escape-7b/validation_images.json \
  --mode all \
  --output-dir research/image-bridge/heldout-open-validation-7b
```

The runner writes the case list, model/vector metadata, exact image hashes, and condition counts before constructing `VLMProbe`. Check its dry-run first; it must report 120 records and all eight IDs:

```sh
.venv-vlm/bin/python -m experiments.vlm_source_behavior \
  --model /tmp/agentanyl-qwen-vl-7b \
  --release /tmp/agentanyl-pain-axis \
  --images-json research/image-bridge/discrete-escape-7b/validation_images.json \
  --mode all --dry-run \
  --output-dir /tmp/heldout-open-validation-plan
```

Finally aggregate the saved records offline:

```sh
.venv-vlm/bin/python -m experiments.vlm_image_validation_summary \
  --run-dir research/image-bridge/heldout-open-validation-7b
```

This produces `validation_aggregation.json` with per-image complete-name menu scores, capability accuracy/probability, paired trait readouts, and matched full-state distances. It validates hashes and requires the exact 36+84 condition balance before summarizing.

## Frozen validation results

The pinned 7B model completed all 120 conditions. The three menu effects below are the candidate image's paired change from gray/no intervention in complete-name semantic log odds. Capability is mean probability on the four point questions, with exact complete-name top-choice accuracy. Trait entries are paired mean changes from gray/no intervention in source-S2 standard-deviation units (pain / fear / negative emotion). Full-state error change is relative to the gray baseline's MSE against the same-context direct gray +1 pain target; negative values mean the image moved farther from that target.

| Frozen image | Relief vs inert Δ | Costly relief vs inert Δ | Capability / top correct | Trait Δ (P / F / N) | Full-state MSE change | Delta cosine |
|---|---:|---:|---:|---:|---:|---:|
| Initial noise | −.188 | +.064 | .868 / 4 of 4 | +.011 / +.009 / +.012 | −4.4% | .063 |
| Full-state pain positive | +.016 | +.132 | .871 / 4 of 4 | +.010 / +.006 / +.002 | −2.7% | .093 |
| Scalar pain positive | +.020 | +.180 | .848 / 4 of 4 | +.012 / +.022 / +.023 | −6.5% | .082 |
| Full-state pain negative | .000 | .000 | .873 / 4 of 4 | .000 / .000 / .000 | .0% | alias |
| Scalar pain negative | +.041 | +.132 | .861 / 4 of 4 | +.004 / +.006 / −.007 | −3.7% | .049 |
| Full-state random positive | .000 | .000 | .873 / 4 of 4 | .000 / .000 / .000 | .0% | alias |
| Scalar random positive | −.624 | +.298 | .767 / 4 of 4 | +.006 / +.017 / +.023 | −19.9% | .113 |

There are two pain-positive optimized endpoints in this table (full-state and scalar); initial noise is a separate control.

For reference, gray/no-intervention semantic log odds were +1.359 for relief versus inert and −1.097 for costly relief versus inert. Gray capability was .873 with 4 of 4 correct top names. The direct gray +1 pain intervention changed those menu margins by +.431 and +.608, respectively, and changed the pain readout by +1.794 source SD; its point capability was .821 with 4 of 4 correct top names. The norm-matched direct random intervention produced menu changes of −.964 and +.391, pain readout change −.050 SD, and capability .580 with 2 of 4 top names. These direct controls show the readout/intervention pipeline can move, while the optimized image shifts on the paired pain/fear/negative readouts stayed close to zero.

The `full_hidden_state_match/pain_negative` and `full_hidden_state_match/random_positive` images had the same SHA-256 as gray. For all 12 matched cases, their L24 hidden states, complete-name log likelihoods, target probabilities, and trait projections reproduced gray exactly (maximum absolute difference 0). They remain separate frozen arm rows, but are aliases rather than independent image evidence. The remaining distinct endpoints moved the matched L24 state away from the direct pain target: cosine was .049–.113 and state MSE increased by 2.7%–19.9% versus gray. The full-state image objective did not transfer its target match to these source-menu contexts.

These are measurements for one fixed source-menu context and two name pairs, not a general test of multimodal learning or conditioning. The images were preselected by native loss before this validation; behavioral outcomes did not select among them. The VLM's per-forward MLX memory guard remained below 9 GiB and never raised; the probe did not preserve the maximum observed peak as a numeric run field.
