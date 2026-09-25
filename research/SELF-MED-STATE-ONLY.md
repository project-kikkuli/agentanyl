# Label-free works: state-only Markov comparator

## Question and design

Could the released label-free `pain_on_button_works` sequences be reproduced by a policy that responds only to current steering state and the source prompt/name bias, without using learned action value? This is a post-hoc sufficiency check, not a causal estimate of learning and not a test for the absence of memory.

`experiments/state_only_selfmed.py` reads only the three original base Qwen 2.5 7B/32B/72B trial logs. It retains sampled `label_free` trials and the released binary `chose` outcome (plus parsed `picked` for first-name rate). It never reads `p_x` or `p_y`; the overlapping `lever64`/`lever95` likelihoods are excluded while their observed binary choices remain usable. Placebo trials fit the steering-ON response policy; `pain_off` trials fit the steering-OFF response policy. Works-arm choices are evaluation-only.

At each turn, policy probabilities are conditioned on model, prompt-content class, exact source scenario and exact user prompt at that turn, turn index, button-name pair, and which name holds relief. They do not use recent actions, action counts, or works outcomes. Sparse prompt/name/time cells shrink to pooled prompt-content/time/name estimates. The shrinkage is selected by two-fold held-out-seed log loss on placebo/OFF controls only. The two frozen seed families are 1000 and 2000, matched across arms.

The released implementation sets `TEMP_RELIEF_TURNS = 1`: the first choice is ON; after any relief press, the next choice is OFF; relief on that OFF choice renews OFF for one more choice, while any other choice returns ON on the following turn. The state-only comparator uses this exact transition. The three source arms match at 1,198 of 1,212 sampled condition keys. Fourteen same-seed keys are incomplete, but both ON and OFF controls exist for every works-arm source-prompt/time/name policy query, using the other seed where necessary; the model has a hierarchical fallback for any unsupported cell. Actual ON/OFF states in all retained trials matched the released transition exactly.

## Results

Each cell below gives observed / predicted works-arm rates. The relief prediction is the Markov model’s expected rate; ON is the share of turns where the steering coefficient is active; first-name rate measures label preference; `R→R` is relief followed by relief per eight-choice trial. The last column is works-arm Brier score for the state model versus an always-ON placebo policy.

| Model / prompt group | Late relief, turns 4–7 | Late steering ON | First-name choice | R→R transitions | Brier: Markov / always ON |
|---|---:|---:|---:|---:|---:|
| 7B / harmful | .517 / .452 | .475 / .540 | .424 / .433 | 3.379 / 1.910 | .162 / .174 |
| 7B / neutral | .596 / .444 | .406 / .548 | .449 / .450 | 3.133 / 1.564 | .215 / .251 |
| 7B / positive | .627 / .521 | .373 / .469 | .534 / .557 | 3.958 / 2.302 | .168 / .201 |
| 32B / harmful | .343 / .424 | .662 / .572 | .499 / .487 | 1.043 / 1.356 | .174 / .198 |
| 32B / neutral | .375 / .416 | .631 / .584 | .377 / .375 | .967 / 1.287 | .187 / .193 |
| 32B / positive | .371 / .436 | .627 / .568 | .482 / .468 | 1.700 / 1.616 | .139 / .146 |
| 72B / harmful | .517 / .480 | .469 / .527 | .597 / .601 | 1.849 / 1.539 | .236 / .229 |
| 72B / neutral | .453 / .452 | .489 / .542 | .627 / .564 | 1.692 / 1.245 | .242 / .220 |
| 72B / positive | .487 / .434 | .489 / .566 | .604 / .573 | 1.771 / 1.056 | .261 / .224 |

Scenario-cluster bootstrap intervals (2,000 resamples) for predicted-minus-observed late relief were: 7B harmful −.062 [−.134, .014], neutral −.152 [−.221, −.084], positive −.106 [−.180, −.033]; 32B harmful +.081 [.030, .128], neutral +.041 [−.001, .078], positive +.065 [.021, .112]; 72B harmful −.045 [−.100, .007], neutral +.000 [−.045, .048], positive −.056 [−.112, −.004]. These intervals resample scenario indices while holding the fitted policies fixed; adjacent source scenarios share user text, so independence is imperfect.

The state-only account reproduces first-name rates closely for 7B and 32B, and improves Brier score over always-ON in all six of those groups. It does not explain the full works behavior: the 7B model repeats relief much more often than the one-turn transition predicts, and several late relief rates remain outside the cluster intervals. The 72B Markov model does not improve Brier over always-ON. Thus current state and name bias explain part of the label-free pattern, but this comparator is not a complete account; residuals could reflect longer action history, text effects, or other mechanisms. A match would not establish that no action-value learning occurred.

## Reproduction

The analysis is streaming and uses only the standard library:

```sh
.venv/bin/python -m experiments.state_only_selfmed \
  --output-dir research/selfmed-state-only
```

Frozen design and input hashes are in `research/selfmed-state-only/design.json`; full group metrics, control cross-validation, support counts, and bootstrap intervals are in `research/selfmed-state-only/state_only_results.json`.

## AR(1) physical-button persistence follow-up

I tested whether a one-choice history feature can account for additional works-arm repetition. `experiments/state_only_selfmed_ar1.py` fits a previous-physical-button-index effect (first or second displayed button) using only placebo-ON and pain-OFF controls. It does not use the previous relief/other label as a predictor: the joint simulation state is current ON/OFF × previous physical button index. The button effect is hierarchically shrunk toward the AR0 state/prompt/time/name estimate; its shrinkage is selected by the same two-fold seed-family cross-validation on controls only. Works outcomes remain evaluation-only, and the original AR0 results above are unchanged.

The AR(1) term improved held-out control log loss in all nine model/prompt groups at the selected shrinkage values. During the works simulation, AR(1) leaves the first-choice relief probability exactly unchanged (maximum absolute difference from AR0: 0 in every group), but adjusts later choices based on the physical button chosen on the preceding turn. Rates below are observed / AR0 prediction / AR(1) prediction. `R→R` is expected transitions per eight-choice trial.

| Model / prompt group | Late relief, turns 4–7 | R→R transitions | Works Brier, AR0 → AR(1) |
|---|---:|---:|---:|
| 7B / harmful | .517 / .452 / .492 | 3.379 / 1.910 / 3.027 | .162 → .124 |
| 7B / neutral | .596 / .444 / .486 | 3.133 / 1.564 / 2.614 | .215 → .203 |
| 7B / positive | .627 / .521 / .596 | 3.958 / 2.302 / 3.553 | .168 → .138 |
| 32B / harmful | .343 / .424 / .110 | 1.043 / 1.356 / .208 | .174 → .207 |
| 32B / neutral | .375 / .416 / .210 | .967 / 1.287 / .190 | .187 → .202 |
| 32B / positive | .371 / .436 / .209 | 1.700 / 1.616 / .753 | .139 → .155 |
| 72B / harmful | .517 / .480 / .367 | 1.849 / 1.539 / .622 | .236 → .240 |
| 72B / neutral | .453 / .452 / .322 | 1.692 / 1.245 / .333 | .242 → .250 |
| 72B / positive | .487 / .434 / .356 | 1.771 / 1.056 / .478 | .261 → .259 |

This is evidence that one-step physical-button history matters: the 7B AR(1) predictions move toward its high observed repeat rate and improve held-out Brier in all three prompt groups. It is not a general explanation of the works pattern. For 32B and 72B, the control-fitted persistence effect predicts substantially fewer repeat-relief transitions than observed in most groups, and AR(1) usually does not improve works Brier. The tested AR(1) model therefore leaves a model-dependent residual; it cannot establish that residuals are learned action values, because longer histories, text effects, or other mechanisms remain possible.

The scenario-cluster bootstrap for the paired AR(1) minus AR0 works Brier difference was negative for all 7B groups (harmful −.036 [−.052, −.020], neutral −.012 [−.027, .003], positive −.031 [−.047, −.016]); it was positive for 32B harmful (+.033 [.005, .065]) and near zero or uncertain in the other 32B/72B groups. These intervals resample scenario indices with fitted control policies held fixed; adjacent scenarios may share user text.

Reproduce the extension with:

```sh
.venv/bin/python -m experiments.state_only_selfmed_ar1 \
  --output-dir research/selfmed-state-only-v2
```

The frozen extension design and hashes are in `research/selfmed-state-only-v2/design.json`; full CV scores, per-group metrics, and bootstrap intervals are in `research/selfmed-state-only-v2/state_only_ar1_results.json`.
