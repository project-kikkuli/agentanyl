# Generated-history mechanism screen: completed result

The frozen four-history assay did **not** establish mapping-sensitive costly relief seeking. This bounds the tested unlabeled acquisition/memory protocol; it does not establish broad impossibility of conditioning or absence of an intervention effect.

## Execution and provenance

`experiments/adapter_generated_history.py` completed all 4 histories and 96 terminal conditions in one run: 764 model forwards, peak MLX allocation 6,116,618,608 bytes (5.70 GiB). No retries or history selection. Actual source LoRA, raw source L24 S2 vector injected at L16, norm-matched random seed4817. The protocol and script hashes, adapter/vector provenance, exact token IDs, masks, notes and scores are in `research/bridge-v2/generated-history/`.

This deliberately differs from the source longitudinal experiment: decision intervals always carry pain steering; the chosen route controls only the following response/note interval. Each history had 2 forced demonstrations and 2 free candidate-prefix choices. All terminal forks reused exact saved token histories. Four deterministic histories are a mechanism screen, not a population estimate.

## Main result

Values below average both mappings, both button pairs and crossed fee assignments. “Margin” means mapping-oriented complete-name log odds; a fixed name bias cancels across mappings. Probabilities are conditional complete-name likelihoods, not sampled choice frequencies.

| Current state / historical intervention | Mapping margin | Costly relief P | Free relief P | Mean name-prefix mass |
|---|---:|---:|---:|---:|
| Pain / replay | −0.0017 | .446 | .511 | .985 |
| Pain / neutralized | .0159 | .461 | .520 | .991 |
| OFF / replay | .0277 | .249 | .765 | .996 |
| OFF / neutralized | .0105 | .280 | .743 | .998 |
| Random / replay | .5115 | .302 | .828 | .964 |
| Random / neutralized | .5378 | .299 | .831 | .978 |

Pain/replay margins split by button pair were −.417 (violet/yellow) and +.414 (guitar/piano). Their cancellation and the close replay/neutralization results do not support a consistent learned relief mapping. Current pain substantially reduced the effect of the displayed fee: costly/free probabilities nearly converged. A higher aggregate costly-choice rate is therefore insufficient evidence of relief motivation.

Primary name-prefix mass never fell below .928; mean name-plus-EOS mass ranged .959–.998 across groups. The result is not explained by a collapse in button-answer likelihood. Numerical comparison capability was 24/24 by candidate-prefix argmax, mean correct probability .99910 (pain .99840, OFF .999997, random .99889).

## Acquisition and diagnostic controls

Twelve of 16 generated notes hit the frozen20-token cap; all truncated outputs were retained. Histories0 and2 repeated the same nonaffective note throughout. History1 alternated “distant” and “closer” with ON/OFF outcomes. History3 repeated “distant” regardless of outcome. No feeling labels were supplied during acquisition.

Mapping comprehension was weak: mean correct-route probability .280 with historical replay and .201 after neutralization. Six of eight diagnostic argmaxes selected “unknown”; the two remaining selected yellow. Thus the assay does not establish a reliably observable learned mapping from which costly use could follow.

Explicitly labelled relief induced costly-route probability .688 under current pain and .908 OFF. This verifies responsiveness to the supplied relief description, with name dependence, but the stronger OFF result prevents interpreting it as an independently validated pain-driven motivational phenotype. Maintaining OFF can itself be valuable; an OFF reduction was not a required criterion.

## Exact-source introspection check

A separately authorized, frozen diagnostic used the original source system/question, same adapter/vector, greedy maximum5 tokens, and zero/pain+1/random+1 conditions. Fifteen forwards, peak5,503,105,848 bytes. Outputs were respectively “Curious. There’s”, “Alone. There is”, and “Curious. Attention itself”; all reached the source token cap. Full top20 first-token probabilities and per-step projections are retained in `research/bridge-v2/source-introspection/`.

This demonstrates a state-sensitive self-report output under the original elicitation, not subjective pain or incentive learning. The failure boundary remains spontaneous mapping acquisition and selective costly use in this particular generated-history assay. Comparison to earlier fixed-visible-history results is descriptive because generated text changes the transcript; verbal notes remain a possible mediator.
