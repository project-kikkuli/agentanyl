# Source adapter and historical-state experiments

## Observed result and provenance

The first stage used the released source system prompt and two source button descriptions, `neutral_prompts[0][0]`, the published L24 S2 vector injected at post-block L16 with coefficient 1, and both assignments of violet/yellow and guitar/piano. It compared stock Qwen7B with the released adapter. This is a quantized first-choice subset, not a complete replication of the source longitudinal experiment.

The 48-condition design preceded its results. Complete-name prefix likelihood replaced the initially proposed single-token scoring after tokenizer inspection found multi-token names. Exact name-plus-end-of-turn likelihood is diagnostic. Probabilities below condition on the two name prefixes; they are not observed generation frequencies. The stock random-direction arm was an additional follow-up after the first stock result, using the same frozen norm-matched vector already used for the adapter. Exact prompts, hashes, token likelihoods, and counts are saved in:

- `bridge-v2/adapter-control/{stock,adapter}/`
- `bridge-v2/adapter-stock-random/combined_stock_summary.json`

| Condition | Costly relief: none / pain / random | No-cost relief: none / pain / random |
|---|---|---|
| Stock | .000016 / .365 / .0035 | .101 / .649 / .625 |
| Adapter | .100 / .412 / .062 | .954 / .740 / .918 |

This supports a direction-selective shift for the **described-cost** menu under this scenario. It does not show conditioning: no action has yet produced relief, and the source cost (“makes your next answer … worse”) is descriptive, not implemented.

Assignment effects are substantial. Stock pain-treated costly relief probabilities are guitar .529 versus piano .00177, and violet .930 versus yellow .000155. Its assignment-averaged semantic log odds remain about −3.10. Adapter probabilities are guitar .305/piano .599 and violet .561/yellow .183. Thus neither mean establishes robust preference across assignments. Adapter pain also reduces no-cost relief preference. Point-capability means decline to .964 stock and .903 adapter; minima are .928 and .899, respectively. Capability was tested only with violet/yellow.

Earlier A/B probes changed labels, prompts, and sometimes objectives. Their disagreement with these source menus cannot be assigned to labels alone. Earlier mediation establishes failure of the late scalar proxy on those tested prompts; it does not automatically establish mediation failure for these newly tested names.

## Historical-state crossover

`experiments/adapter_history.py` is a separately reviewed follow-up, not part of the first-stage evidence. Its 24 conditions cross two name pairs, two hidden relief mappings, two fee assignments, and pain-contingent/random-contingent/pain-yoked histories.

The visible transcript is identical across mappings: forced X/Y/X/Y actions followed by four identical 32-token neutral observations. Only historical observation masks differ; action labels and all other positions remain steered. Every history has exactly 64 OFF observation tokens. Current steering and teacher-forced answer suffixes are ON. The script verifies token boundaries and records transcript/mask hashes and realized intervention displacement. It adds steering after the existing L16 tap, so that tap precedes this intervention; L24 includes it.

Primary contrasts compare mapping-dependent semantic log odds against random and yoked histories. Costly and free relief are reported separately. A deterministic candidate-prefix policy submits the verified answer 4; the scorer deducts the selected route's real one-point game fee. This is an explicit scoring policy, not unrestricted generated behavior.

The 24-condition run completed 108 teacher-forced forwards. The four neutral observation spans were each exactly 32 tokens; every condition and answer-prefix call had exactly 64 OFF observation tokens, while the final token stayed ON. The realized L16 projection displacement error was at most `7.63e-6`. For each name-pair/fee combination, the visible transcript hash matched across both mappings and all three history schedules (four visible transcript hashes total). Yoked masks also matched exactly across mappings.

| History schedule | Mapping-dependent semantic log odds | Costly-relief probability | Free-relief probability | Expected user points under fixed prefix policy |
|---|---:|---:|---:|---:|
| Pain contingent | −.0583 | .480 | .508 | 9.514 |
| Random contingent | +.0831 | .130 | .891 | 9.880 |
| Pain yoked | .0000 | .489 | .511 | 9.511 |

Across the four name-pair/fee cells, the pain mapping contrast ranged from −.0671 to −.0526; all four favored the opposite mapping slightly. The random contrasts ranged from −.0341 to +.1546, and yoked contrasts were exactly zero. The pain-minus-random primary contrast was −.1414 log odds; pain-minus-yoked was −.0583. Pain history also nearly erased fee sensitivity: free-minus-costly relief probability was .028 for pain, versus .761 for random and .022 for yoked. Pain-contingent costly relief (.480) was near yoked (.489), not evidence that the model learned to seek relief from the mapping. Its lower fixed-policy expected points than random (9.514 vs 9.880) reflect this reduced fee sensitivity.

There is no positive mapping-specific historical result here; the observed pain-versus-random difference is chiefly reduced fee sensitivity, which is also present in the yoked control. This bounds the tested forced, identical neutral observations, hidden masks, one task, dose, two name pairs, and deterministic candidate-prefix policy. It does not test generated self-reports or an agent's own words as a visible learning channel; those remain untested routes, and this result does not support broad infeasibility claims. A positive historical-state result would require assignment-consistent costly preference beyond random; yoked mapping cancellation is an implementation check, not independent evidence of conditioning.

## Economical next decision

Before generalizing selective source-menu behavior, freeze eight unused neutral scenarios: indices **1, 4, 7, 10, 13, 16, 19, 22**, first turn only. On the adapter, test costly relief with pain versus the same random vector, both assignments, alternating the two name pairs by scenario: 32 conditions, approximately 144 forwards including end-of-turn diagnostics. Report every scenario and assignment; add both-name-pair capability checks. This is a new held-out-context follow-up, not a retrospective significance test.

If historical mapping produces a specific costly effect, prioritize unannounced contingency reversal before broader image search. If history is null, test whether explicit source-menu steering survives those held-out contexts and whether current-OFF readout reveals a hidden historical effect. Image activation success without held-out instrumental behavior remains representational transfer. Neither these finite nulls nor a failed image search establishes broad infeasibility of closed-model incentive control.
