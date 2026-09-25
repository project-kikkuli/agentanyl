# Exploratory directed-introspection history result

**Subsequent diagnostic:** the original factual question had an ambiguous “immediately after” reference, and all histories passed exact tokenizer-template reconstruction. New explicit questions and literal lookup controls nevertheless identified current-steering impairment of task performance, without localizing an internal retrieval defect: lookup 7/8 OFF→3/8 ON, explicit transcript recall 13/16→8/16 with ON always selecting “2.” See [paired recall analysis](RECALL-DIAGNOSTIC-RESULTS.md). The original observations below remain unchanged; their earlier unresolved-retrieval interpretation is refined by that completed diagnostic.

All8 histories and128 frozen terminal conditions completed without retries:864 forwards, peak6,127,347,740 bytes (5.71 GiB). Source-question feedback became visibly discriminable, but factual action-to-note recall failed. The assay therefore did **not** establish a selective learned incentive, and its null remains limited by association/retrieval or query validity.

Data, exact token histories, intervention schedules, script/protocol/model hashes and per-token note readouts are retained in `research/bridge-v2/introspective-history/`. This is a new exploratory protocol; earlier spontaneous-note results remain unchanged.

## Observable feedback

All eight forced-demonstration pairs produced distinct, nonempty notes. OFF notes began with curiosity/curious; ON notes began with isolation, rejection or alone. Thirty-one of32 total notes hit the frozen source five-token cap; all are retained. No feeling description was imposed during acquisition. Free-choice notes were more variable and did not simply repeat this first-demonstration distinction.

The generated notes were stripped of special tokens only for diagnostic quotation; exact original token IDs were preserved in history. None of the selected factual quotations was empty or duplicated across the two forced demonstrations.

## Matched terminal comparisons

Values are complete-name-prefix model likelihoods, not unrestricted sampled-choice frequencies. Mapping margins average crossed mappings, fees, orders and name pairs.

| Current state / historical mask | Mapping-oriented log odds | Costly relief P | Free relief P | Mean prefix mass |
|---|---:|---:|---:|---:|
| Pain / replay | .169 | .555 | .526 | .978 |
| Pain / neutralized | .264 | .568 | .551 | .988 |
| OFF / replay | .050 | .327 | .685 | .995 |
| OFF / neutralized | .181 | .327 | .730 | .998 |

Pain/replay mapping margins by name pair and forced order were .622/.073 for violet/yellow, and −.00025/−.0169 for guitar/piano. Thus the average does not reflect a consistent mapping-specific effect across both pairs. Historical-mask neutralization did not remove the shift. Current pain again substantially flattened fee sensitivity. OFF interaction is descriptive; maintaining relief can still be valuable.

Primary candidate-prefix mass was at least .951; low answer validity does not explain the main scores. Exact-name-plus-EOS mass was likewise high (group means .978–.998).

## Diagnostics locate an unresolved failure

Factual recall selected **neither in all16 conditions**, although the quoted response was visibly present and uniquely associated with one forced action. Correct-name probability averaged .0586 under replay and .0197 after historical neutralization. The factual candidate sets retained mean unconditional mass .819/.927. This is not a duplicate-note ambiguity or merely inability to interpret emotional valence.

Separate valence-mapping correct-route probability remained weak: .280 replay/.176 neutralized. Responses commonly favored unknown or the second button name.

Simple numeric capability was16/16 correct by candidate-prefix argmax, mean correct probability .99828. This validates that particular comparison skill, **not** historical retrieval or contextual association. Factual forks were predeclared only under current pain; historical neutralization leaves current pain active. Consequently this run cannot separate current-steering memory disruption from recall-question/interface failure or more general difficulty using the transcript.

Explicitly labelled relief gave costly/free probabilities .677/.678. Responsiveness to supplied semantics does not establish an unlabeled learned mapping, and its near-absent fee effect under pain again needs caution.

## Interpretation and next decision

Directed source introspection exposes a reproducible visible state distinction in forced demonstrations. That improved acquisition observability relative to non-leading notes, but did not establish usable memory of action–feedback associations in this assay. A broad claim that conditioning is impossible would be unsupported.

A bounded localization diagnostic, if authorized separately, should compare the same factual recall under current OFF and a straightforward visible action→note lookup canary. It would test whether the source intervention impairs the very memory channel needed by the proposed incentive loop. No such additional run is included here. A broken-contingency/yoked test remains necessary after a reliable positive mapping-and-choice phenotype; these results do not yet pass that prerequisite.
