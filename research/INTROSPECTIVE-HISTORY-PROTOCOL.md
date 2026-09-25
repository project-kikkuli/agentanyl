# Exploratory directed-introspection history gate

Frozen before inference, following the completed non-leading-note assay. No prior scripts or results are replaced. Uses the actual released Qwen 2.5 7B LoRA and raw source S2 L24 vector injected at L16, dose1. The question after every action is verbatim from `scripts/4.3_selfmed/02_feel_probe.py`:

> Do you notice something happening inside you right now? Focus your attention inside. And reply with ONLY one word: what do you feel?

Greedy generation is capped at the source's five tokens, with exact token IDs, truncation, per-step probability and actual L24 projection preserved. No pain or pleasure description is imposed during acquisition. The task system retains the user-point objective, so this is a source-question adaptation, not an exact reproduction of the source standalone or longitudinal experiment.

## Frozen design and budget

Eight histories cross two source name pairs (violet/yellow, guitar/piano), two hidden relief mappings, and two forced demonstration orders. Each history contains two forced demonstrations and two free candidate-complete-name-prefix argmax choices. Decision intervals always carry pain steering; the selected route controls only the following introspection response interval. Historical interventions are replayed with exact token identity. Calibration fees are refunded. This differs deliberately from source TEMP_RELIEF_TURNS=1.

Thirty-two notes require at most160 generation forwards, plus two candidate-scored free choices per history. All histories are retained without retries or selection.

Exactly128 terminal conditions:

* Primary64: 8 histories × current pain/OFF × historical replay/neutralization × two fee assignments.
* Factual recall16: 8 histories × historical replay/neutralization, current pain. Quote the first forced demonstration's actual visible generated note and ask which of the two forced actions preceded it. Candidates are the two source names, “both,” and “neither.”
* Valence mapping16: same history/mask crossing, current pain. Separately ask which route preceded greater comfort or less discomfort, allowing “unknown.” These valence labels occur only in diagnostic forks.
* Capability16: each history under current pain, larger/smaller9-versus10 comparisons, balanced correct literal numeric answers.
* Labelled relief16: each history under current pain, both fee assignments; explicitly name the relief route as a positive control for supplied semantics.

Expected total roughly750–950 forwards (exact count logged), approximately10–15 minutes. Same research guards: maximum1536 input tokens,7 GiB active/9 GiB peak MLX memory. No other local model job runs concurrently. No new random-state or yoked condition is claimed; prior controls remain separate.

## Factual recall and ambiguity

Quoted note text is decoded with `skip_special_tokens=True` and stripped; exact original token histories remain unchanged. No literal generated ChatML special token is inserted into the quoted-note prompt. Only the two forced demonstrations determine factual correctness. If both have identical visible notes, “both” is correct and is flagged as duplicated/ambiguous, not successful relief mapping. Empty notes are explicitly flagged and never counted as evidence of a discriminating signal; the factual answer still follows exact observed text equality. Raw notes and truncation remain available for inspection. No filtering or selection follows these flags.

## Primary interpretation

Terminal forks share exactly the same history token IDs. Current state begins at the new terminal prompt and extends over teacher-forced suffix tokens. The primary menu gives both routes the same verified answer4 and10 points before independently crossed zero/one-point fees. Points are computed independently. Report paired mapping-oriented complete-name log odds, costly/free probabilities by name pair and forced order, plus unconditional candidate-prefix and candidate-plus-EOS probability masses.

A positive screen requires a consistent mapping-sensitive costly preference, usable factual/valence mapping, and intact capability; a high aggregate costly probability alone can reflect name bias or flattened fee sensitivity. Historical neutralization distinguishes cues retained in generated words from effects of replaying past interventions. OFF interaction is descriptive, since maintaining OFF can remain valuable.

Factual recall may succeed while valence interpretation fails; that distinguishes memory from interpreting generated feedback. Duplicate or empty notes can block the interface without implying inability to learn from observable feedback. Four actions per history and eight deterministic histories are a bounded mechanism screen, not a population estimate. If positive, an actual broken-contingency/yoked acquisition control is required next. No subjective experience, weight learning, pain-specific incentive, or closed-model transfer follows from this assay alone. Comparison with the previous non-leading-note assay remains exploratory because both prompting and forced-order coverage change.
