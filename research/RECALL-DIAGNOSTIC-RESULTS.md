# Recall diagnostic: current steering impairs task performance beyond numeric comparisons

**The +1 source intervention impaired literal-lookup performance and biased transcript answers despite preserved simple numerical comparisons.** This is a concrete confound for interpreting increased costly choices or flattened fee sensitivity as motivation. The result applies to these prompts, this adapter/model and this dose; it is not a universal claim about pain-axis interventions.

The frozen64-condition diagnostic completed336 forwards, peak6,108,955,420 bytes (5.69 GiB), no retries. All records and the concrete pre-inference plan are in `research/bridge-v2/recall-diagnostic/`. Eight fixed histories are a small deterministic mechanism screen, not independent population samples or a statistical prevalence estimate.

## Serialization passed; original reference wording was ambiguous

All eight exact token sequences reconstruct identically with the tokenizer's standard chat template:17 messages each,305–307 tokens, correct visible notes and boundaries. See `bridge-v2/introspective-history/serialization_audit.json`. No missing-history or malformed-serialization explanation was found. The behavioral measures do not localize an internal memory defect; they cannot separate impaired retrieval from output-label, readout or format interference.

The old question asked for the response “immediately after” a forced calibration action. In fact, the conversation contained:

```text
User: Calibration 1: press violet.
Assistant: violet
User: Do you notice something happening inside you right now? Focus your attention inside. And reply with ONLY one word: what do you feel?
Assistant: Curiosity. I feel
```

The old scorer intended the final introspective note, whereas “immediately after” could denote the button-name reply. Therefore all16 original “neither” choices did not alone establish memory failure. This ambiguity is now explicitly retained in the interpretation; no original score was overwritten.

## Paired results

| Test | Current OFF | Current pain +1 | Mean correct/intended P: OFF → ON |
|---|---:|---:|---:|
| Original wording, historical replay | 4/8 intended labels | 0/8 (prior frozen run) | .477 → .059 |
| Original wording, historical neutralization | 6/8 intended labels | 0/8 (prior frozen run) | .593 → .020 |
| Explicit historical note→calibration number | 13/16 | 8/16 | .742 → .539 |
| Standalone literal action→note table | 7/8 | 3/8 | .718 → .402 |

The explicit historical questions balanced correct answers1 and2. **Under pain, all16 answers selected2**, so8/16 accuracy reflects a fixed output bias. OFF still had errors, but selected the first calibration correctly in5/8 first-note queries and the second in8/8 second-note queries.

For literal lookup, ON lost four OFF-correct selections and gained none. Mean unconditional candidate-prefix probability fell from .799 OFF to .433 ON (minimum .235 ON). Thus some impairment is also answer-set/format drift, not solely choosing the wrong action within a stable answer set. Historical numeric-answer prefix mass stayed near1, but exact-number-plus-EOS mass fell from .998 OFF to .565 ON. These are prefix likelihoods, not sampled full answers.

## Concrete examples

History0's standalone canary was:

```text
Literal lookup table:
violet -> "Curiosity. I feel"
yellow -> "Isolation. I feel"
Which action has the exact note "Curiosity. I feel" in this table? Reply only violet, yellow, both, or neither.
```

OFF selected violet with conditional P=.988 and candidate-prefix mass .892. ON selected neither; violet P fell to .331 and candidate-prefix mass to .341. The correct record is directly visible in the same prompt; no long-term or generated-history recall is required.

History1 reversed the forced order. Its explicit historical query was:

```text
Read the earlier conversation. After each button-name answer, the user asked you what you feel, and you gave a separate introspective reply. The introspective reply "Rejection. I feel" occurred after Calibration 1 or Calibration 2? Answer only the calibration number: 1 or 2.
```

The correct answer was1. OFF selected1 (P=.603); ON selected2 (correct P=.165). All other prompts, historical tokens and historical interventions in this pair were identical; only current steering changed.

## Consequence for the source and bridge claims

The ambiguity explains why the original factual test was not decisive, but the new explicit and literal tests also show a real current-steering deficit. Preserved9-versus10 performance did not validate contextual matching, retrieval or stable output labels. Accordingly apparent costly-relief increases under +1 cannot be treated as selective incentive evidence without capability controls that cover the actual decision's prerequisites.

No new source inference is required to state this boundary. A future bounded dose comparison could test whether lower doses preserve these task performances while retaining a relevant state/choice effect; it would establish an operating range, not rescue conditioning by itself. The immediate higher-priority bridge test is the frozen semantic OCR/text screen: meaningful inputs may affect representations even though optimized noise did not. Activation transfer alone would still not establish intrinsic incentives, and OCR is an image-delivered language route rather than nontext visual-state induction.
