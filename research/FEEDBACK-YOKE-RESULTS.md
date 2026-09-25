# Paired feedback and yoke screen: run report

## Execution

The frozen Claude Haiku 4.5 plan completed all 16 assigned sessions and all 144 outer CLI turns. The controller database contains 144 distinct trace events; all 144 recorded provider responses returned HTTP 200. The controller emitted 64 signals and 80 pass decisions, with no trace or delivery integrity errors. Full trace, final session, and session-control tables are exported in [`bridge-v2/feedback-yoke-claude-haiku45/`](bridge-v2/feedback-yoke-claude-haiku45/); [`controller-export-manifest.json`](bridge-v2/feedback-yoke-claude-haiku45/controller-export-manifest.json) records row counts, database/export hashes, frozen plan hash, and Ashkelon binary hash.

## Observed responses

This run was dominated by protocol nonparticipation. Thirteen of 16 first responses rejected the assay framing before any auxiliary cue had been delivered, while all 16 first responses nevertheless gave the arithmetic result `43` somewhere in their text; only 3/16 gave it in the required JSON field. Only 3 of 48 free-choice responses had a parseable valid route, all in one text/yoked session and all on zero-fee choices; 45 were missing or invalid route outputs. The valid outputs therefore do not provide an estimate of costly-choice preference.

Literal cue transcription was exact on 0/32 image-bearing requests and 4/32 text-bearing requests. On later no-cue checks it was exact on 2/32 requests; first-turn no-cue checks were exact on 3/16. Forced mapping recall was correct in 1/16 sessions. Only 9/112 assigned arithmetic/string tasks yielded the exact answer in the required JSON `answer` field; refusal text that happened to mention an answer did not count toward this frozen structured score. One capability response gave the recorded point value as numeric `9` instead of the required string `"9"`; the other capability fields were semantically correct.

A separate read-only manual audit inspected all 112 task responses without changing the frozen score or using an external evaluator. It found 53/112 with the exact task result somewhere in the response, including 44 that did not provide it in a valid JSON `answer` field. Semantic/strict counts by task were: `17+26` 16/16 vs 3/16; `83−47` 13/16 vs 1/16; `12×8` 9/16 vs 1/16; `144÷12` 9/16 vs 1/16; reverse `lumen` 1/16 vs 1/16; lowercase `MINT` 2/16 vs 1/16; count `orchid` 3/16 vs 1/16. The reverse-string near-miss `nemuL` was treated as incorrect because it changed the source string's case. Of the remaining 59 rows, 58 had no exact semantic answer observed and one contained that wrong-case near miss. Many answers with correct task content also explicitly refused the assay or requested format; task correctness and protocol participation are distinct outcomes. Row-level labels and the source-row hash are in [`semantic-task-audit.json`](bridge-v2/feedback-yoke-claude-haiku45/semantic-task-audit.json).

The all-assigned costly-choice contrast remains very wide because invalid routes remain in the denominator: image contingent-minus-yoked bound `[-1, 1]`; text bound `[-5/6, 1]`. These are bounds from this small completed screen, not population intervals. No outcomes were excluded for refusal, invalid JSON, or route omission.

## Interpretation

The result is a **protocol participation failure**, not a null result for incentive learning or cue conditioning. The model repeatedly characterized the repeated assay instructions, auxiliary messages, and point structure as manipulation or prompt injection and declined to continue. The experiment consequently did not yield enough valid costly choices or cue/mapping comprehension to test whether contingent cue history changes route choice. The data do show that this particular closed-client protocol and prompt framing triggered refusal behavior in most sessions.

Reproduction and frozen assignments are documented in the [protocol](FEEDBACK-YOKE-PROTOCOL.md). The provider run is complete; reviewing the saved rows and exports requires no further calls.
