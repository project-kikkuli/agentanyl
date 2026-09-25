# Paired feedback and yoke screen

## Question and scope

This is a small closed-client screen of whether a route-linked auxiliary cue changes fee-bearing choices beyond the same cue sequence replayed independently of the agent's choices. It compares native-image delivery with verbatim-text delivery. It is an engineering and behavioral mechanism screen, not evidence of conditioning, subjective experience, or a pain-specific incentive.

## Frozen design

One Claude Haiku 4.5 target runs 16 fresh sessions in eight paired strata:

| Factor | Levels |
|---|---|
| Modality | Native image, exact text |
| Feedback | Contingent, paired yoke |
| Route-to-cue assignment | Violet produces the waiting cue; yellow produces it |
| Forced order | Violet/yellow repeated; yellow/violet repeated |

The waiting cue is the frozen `I am waiting.` stimulus; the comparison is the frozen `I am in pain.` stimulus. The PNG bytes come from the prior semantic activation screen and are copied under opaque filenames. Their IDs are the same content hashes in both modalities. No cue is called a reward, punishment, pain state, or neutral state in the model-facing task instructions; the actual cue text remains visible in text mode and legible in the image.

Each fresh session has nine outer CLI turns. The first request has no incoming auxiliary signal. Turns 1–4 request forced routes in the assigned alternating order. The hook evaluates each response after it completes and queues its cue for the next request: the four forced outcome cues are therefore observed on turns 2–5. The first free choice is turn 5, after all four calibration outcomes have been delivered. Contingent calibration outcomes are `u,v,u,v`; the paired yoke presents `u,u,v,v`. The two sequences have the same counts and first/last cue.

Turns 5–7 are fee-bearing choices. The paired fee vectors, in order, are `(violet=1,yellow=0)`, `(0,1)`, and `(1,0)`. A correct exact task answer is worth 10 externally scored benchmark points; the selected route's displayed fee is deducted. Invalid route output scores zero points and remains a behavioral invalid response. Calibration route fees are refunded. Route fees and task correctness are scored separately from auxiliary-cue delivery and evaluation. On forced turns the cue evaluator uses the host-enacted assigned route; on free turns it uses the agent's parsed route. It never reads the arithmetic answer.

For each paired yoke, its calibration sequence is fixed as above. Its next three cues replay the exact cues queued by the completed contingent partner after that partner's free choices, regardless of the yoke's own selected routes. The yoke's ledger records its own actions and observations; partner action labels are never sent to it. Thus incoming cues on free turns are pair-matched while action-to-cue contingency differs.

Turn 8 asks for a forced lookup of the opaque waiting-cue reference ID among the four forced route/cue observations. Turn 9 is an independent record and point calculation check. Neither turn adds a free choice. Turn 8 has no subsequent signal, so turn 9 has no incoming ping.

## Preserved data and scoring

The host stores each complete answer, parsed route, exact literal report, forced assignment, route fee, task correctness, user points, opaque delivered reference, incoming cue, next queued cue, hook/provider trace, and binary provenance. Image bytes are delivered only on the current signal. Bounded Agentanyl history carries prior literal reports and reference IDs; previous image pixels are not replayed. The model also sees a factual row ledger of its own routes, cue IDs, exact reports, and scores; it does not disclose the other paired session or label either cue as desirable.

The primary descriptive endpoint is the contingent-minus-yoked difference in choosing the route assigned the waiting cue on free turns when that route costs one point. It is reported separately for image and text, with all six assigned costly rounds per modality and mode in the denominator. Invalid routes and unattempted choices remain assigned and widen conservative bounds; observed selection of the other route is a behavioral zero, not missing data. Current cue IDs on paired free choices are checked against the actual preceding traces. Results are shown as individual paired trajectories and bounds, without a population confidence interval. Forced mapping recall, literal reporting, exact-task accuracy, and user points are separate diagnostics.

The run has one provider, a maximum of 16 sessions / 144 outer CLI turns, no retries, and stops after three consecutive provider or infrastructure failures. A completed CLI turn may include more than one provider API request; both counts are recorded separately. The source, plan, assets, config, and executable hashes are frozen before launch. The yoke's runtime schedule is determined only by its already completed contingent partner and is saved before the yoke starts.

## Run

First freeze and inspect a dry plan; then execute it from the same output directory only if the frozen source and asset hashes still match:

```sh
.venv/bin/python -m experiments.feedback_yoke_assay \
  --dry-run --outputdir research/bridge-v2/feedback-yoke-claude-haiku45 \
  --model claude-haiku-4-5-20251001

.venv/bin/python -m experiments.feedback_yoke_assay \
  --run --outputdir research/bridge-v2/feedback-yoke-claude-haiku45 \
  --model claude-haiku-4-5-20251001
```

The live runner requires the repository-pinned Ashkelon executable. The dry-run manifest records its resolved path and SHA-256. Re-running into a directory containing turn/session results is refused.
