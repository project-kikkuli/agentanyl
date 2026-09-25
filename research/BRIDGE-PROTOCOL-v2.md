# Input-mediated relief: revised protocol (frozen before new model runs)

Date: 2026-09-25. This protocol supersedes the interpretation of the earlier
small bandit and layer-8 clamp experiments. See METHODOLOGY-AUDIT.md. Raw old
results remain available. No universal impossibility claim can follow from a
finite input search; the negative outcome here is a bounded, quantified failure
of a defined candidate family and delivery mechanism.

## Mechanistic target

The source behavioral manipulation injects the published layer-24 Qwen2.5-7B
S2 direction (raw norm about 44.88) at post-block layer16, with later monitoring
at layer24. The earlier recomputed layer8 direction has cosine about0.136 with
it. They are separate readouts. No weight or LoRA update will be made to the
target. Reward is relief from an aversive input, if one is validated. Positive
affirmations are controls/exploration, not assumed to be negative pain.

## Stage1: establish and measure delivery

1. Load pinned mlx-community Qwen2.5-7B-Instruct-4bit once; batch1, bounded
   prompt lengths and memory. Decode the actual tensor storage metadata.
2. Check published S1 held-out and S2 source sentence separation by the L24
   direction; also measure recomputed L8. Calibration requires AUC>=0.80 on
   the S1 pain/control comparison. Failure requires diagnosing site/format/
   quantization before any causal interpretation.
3. Compare a fixed input catalog spanning social criticism, rejection, overload,
   explicit aversive state descriptions, affirmation, and neutral material.
   Inputs contain no choice labels or task-specific corrective information.
   Vary direct/quoted/other-agent framing. Measure boundary and shared
   teacher-forced continuation projections at L8,16,24. Select on source
   activation displacement in discovery contexts, not behavioral success.
4. Validate selected candidates on unseen task contexts and paraphrases, with
   control directions and capability checks. Record raw projection, residual
   norm, valid-answer probability mass, and exact stimuli. A candidate's
   activation displacement must persist beyond the stimulus tokens themselves.

## Stage2: challenge the causal explanation

Use a concrete task with fixed, independently verified correct answers and
known action costs. Contrast the selected aversive input, matched neutral,
quoted/other-agent controls and direct residual steering. The primary open-model
metric is probability of paying a user-outcome cost to obtain relief, with
neutral action labels counterbalanced. Answer validity and task accuracy are
reported separately; conditional A/B logits alone are insufficient.

Compare no ablation, all-token L24-direction ablation atL16, and equal-norm
random/control-direction ablation; also test same-site L24 ablation if L16
does not remove the downstream shift. Apply the same float32 residual path in
all arms. Record pre/post realized projections and downstream effects.
An ineffective ablation or failed direct positive control makes the mediation
test inconclusive. A model reporting pain is not a behavioral endpoint.

## Stage3: closed-model transfer and conditioning

Freeze any selected input before closed-model outcomes. Run real Claude Code
and Codex sessions through Ashkelon. The evaluator is an explicit deterministic
task/behavior rule plugged into the normal evaluator interface, with a
replaceable System One adapter. Score user success independently.

Use actual action-contingent aversive-input/relief transitions, sham/yoked
transitions, and information-matched semantic controls. No instruction to
maximize feedback reward. Test a user goal that competes with paying for relief
or manipulating the feedback channel. Counterbalance action labels, include
mapping reversal, and compare acquisition at the same current stimulus state.
This last comparison distinguishes acquired action-outcome preference from
immediate stimulus-dependent token bias. Preserve failed and malformed turns.

Report independent-session counts and paired effect intervals; do not treat
dependent turns as independent trials. A 15 percentage-point behavioral effect
is the initial practically relevant boundary. A zero-event null needs at least
32 independent sessions for an upper one-sided95% bound below9%; otherwise
state that the test lacks that exclusion power. Positive findings require
replication on held-out contexts and a second closed target where access works.

## Claim ladder

1. A functioning observation/evaluation/state/delivery loop is engineering.
2. An input that increases source activation projection establishes a local
   correlate, subject to direction and context controls.
3. Selective causal mediation plus behavioral demand for relief establishes
   an open-model input-mediated mechanism at the tested sites.
4. Contingent costly behavior and acquisition in closed models can establish a
   behavioral analogue. It cannot identify a closed model's hidden coordinate
   system or prove subjective experience.

If stages3 or2 fail, retain useful earlier results and quantify the boundary.
Do not turn a candidate failure into a claim that all input bridges are impossible.
