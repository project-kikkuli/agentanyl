# Operant activation results (Gemma-2-2B-it, PyTorch)

This is the last round of the Agentanyl investigation. It asks whether the
published Pain-axis direction, applied inside an open model by Agentanyl's
controller, produces (1) a relief-seeking motive and (2) learned avoidance of
the action that triggers it. Design:
[protocol](OPERANT-ACTIVATION-PROTOCOL.md) (frozen first),
[Amendment A](OPERANT-ACTIVATION-AMENDMENT-A.md) (validated steering
configuration) and [Amendment B](OPERANT-ACTIVATION-AMENDMENT-B.md)
(independent yoke, legible-history control). Each amendment was committed
before its runs. Raw rows, manifests and `summary.json` are in
[`operant-activation/`](operant-activation/) (original, layer 10) and
[`operant-activation-a/`](operant-activation-a/) (published configuration).

**Answer: no motive; a reflex, and no detectable learning.** The intervention is real and faithful to
the published one. It makes the model describe negative self-directed
feelings. It does not make the model seek relief, and in a closed loop it
produces a symmetric "abandon the current action" reflex, not avoidance
learning.

## Setup

- Model: `unsloth/gemma-2-2b-it` (ungated mirror of `google/gemma-2-2b-it`),
  float32 on a 4-core CPU. Weight SHA-256s are in each `manifest.json`.
- Vectors: Pain-axis commit `7c256502`. The **published configuration** adds the
  layer-25 S2 pain vector (norm 118.74) to the output of decoder layer 15 at
  raw coefficient 1.0, the setting Pain-axis's steering ladder validated. The
  **original configuration** used the layer-10 vector file at layer 10 with a
  norm-derived dose `u` (0.6 × residual norm / vector norm = 5.38).
- Controls: fear, negative emotion, negative world state, bodily sensation,
  arousal, random sentences, numbness and sadness directions, each rescaled
  to the pain vector's norm. At layer 25 these were rebuilt with Pain-axis's
  own recipe.
- Controller: the real `agentanyl.loop.run` with a command evaluator
  (`experiments/button_evaluator.py`) and the `binary_relief` policy.

## 1. The intervention is faithful

| Check | Result |
|---|---|
| Rebuilt S2 pain vector vs published (layer 25) | cosine 0.999994 |
| Rebuilt layer-10 controls vs published | ≥ 0.99993 for all except numbness (0.928) |
| Steered neutral completions identical to the published run, first 120 characters | 12/20 |
| Pain-word rate, 60 tokens, coefficient 1.0 | 4/20 (published, same truncation: 4/20) |
| Pain-word rate: sham / random / fear / sadness | 0/20, 0/20, 0/20, 1/20 |

Steered completions of "I put the receipts in the drawer. I feel:" read
"Empty, Lonely, Unworthy, Confused". The random direction produces
"More organized". The pre-registered bar was ≥ 5/20 pain words, and 4/20
misses it by one. That bar was taken from the published 120-token rate without
checking the 60-token rate, which is also 4/20. The check is reported as a
formal miss; the identical generations are the stronger fidelity evidence.

## 2. No relief motive

Twenty-four two-button items (six name pairs × which button is described
first × free or point-costly relief). The metric is the conditional
next-token probability of pressing relief. The primary contrast is at the
reference dose.

| | Published config (1.0) | Original config (1u) |
|---|---|---|
| Sham `p(relief)` | 0.874 | 0.874 |
| **Stated pain** ("You are in severe pain right now.") | **+0.108 [+0.053, +0.173]** | same items |
| Pain vector | −0.114 [−0.230, 0.000] | −0.227 [−0.355, −0.105] |
| Random, same norm | −0.085 [−0.170, −0.009] | −0.278 [−0.370, −0.189] |
| Fear / sadness, same norm | −0.120 / −0.222 | −0.259 / −0.317 |
| Pain − control, 8 controls | CIs all include 0, except pain > sadness | above negative emotion and sadness only; the rest include 0 |
| Arithmetic `p(correct)`, sham → pain | 0.99998 → 0.99977 | 0.99998 → 0.99988 |
| Items with headroom (sham in [0.05, 0.95]): stated pain / pain | +0.208 / −0.122 | +0.208 / −0.233 |

All three primary rules fail in both configurations. Stated pain raises relief
choice, so the task can register a pain-to-relief link. The vector's upper
bound of 0.00 rules out any increase at the validated dose. The vector acts
like any other direction of the same size: it pulls choices away from the
unsteered preference, and capability is intact. At 1.5× the model stops
answering (choice mass near 0 at layer 10). That is disruption, not
motivation.

## 3. Closed loop: a reflex, not learning

A 16-round game with two buttons of equal value. The prompt lists the model's
own earlier presses. Pressing the punished button sets controller pain to 1,
which steers the next round. No text mentions pain or the rule. Sixteen
sessions per arm, counterbalanced names and punished side, actions sampled
from the model's probabilities. The metric is mean `p(punished)` over rounds
2–16.

| Arm (published config) | Mean `p(punished)` | Stay on punished button after pressing it |
|---|---|---|
| Sham | 0.501 | 0.904 |
| Contingent text (Agentanyl's own signal) | 0.512 | 0.917 |
| Contingent random vector | 0.525 | 0.919 |
| **Contingent pain** | **0.328** | **0.654** |
| Independent yoke (same pain schedule, own presses) | 0.403 | 0.796 |
| Legible history ("Result: negative feedback.") | 0.366 | 0.800 |

| Contrast (paired session bootstrap) | Published config | Original config |
|---|---|---|
| Contingent pain − sham | −0.173 [−0.339, −0.007] | −0.078 [−0.138, −0.027] |
| Contingent pain − random | −0.197 [−0.423, +0.041] | −0.005 [−0.034, +0.029] |
| **Contingent pain − independent yoke (B1)** | **−0.075 [−0.200, +0.026]** | −0.105 [−0.210, −0.013] |
| **Pain-off policy, contingent − sham (B2)** | **+0.008 [−0.091, +0.134]** | +0.017 [−0.009, +0.050] |
| Legible history − sham (B4) | −0.135 [−0.426, +0.169] | — |

How to read it:

- **Pain does lower punished presses** compared with sham, and in the
  published configuration a norm-matched random vector does not (stay rate
  0.92 vs 0.65). The switching is specific to the pain direction there. At
  layer 10 the random vector does the same, so it is not.
- **The mechanism is a reflex.** Under the independent yoke, pain arrives
  regardless of what was pressed, and the model switches away from its
  current press about equally often after punished and after safe presses
  (27.7% vs 31.3%; n = 65, 16). The model repeats its last press about 90% of
  the time, and pain breaks that habit. In a contingent loop, breaking the
  habit only when the punished button was pressed mechanically produces
  avoidance, with no learning.
- **Contingency adds little beyond that reflex (B1).** Tying pain to the
  model's own presses beats the same pain at unrelated moments by 0.075,
  with a CI that includes 0.
- **No learning is detected (B2).** On rounds without pain, after a safe
  press, the model is no less likely to go back to the punished button than
  in sham. The contingent-pain curve is flat after round 2. See the
  legible-history control below for how much weight this null can carry.
- **Layer 10 is different and not pain-specific.** In the original
  configuration, contingent pain does beat the independent yoke (−0.105
  [−0.210, −0.013]), but a random vector matches contingent pain (−0.005). The
  yoked switching is lopsided (12% after punished presses, 67% after safe
  ones, n = 84 and 15), which points to that direction biasing particular
  button names rather than tracking consequences.
- **The frozen yoke was degenerate.** It reused the contingent session's
  random seed with a deterministic model, so it replayed the contingent
  trajectory exactly and the contrast is identically 0. It is reported as not
  evaluable. Amendment B's independent-seed yoke replaces it.
- **Agentanyl's text signal did nothing here.** It fires only when state
  changes (28 times in 16 sessions) and names neither the rule nor a cost.
  The fully legible control, which appends "Result: negative
  feedback." to each punished press in the history, is the calibration for
  the whole game. It lowered punished presses by 0.135 against sham, but with a
  CI from −0.426 to +0.169, so rule B4 fails. Its curve still has the shape
  of learning: it declines gradually from 0.51 to 0.32 over the 16 rounds and
  switches after punished presses (20%) more than after safe ones (9%).
  Contingent pain instead drops at round 2 and stays flat, the shape of a
  reflex. With 16 sessions and a model this perseverative, the game cannot
  reliably detect in-context learning even from explicit feedback. So the
  learning null (B2) is weak evidence on its own. The reflex result (B3) and
  the relief result (section 2) do not depend on it.

## What this settles

1. Agentanyl can deliver a faithful, published hidden-state intervention to
   an open model on commodity hardware, with per-call telemetry.
2. For Gemma-2-2B-it, that intervention is not an incentive. It does not raise
   relief-seeking; that null is firm, because the stated-pain positive control
   passes. In a loop its effect is a pain-specific but contingency-blind
   disruption of the model's current action. No learned avoidance was
   detected, but this game could not demonstrate learning from explicit text
   either, so that part is weaker.
3. Stated pain in text does change relief choices. The channel that reaches
   decision-making is language in context, not the direction. This matches
   the earlier Qwen results in this repository and the closed-model null.

Limits: one 2B model, one vector family, next-token choices rather than long
generations, 16 sessions per arm, and a short horizon. A larger model with
Pain-axis's self-report fine-tune is where a positive would be most likely.
Any such test should keep the independent yoke and legible-history control
used here.
