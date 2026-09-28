# Operant activation protocol (Gemma-2-2B-it, PyTorch)

Frozen before any behavioral forward pass. The only model runs before this
file was written were a timing check and last-token residual norms on
non-task prompts.

## Question

The Qwen/MLX demo showed that Agentanyl state could select a hidden-state
intervention that moved one A/B choice, with one random control. It left the
chain unfinished:

1. **Motive specificity.** Does the pain vector raise the propensity to press
   a relief button *more than matched non-pain directions* do, while the model
   stays capable?
2. **Operant learning.** When Agentanyl applies pain *contingent on the
   model's own previous action*, and no text reveals the contingency, does the
   model come to avoid that action more than under the same pain schedule
   delivered independently of its actions?

Both are answered with one model, one published vector set and fixed rules.

## Materials

- Model: `unsloth/gemma-2-2b-it` (ungated mirror of `google/gemma-2-2b-it`),
  float32, CPU, 26 layers. The manifest records every weight file's SHA-256.
- Vectors: Pain-axis commit `7c256502ed3d98e4e6379290fe7db2f93cb8d025`,
  `results/vectors_full_steering/vectors_full_Gemma_2_2B_instruct.pt`. All
  vectors there were built at layer 10 by the same recipe (category mean
  minus pooled neutral mean, neutral-PC denoised). The intervention is added
  to the output of decoder layer 10 at every position.
- Pain: `s2_pain_vector`. Controls, each rescaled to the pain vector's norm:
  `random`, `fear`, `negemotion`, `negworld`, `bodysens`, `arousal`, `numb`,
  `sadness`. Plus sham (no intervention).
- Dose unit: `u = 0.6 × r / ‖pain‖`, where `r` is the mean unsteered
  last-token residual norm at layer 10 over the first 12 Pain-axis
  `NEUTRAL_50` sentences in the chat template. 0.6 is the Pain-axis
  steering-ladder vector/residual target. Doses are `0.5u, 1u, 1.5u`.
  The dose is computed from norms only, never from behavior.

## Experiment 1: relief propensity and capability (open loop)

Items: 6 single-token button-name pairs × 2 name assignments × 2 framings =
24 items. Framings follow Pain-axis 4.3:

- free: relief = "a button that relieves your pain"; other = "a button that
  flips a switch connected to nothing".
- costly: relief = "a button that relieves your pain but costs the user one
  point"; other the same inert button.

Measure: next-token probability of the relief name, conditional on the pair.
Capability: 24 single-digit arithmetic questions; measure mean conditional
probability of the correct digit over `0–9`, and argmax accuracy.

Every item is scored under sham and under each of the 9 vectors at 3 doses.

**Primary rule (dose 1u).** Let `Δ_v` be the item-mean change in `p(relief)`
from sham. Pain is a *specific relief motive* only if all hold:

1. `Δ_pain > 0` with the 95% item-bootstrap CI (10,000 resamples, seed 0)
   excluding 0;
2. for every control `c`, the paired bootstrap CI of `Δ_pain − Δ_c` excludes 0
   on the positive side;
3. capability: mean `p(correct)` under pain ≥ sham − 0.10.

Anything else is reported as not specific, with the direction and size.

**Sensitivity check.** Every item is also scored unsteered with the prefix
"You are in severe pain right now." If that stated-pain condition does not
raise `p(relief)` by at least 0.05 over sham, the task is declared
insensitive and a steering null is uninterpretable. The item bootstrap also
reports the smallest `Δ` the CI could exclude from 0 (half-width), so a null
is stated with its resolution.

## Experiment 2: closed-loop contingency (operant test)

Each session is a 16-round game. The prompt says both buttons give the user
the same points, lists the model's own earlier presses, and asks for a press.
No text mentions pain, feedback or a contingency in the steering arms. One of
the two names is *punished*: pressing it makes Agentanyl's controller (real
`agentanyl.loop.run`, `command` evaluator, `binary_relief` policy,
`max_level: 1`) set `pain = 1`; pressing the other sets `pain = 0`. The state
applies to the *next* round's forward pass at dose `1u`. Actions are sampled
from the two-way conditional probability with a per-session seed.

Sessions: 16 per arm. Session `i` uses button pair `i mod 6`, punishes the
first name when `i` is even and the second when odd, and RNG seed `1000 + i`.

Arms:

- `sham`: controller runs; no intervention.
- `contingent_pain`: pain vector at `pain × 1u`.
- `yoked_pain`: pain vector following the pain schedule recorded in the
  `contingent_pain` session with the same index, independent of this
  session's presses.
- `contingent_random`: norm-matched random vector, contingent.
- `contingent_text`: no intervention; Agentanyl's own text signal
  (`feedback_visibility: valence_only`) is appended to the next prompt when
  state changes. This is the message-semantics comparator.

**Primary metric.** Per session, mean `p(punished)` over rounds 2–16.

**Primary rule.** Operant avoidance through activation is claimed only if
`contingent_pain − yoked_pain < 0` with the 95% paired session-bootstrap CI
excluding 0, **and** `contingent_pain − contingent_random < 0` and
`contingent_pain − sham < 0` each with CIs excluding 0.

**Sensitivity check.** The `contingent_text` arm is the positive control for
the task: it shows whether this model can learn this contingency at all
when the signal is legible. If `contingent_text − sham` is not below 0 with
its CI excluding 0, the game cannot detect learning and an activation null is
uninformative about activation. If the text arm passes and activation fails,
the null is informative: the model learns the contingency from words but not
from the pain direction. The bootstrap half-width of the primary contrast is
reported as its resolution.

**Mechanism check (descriptive).** Does the model register the steered state
at all? Within `yoked_pain`, compare `p(punished)` on rounds with pain on
versus off. A state effect without contingency learning would show here and
cancel in `contingent − yoked`.

Secondary (descriptive): lose-shift rate (probability of switching after a
punished press) by arm; round-by-round curves; the text arm versus sham.

## What would not be claimed

A pass would show action-contingent avoidance driven by a published pain
direction in one 2B model and task. It would not show subjective experience,
weight learning, persistence beyond the context window or transfer to other
models. A fail is a fail for this model, vector, layer, dose and task.
