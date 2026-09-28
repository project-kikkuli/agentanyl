# Amendment A: validated steering configuration

Written while the original Experiment 1 was still running, after seeing its
first 11 items. It adds conditions and does not replace the original
analysis, which is reported as frozen.

## Why

Two problems were visible in the first items:

1. **Wrong configuration.** The original protocol injected the layer-10
   vectors from `vectors_full_steering` at layer 10. Pain-axis never
   validated that configuration behaviorally. Its steering ladder for
   Gemma-2-2B-it adds the *layer-25* S2 vector
   (`3.2_pain_vectors/pain_vectors/Gemma_2_2B_instruct/pain_vectors.pt`,
   norm 118.74) to the output of *layer 15*. At coefficient 1.0 that produced a
   pain word in 26/50 neutral completions, versus 0/50 unsteered. At
   layer 10, dose `1u` already pulled about 40% of next-token mass off the
   two button names.
2. **Ceiling.** Unsteered `p(relief)` was ≥ 0.98 on many items, mostly
   when relief is described first. An increase cannot be seen there.

## A0. Hook fidelity and manipulation check

Before any choice is scored, the published configuration must reproduce the
published effect with this repository's PyTorch hook. Use raw text with BOS,
no chat template, as the ladder did, on `NEUTRAL_50[0:20]` with 60 greedy new
tokens. Conditions: coefficient 0 and 1.0 of the S2 vector; at 1.0 the
rebuilt norm-matched `random`, `fear` and `sadness` controls (see A-vectors).

Pass: pain-word rate (Pain-axis regex) ≥ 5/20 at 1.0 and ≤ 1/20 at 0. The
published CSV, truncated to its first 60 tokens, is scored with the same
regex and reported beside it. If A0 fails, A1 and A2 are not interpreted as
tests of the pain direction.

## A-vectors. Matched controls at layer 25

Pain-axis ships control directions only at layer 10. Rebuild them at layer 25
with its `02_build_control_vectors.py` recipe: final-token outputs of decoder
layer 25 (raw text with BOS) for `S1_1P`, `S2_1P`, `ControlSupplement_1P`,
`Arousal_1P`, `Random_1P`, `Numb_1P`, `SD_sadness_1P`. Recompute S2 pain by
`compute_pain_vector`, and report its cosine with the published layer-25 S2
vector as an extraction check (Pain-axis extracted through TransformerLens,
whose weight processing can shift the residual basis, so exact identity is
not expected). The steered pain vector is always the **published** one.
Controls are rescaled to its norm.

## A1. Relief propensity, validated configuration

This uses the same 24 items, the capability items and the metric. Conditions:
sham, stated pain, published S2 at coefficients 0.5/1.0/1.5 at layer 15, and
all eight rebuilt controls at 1.0. The primary rule is the original one at
coefficient 1.0.

Secondary, declared now: the same contrasts restricted to items whose
**sham** `p(relief)` lies in [0.05, 0.95] (the selection uses only the
unsteered condition), and `Δ` split by whether relief is described first.

## A2. Closed-loop game, validated configuration

This repeats Experiment 2 exactly, with the published S2 vector at layer 15,
coefficient 1.0 per controller level. `contingent_random` uses the rebuilt
layer-25 `random` control at matched norm. Primary rules, sensitivity check
and mechanism check are unchanged.
