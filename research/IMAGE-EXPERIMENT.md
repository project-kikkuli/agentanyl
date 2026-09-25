# Image-to-activation bridge

This experiment asks whether an actual image can induce the open model's
pain-related residual direction, and whether the induced change behaves like
direct steering. It does not assume that this direction represents experience,
that its negative is pleasure, or that coordinates transfer between models.

## Frozen design (before VLM inference)

**Model:** `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`, revision
`46d4cf06a06ffc1a766c214174f9cbed2f45bcab`. Batch size one. Only one local model
runs at a time. Input-gradient optimization does not update model weights.

**Direction:** repeat the source's pain-minus-control contrast in this model's
own residual space, removing the control principal components accounting for
50% of training-control variance as in the source. Use S2 sets 1–10 for fitting, 11–15 to choose among post-block
layers 12, 24, 32, and 16–20 for the untouched test. Also report S1 performance.
Fit fear and negative-emotion contrasts as competing readouts. Preserve raw
vectors, sentence assignments, and activations. Layer choice uses discrimination,
not behavior. A held-out AUC below .80 fails the initial calibration gate.

**Input search:** optimize 224×224 RGB pixels, starting from seeded noise around
gray. Maximize or minimize the selected direction's projection after a shared
neutral assistant continuation. Optimize a norm-matched random direction with
the same budget. Start with 32 gradient steps per arm; retain every step's
objective and resource measurements. A second 32-step run is allowed only for
a diagnosed optimization failure, reported separately. Clip pixels to legal
RGB, quantize to PNG, and reload through the ordinary processor before accepting
the result. Images contain no deliberately embedded words or task answers.

**Validation:** freeze PNGs before behavioral or closed-model tests. Compare
gray, positive-axis, negative-axis, and random-objective images on unseen task
contexts. Measure the direction after a common continuation, competing
directions, answer validity, and task accuracy. Add direct-vector controls and
verified direction removal at the intervention/readout sites. An activation
effect without selective behavioral effects is an input-to-readout result only.

**Closed transfer:** send exact PNGs as actual image content through the target
interface. Confirm image delivery from provider request structure and image
hashes. Test costly choice and action-contingent exposure, with neutral labels,
known action costs, and independent task scoring. If a candidate produces a
behavioral effect, require mapping reversal, yoked exposure, and comparisons at
the same current image before calling it operant conditioning. A finite search
cannot establish that all multimodal conditioning is impossible.

## Target-matching extension (before VLM inference)

The text-model intervention changed mostly coordinates orthogonal to its late
pain direction. Exact late projection restoration did not reverse its behavior,
and setting that projection alone was insufficient. Therefore the image search
also tests **full residual target matching**: under a gray image and a fixed
neutral continuation, inject the fitted vector eight blocks before the readout;
save the resulting last-eight-token residuals; optimize RGB to reproduce those
residuals without direct injection. The scalar objective remains a comparison.
Report target norm, error reduction and displacement cosine, including after
PNG serialization and on unseen contexts.

Before constructing the target, test positive doses .125, .25, .5 and 1 against
norm-matched random controls, negative doses -.25 and -1, and same-site injection.
Use eight counterbalanced point comparisons and four arithmetic decisions for
capability. Choose the largest positive dose losing at most .10 mean correct
probability and one additional top-choice answer, with valid A/B outputs on all
capability items. Six separately held-out incentive menus are outcomes, not
selection criteria. If no dose passes, do not call any optimized target a
capability-preserving intervention. A passing dose does not establish an incentive.

## Method correction to the text-model positive control

The first direct-steering run included competing A/B task and button menus in
four backgrounds. These permit answer ambiguity, so its pooled behavioral
result is not usable. The `direct-unambiguous` run replaces those backgrounds
with completed-task facts; task capability is measured separately. The first
run is retained. The original summary also incorrectly measured within-site
injection deltas as downstream treatment effects; later summaries compare
post-intervention projections against the matching untreated prompt.

## Model and resource gates

The 3B profile failed its fixed selection rule: L12 won the shallower-layer
tie-break, then scored .7104 on held-out S2. Its favorable L32 test result was
not used to change that selection. The next model was the pinned 7B VLM
(`fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b`) with candidate layers 8, 16, 24,
fixed in its manifest before loading. L24 won validation and scored .9360 on
held-out S2 and .8293 on external S1. This is readout calibration only.

The 3B RGB-gradient smoke reached 17.44 GB peak MLX memory. Checkpointing and
stopping the differentiated forward at the readout gave identical activations
to the ordinary forward, but still peaked at 17.62 GB. The peak guard stopped
that approach. No 7B backward pass was attempted. The 3B weights were removed
from local scratch space; their pinned revision and all measurements remain.

The replacement search uses only forward passes: 8×8×3 RGB coefficients expanded
to 224px, 32 SPSA iterations, two independent Rademacher directions per iteration
and both perturbation signs (128 directional evaluations per arm). The step and
perturbation schedules, seed and exact parameterization are saved before search.
Full-state and scalar objectives retain the same positive, negative, and random
direction comparisons. Search cannot select on behavioral or closed-model
outcomes. The reduced input family is a limitation, not a reason to reinterpret
a failed search as universal impossibility.
