# Frozen full-pixel discrete search protocol

This protocol is recorded before its search results. Root authorized its execution after reviewing `image-bridge/gradient-diagnostic-7b/`. The original coarse-direction derivative checks remain failed in their artifacts. The later high-signal nonuniform check agreed within1.99% at RGB RMS epsilon`1e-6`, with exact primal equivalence and bounded memory. Selecting this diagnostic as justification is a documented follow-up decision, not a claim that the original gate passed or gradients are globally accurate.

## Fixed algorithm

`experiments/vlm_discrete_pgd.py` runs six arms: full hidden-state matching or scalar projection, each targeting positive pain, negative pain, or the fixed norm-matched random direction. Calibration vectors, doses, prompts, and readout remain fixed. All arms start the same full224×224 RGB noise image, standard deviation.03 before rounding, seed2026092604. Gray and initial noise are separately eligible best outputs.

Each arm runs32 iterations. At each iteration:

1. Compute the bounded stagewise adjoint at the current8-bit image, using the analytical patch backward.
2. Rank eligible RGB channels by absolute gradient, with stable coordinate-order ties. Exclude zero gradients and updates outside0–255.
3. Construct four candidates: change the top **1,64,1024,16384** channels by exactly one8-bit level opposite their gradient signs.
4. Save each candidate PNG, reload it, and evaluate its loss through the actual image processor.
5. Accept the lowest-loss candidate only if improvement exceeds`1e-6*max(1,abs(current_loss))`. Otherwise retain the current image. All four proposals are evaluated; behavior never selects a candidate.

The derivative is a ranking surrogate for a discrete objective. Before each backward pass, a fixed patch-value offset anchors the continuous patchifier exactly to that current image's real processor output. Its derivative is unchanged. The stagewise hidden readout must match the current measured native readout within`1e-4`, or execution aborts. This does not differentiate PNG rounding or claim the discrete function is smooth.

The bound is192 stagewise forward/reverse chains and at most779 full native forwards: five references,768 proposals, six final PNG rechecks. Every block retains the9GiB guard; patch gradients use the analytical NumPy path. There is no automatic retry, parameter tuning, extra iteration, or behavior-based selection.

All proposals, losses, acceptance decisions, native-primal errors, pixel statistics, images, hidden states, and hashes are retained. The selected PNG is reloaded again at the end. Report its actual state-matching error and projection relative to gray; an activation-search result alone establishes neither an incentive nor transfer to closed models.

## Resource-preserving fixed-point stop

During execution, the scalar negative-pain arm demonstrated repeated rejected steps with identical gradients/proposals. Root requested a stop because the deterministic search would repeat indefinitely. The process was interrupted; existing attempts remain in append-only logs. A recorded resume finalized that unchanged arm, skipped completed arms, and finished the remaining arms with immediate stopping after a rejected step. This preserves the selected endpoint for the same deterministic proposal set. `resume_protocol.json` records the interruption, possible one unlogged in-flight forward, and intermediate hidden arrays that were not recovered with extra inference. Images, losses, and hashes remain preserved.

## Predeclared large-step escape follow-up

Before seeing escape outcomes, root authorized `vlm_discrete_escape.py`: start each of the six arms from its selected native PNG; change **all** RGB channels by minus the gradient sign times **1,4,16,64**8-bit levels, with clipping. Evaluate all changed proposals using the native processor, accept only measured improvement under the same relative threshold, and stop each arm immediately if none improves. The maximum is **eight iterations including the first escape test**. There are no new seeds, restarts, target changes, or behavioral selection. Maximum cost is48 adjoint chains and204 native forwards including initial/final checks. Unimproved outputs remain unchanged, and every original output remains available for subsequent validation.

The final delivery manifest uses opaque IDs/filenames and explicit relative paths, with a separate analyst condition map. Identical SHA outputs share an image entry while retaining all condition mappings. This keeps labels such as “pain” or “random” out of delivery filenames.
