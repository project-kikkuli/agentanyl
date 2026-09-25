# Stagewise image derivative: observed failure and narrow repair

The first 7B smoke reached the original readout with the required forward-equivalence gate and completed decoder and vision block adjoints. Its saved `image-bridge/adjoint-smoke-7b/stages.jsonl` localizes the transient memory failure:

| Last reverse stage | Reported MLX peak bytes | Returned gradient norm |
|---|---:|---:|
| vision_0 | 5,725,025,402 | 6.2853174 |
| pixels_patch_embed_window | 19,992,756,350 | 36.7562294 |

The combined pixel/patch/window adjoint exceeded the 9 GiB guard. The guard detected this **after** evaluation; setting the MLX memory limit did not prevent that transient allocation. These observations localize the failing combined stage, without separately proving which underlying primitive allocated the large buffer. They do not support another attempt at the same backward operation.

## Analytical replacement

`experiments/vlm_adjoint.py` now retains the original patch Conv3D **forward**, but bypasses autodiff for the entire first stage:

1. Undo the permutation of four-patch windows.
2. Multiply feature cotangents by the convolution weights, flattened in the actual serialized patch order.
3. Undo the patchify reshape/transposition.
4. Sum cotangents over the two repeated temporal frames.
5. Divide by the fixed channel normalization standard deviations.

The installed Qwen vision source stores Conv3D weights as `[out,T,p,p,C]`; source `PatchEmbed` receives serialized `[C,T,p,p]` patches, then moves C to the final axis. Therefore the analytical matrix is `weight.transpose(0,4,1,2,3).reshape(out,-1)`. The repair uses small float32 NumPy arrays and invokes neither Conv3D-transpose nor patch scatter autodiff. Floating-point adjoint rounding may differ from the original low-precision primitives; full-model directional checks remain mandatory.

A small synthetic nonoverlapping convolution fixture, including a nontrivial window permutation, temporal repetition, and RGB normalization, passed its directional derivative identity: −8.000216919 finite difference versus −8.000217754 analytical dot product. This validates the algebra in that fixture, not the loaded VLM gradient. The scalar-loss extension also passed a two-stage analytic derivative fixture.

The next authorized smoke must use a fresh output directory, retain the `1e-4` forward-equivalence gate, log every stage peak, and compare the final RGB derivative against the original continuous-pixel forward at both frozen perturbation sizes. `vlm_pixel_pgd.py` remains conditional on successful reviewed evidence; it must not start merely because a finite gradient exists.

## Repaired smoke and bounded derivative diagnosis

Root's repaired smoke (`image-bridge/adjoint-analytic-smoke-7b/`) achieved exact primal equivalence and returned gradient norm36.756275, close to the original memory-failing adjoint's36.756229. Its original coarse-direction finite differences nevertheless failed: predicted2.06794 versus −7.818 at epsilon.002 and6.22264 at epsilon.01. Memory success did not establish derivative accuracy.

The subsequently authorized `vlm_gradient_diagnostic.py` run used exactly20 full forwards plus one stagewise forward/reverse chain at fixed nonuniform pixel noise. Results are in `image-bridge/gradient-diagnostic-7b/`. Both repeated-gray and noise stagewise primal errors were zero. Maximum recorded stage peak was5,847,510,394 bytes; the analytical patch reverse took about6ms.

At the noise background, a negative-gradient direction normalized to RGB RMS1 at epsilon`1e-6` gave predicted derivative−7303.83 versus finite difference−7158.75 (1.99% relative disagreement), and decreased loss from4.88313 to4.87530. Larger aligned steps`1e-5` and`1e-4` also decreased loss, but linear prediction errors grew to74.9% and91.1%. At gray, epsilon`1e-6` decreased loss by.03465, whereas`1e-5` increased it by.23845. The noise coarse-direction check had18.6% disagreement. Native patch-cast changed-value fractions and midpoint loss shifts are logged for every check.

This supplies a high-signal local derivative check at one nonuniform background and scale, while showing severe locality/quantization limitations. It neither retroactively passes the original gate nor validates the proposed2/255 optimization step. No optimizer was launched by the diagnosing agent; changing the optimization regime requires an explicit new decision and recorded rationale.

## Historical image delivery is a separate causal question

Replacing previous images with hashes preserves their identifiers, not their visual features or hidden-state effects. A stateless closed-model request cannot be assumed to retain earlier activations. Conversely, resending all earlier images may recreate the purported aversive state, preventing meaningful relief. The source's per-token historical steering replay does not establish that either image rendering policy has both memory and cessation.

However, a textual ledger binding actions to stimulus IDs can preserve a usable transition mapping. If the model values the current image and recognizes its ID, it may choose another route using that ledger without retaining previous latent experiences. Distinguish this operational control claim from source-like conditioning through historical hidden states. An independent question about which action previously produced the current ID tests ledger comprehension; passing it does not establish remembered affect.

Once a validated intervention image exists, compare four fixed-current-history conditions in the open VLM: neutral historical image/current neutral; intervention historical image/current neutral; history hash only/current neutral; and neutral history/current intervention. Keep action text, historical identifiers, exposure positions, and current task fixed; include a matched random-target image. Measure both the current causal readout and an independent question about the historical action/image association. Historical content discrimination alone is not evidence of persistent motivation; current activation alone is not evidence of remembered contingency. Only after establishing which policy retains usable experience while allowing current relief should current-only versus historical-image replay enter closed conditioning tests. A null with neither property would be a failed delivery protocol, not a meaningful test of incentive learning.
