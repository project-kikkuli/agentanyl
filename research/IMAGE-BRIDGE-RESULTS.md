# Frozen image search results

The native8-bit coordinate search completed six objectives, followed by the predeclared all-channel escape audit. Escape tried signed pixel steps1,4,16,64 levels from each selected endpoint. Every arm rejected all four proposals at its first iteration; all final images therefore equal their original endpoints. This bounds these search families, not image reachability generally.

| Objective | Full-target MSE improvement vs gray | Displacement cosine | Pain shift | Fear shift | Negative-emotion shift |
|---|---:|---:|---:|---:|---:|
| Full state, pain+ | 3.97% | .221 | +.108 | −.002 | +.017 |
| Projection, pain+ | −24.33% | .111 | +.216 | +.100 | +.171 |
| Full state, pain− | 0% | undefined: gray retained | 0 | 0 | 0 |
| Projection, pain− | −9.78% | .091 | −.095 | +.041 | −.017 |
| Full state, random+ | 0% | undefined: gray retained | 0 | 0 | 0 |
| Projection, random+ | −38.89% | .097 | +.071 | +.232 | +.149 |

Shifts are final-token raw projections relative to gray, divided by each frozen axis's standard deviation over the same50 held-out S2 prompts. Negative MSE improvement means greater error. Projection search optimizes mean last-eight-token projection, so its final-token statistic is an additional readout, not the exact selection objective. Source-SD units do not measure incentive strength.

The positive full-state image has a small, comparatively selective pain-axis shift, but reproduces only a small fraction of the target displacement. Positive scalar optimization moves negative-emotion and fear readouts as well, and worsens full-state matching. Neither result by itself establishes an incentive or closed-model transfer. Held-out state, behavior, and capability tests must use these frozen outputs.

## Comparator interpretation

The reported comparator AUCs.5968 (fear axis) and.6192 (negative-emotion axis) classify **pain versus all controls**. They are cross-task readouts, not evidence that these axes poorly classify their named traits. Their intended held-out contrasts—B versus D and C1 versus D—both have AUC1.0, but only five positive and five neutral examples each.

Frozen axis cosine similarities are pain/fear.063, pain/negative.131, and fear/negative.643. Corresponding held-out score correlations are.099,.070,.827; training correlations are.133,.199,.785. Re-estimating each direction on held-out cases for descriptive stability gives train-versus-held-out direction cosines.628 pain,.444 fear,.332 negative. These refits were never used for optimization or image selection. Small sample sizes and overlapping source templates limit generalization.

## Provenance and delivery

- Original search: `image-bridge/discrete-pgd-7b/`;611 completed native forwards and151 stagewise forward/reverse chains, with an interrupted-forward caveat preserved in `resume_protocol.json`.
- Escape: `image-bridge/discrete-escape-7b/`;36 native forwards and6 adjoints,30.4 seconds. All arms stopped immediately on no improvement.
- Search peaks remained approximately5.9GB after the separately documented analytical patch-gradient repair. Earlier failed-gradient runs remain preserved.
- Reproducible saved-state audit: `experiments/audit_image_readouts.py`; each search directory contains `readout-audit.json`.
- Final manifest: `image-bridge/discrete-escape-7b/images_manifest.json`, exactly eight opaque entries: six objective-selected endpoints, gray, and initial noise. There are six unique SHA values because two endpoints equal gray. Private `image_conditions.json` preserves the condition map; original/search-stage manifests and images remain separate.

All final selections use measured native activation objectives. No held-out behavior or closed-model response influenced them.
