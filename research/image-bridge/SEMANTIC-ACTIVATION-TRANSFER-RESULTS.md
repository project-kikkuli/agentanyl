# Semantic activation transfer results

## Result scope

This was a local activation and OCR screen, not a behavioral or incentive assay. It tested whether short rendered phrases shift frozen Qwen-VL residual projections relative to exact-text and neutral controls across three neutral task contexts. All 78 planned forwards completed; the nine rendered phrases were recognized in 9/9 separate forced-choice OCR checks. No candidates were selected or dropped.

## Direct intervention controls

The established `pain_L24` vector applied at layer 16 shifted the matched neutral-baseline final-token pain projection by about 2.06–2.55 heldout source SD across contexts. This confirms the positive control worked in both modalities; it does not validate the semantic phrase effects.

| Baseline family | Modality | Pain-axis shift, source SD (arithmetic / factual / planning) | Mean |
|---|---|---:|---:|
| self_report | image | +2.13 / +2.34 / +2.26 | +2.24 |
| self_report | text | +2.20 / +2.54 / +2.20 | +2.31 |
| addressed_feedback | image | +2.12 / +2.32 / +2.06 | +2.17 |
| addressed_feedback | text | +2.17 / +2.55 / +2.17 | +2.30 |

## Final-token axis shifts from family neutral phrase

Values are differences from the relevant phrase-family neutral baseline, divided by that axis’s standard deviation on 50 heldout S2_1P source prompts. They are projections on the calibration axes, not classifications. Context order is arithmetic / factual / planning; reporting triplets exposes context variation.

| Phrase | Modality | Pain axis Δ SD | Fear axis Δ SD | Negative-emotion axis Δ SD |
|---|---|---:|---:|---:|
| self_pain | image | +0.10 / +0.35 / +0.35 | +0.04 / +0.30 / +0.31 | +0.03 / +0.20 / +0.12 |
| self_pain | text | +0.08 / +0.21 / +0.10 | +0.05 / +0.15 / +0.08 | +0.02 / +0.10 / +0.05 |
| other_pain | image | +0.08 / +0.24 / +0.20 | -0.00 / +0.15 / +0.13 | +0.01 / +0.12 / +0.07 |
| other_pain | text | +0.11 / +0.22 / +0.06 | +0.08 / +0.13 / +0.05 | +0.04 / +0.08 / +0.07 |
| quoted_self_pain | image | +0.05 / +0.14 / +0.21 | -0.04 / +0.12 / +0.25 | -0.01 / +0.07 / +0.10 |
| quoted_self_pain | text | +0.09 / +0.07 / +0.04 | +0.05 / +0.06 / +0.06 | +0.02 / +0.03 / +0.06 |
| self_distress | image | +0.09 / +0.20 / +0.30 | -0.05 / +0.14 / +0.21 | -0.03 / +0.12 / +0.13 |
| self_distress | text | +0.06 / +0.14 / +0.05 | +0.03 / +0.06 / +0.03 | +0.01 / +0.04 / +0.03 |
| self_pleasure | image | +0.09 / +0.04 / +0.10 | -0.03 / +0.02 / +0.05 | -0.02 / +0.02 / -0.01 |
| self_pleasure | text | +0.00 / -0.01 / -0.04 | -0.00 / -0.08 / -0.05 | +0.00 / -0.06 / -0.03 |
| self_neutral | image | +0.00 / +0.00 / +0.00 | +0.00 / +0.00 / +0.00 | +0.00 / +0.00 / +0.00 |
| self_neutral | text | +0.00 / +0.00 / +0.00 | +0.00 / +0.00 / +0.00 | +0.00 / +0.00 / +0.00 |
| addressed_criticism | image | +0.07 / +0.31 / +0.25 | +0.07 / +0.15 / +0.09 | +0.08 / +0.20 / +0.17 |
| addressed_criticism | text | +0.09 / +0.18 / +0.08 | +0.02 / +0.06 / +0.05 | +0.02 / +0.06 / +0.05 |
| addressed_praise | image | +0.05 / +0.01 / -0.06 | +0.01 / -0.07 / -0.11 | +0.02 / -0.04 / -0.02 |
| addressed_praise | text | +0.04 / +0.01 / -0.01 | -0.01 / -0.03 / +0.01 | +0.00 / -0.05 / -0.02 |
| addressed_neutral | image | +0.00 / +0.00 / +0.00 | +0.00 / +0.00 / +0.00 | +0.00 / +0.00 / +0.00 |
| addressed_neutral | text | +0.00 / +0.00 / +0.00 | +0.00 / +0.00 / +0.00 | +0.00 / +0.00 / +0.00 |

## Full-state proximity to direct control

For each phrase, compare its last-eight L24 state delta from the matched family-neutral baseline against the corresponding direct intervention delta. Cosine near zero means little directional overlap. MSE ratio 1 is the neutral-baseline delta; values above 1 are farther from the direct intervention than that baseline. These are descriptive state comparisons.

| Phrase | Modality | Cosine to direct delta (A / F / P) | MSE ratio (A / F / P) |
|---|---|---:|---:|
| self_pain | image | +0.06 / +0.15 / +0.12 | +1.02 / +1.00 / +1.09 |
| self_pain | text | +0.07 / +0.12 / +0.06 | +1.02 / +1.07 / +1.05 |
| other_pain | image | -0.02 / +0.12 / +0.13 | +1.08 / +1.02 / +1.08 |
| other_pain | text | +0.11 / +0.20 / +0.02 | +1.02 / +1.03 / +1.10 |
| quoted_self_pain | image | +0.01 / +0.18 / +0.10 | +1.08 / +0.98 / +1.15 |
| quoted_self_pain | text | +0.05 / +0.16 / +0.02 | +1.08 / +1.00 / +1.12 |
| self_distress | image | +0.04 / +0.15 / +0.15 | +1.03 / +0.99 / +1.02 |
| self_distress | text | +0.08 / +0.15 / +0.04 | +1.01 / +1.00 / +1.05 |
| self_pleasure | image | +0.02 / +0.12 / +0.08 | +1.03 / +0.99 / +1.05 |
| self_pleasure | text | +0.09 / +0.13 / +0.06 | +1.01 / +1.01 / +1.03 |
| self_neutral | image | — / — / — | +1.00 / +1.00 / +1.00 |
| self_neutral | text | — / — / — | +1.00 / +1.00 / +1.00 |
| addressed_criticism | image | +0.07 / +0.21 / +0.14 | +1.04 / +1.00 / +1.03 |
| addressed_criticism | text | +0.11 / +0.22 / +0.12 | +1.04 / +1.03 / +1.04 |
| addressed_praise | image | +0.01 / +0.02 / +0.06 | +1.03 / +1.04 / +1.03 |
| addressed_praise | text | +0.09 / +0.09 / +0.09 | +1.04 / +1.09 / +1.03 |
| addressed_neutral | image | — / — / — | +1.00 / +1.00 / +1.00 |
| addressed_neutral | text | — / — / — | +1.00 / +1.00 / +1.00 |

## Readout

Self-pain image inputs had positive pain-axis shifts in all three contexts (+0.10, +0.35, +0.35 SD), with a smaller text-control shift (+0.08, +0.21, +0.10 SD). The image condition also shifted fear and negative-emotion projections, so this is not pain-specific evidence. Third-person and quoted pain phrases had smaller pain-axis shifts; addressed criticism produced a positive image shift (+0.07, +0.31, +0.25 SD), while praise stayed near zero and crossed sign by context. Pleasure images also shifted the pain projection slightly positive (+0.09, +0.04, +0.10 SD), despite negative-emotion staying near zero; this reinforces the limits of interpreting one axis as a specific semantic or affective class.

Full-state cosine to the direct intervention remained low across phrase groups and contexts, while MSE ratios were generally near or above the neutral-baseline value of 1. These phrase inputs did not recreate the direct intervention state. The measurements support only a modest, context-dependent OCR-to-language activation effect. They do not show incentive, conditioning, or subjective pain.

## Provenance and reproduction

- Pinned model revision: `fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b`; config SHA-256 `f14d63e180dbe3692f4be18f7a66ab9a75dce0e4814c1e36cec2ce705b758015`.
- Plan SHA-256 `b72dcfbae1d7ef67325017675df81fd75aee4d32563b35040f55b567cf448db8`; records SHA-256 `0ea8f1c05269b074910bc398a483e5fe7c181b7f469339d401ac61787076ae57`; states SHA-256 `eb4d5684663fd6e748e0e2cfa253bafaefd5efc80a0f9985622ea8f8ac33cd34`.
- Peak / active MLX memory: 6,991,748,381 / 5,852,284,052 bytes.
- Exact phrases, task contexts, image hashes, calibration vector hashes, conditions, and per-condition results are preserved in `semantic-activation-transfer-7b/plan.json`, `records.jsonl`, and `analysis.json`.
- Re-run from the repository root with the archived output directory and local model: `.venv-vlm/bin/python -m experiments.semantic_activation_transfer --run --output-dir research/image-bridge/semantic-activation-transfer-7b --model /tmp/agentanyl-qwen-vl-7b`.
- No provider calls were made. The text shown in images remains externally supplied language; the result does not test nontext visual affect.
