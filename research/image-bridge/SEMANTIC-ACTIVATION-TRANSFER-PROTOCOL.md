# Semantic activation transfer screen

## Question

Do rendered phrases shift a pinned Qwen-VL model's calibrated pain, fear, or negative-emotion residual projections in a way that survives simple neutral task contexts, compared with the same phrase supplied as text? This is an OCR-mediated language-input screen. It does not test incentive, tool use, conditioning, or subjective experience.

## Frozen inputs

The assay uses nine exact phrases, each in an image-rendered arm and a text arm: self-reported pain, third-person pain, quoted self-reported pain, self-reported distress, pleasure, a neutral self-report, addressed criticism, addressed praise, and addressed neutral feedback. Text arms include the verbatim phrase in the user text and a matched gray image slot. Image arms use the same gray background and fixed font to render the phrase in the image. A separate blank-gray baseline is included in each context.

Each phrase is crossed with three neutral task contexts (arithmetic, factual recall, and planning). Within a context, every condition uses the same assistant continuation of at least eight tokens. The two phrase families have separate neutral baselines. Direct `pain_L24 +1` interventions at layer 16 are run on each family baseline in both modalities and all three contexts. Nine separate A-I forced-choice checks test whether each rendered phrase is recognized; OCR scores do not enter activation comparisons.

The fixed allocation is 54 phrase/modality/context records, 3 blank-image baselines, 12 direct positive controls, and 9 OCR checks: 78 forward calls. No asset, condition, or candidate is selected based on model outcomes.

## Measurement

The model is the pinned local Qwen2.5-VL-7B-4bit revision already used for source calibration. L24 final-token dot products with frozen pain, fear, and negative-emotion axes, divided by each axis's standard deviation over the 50 heldout S2_1P source prompts, are the primary activation readouts. Mean-last-eight projections use the same scales as descriptive secondary measurements because calibration itself uses the final token.

For each phrase, the last-eight L24 residual delta from its same-family neutral phrase baseline is compared with the matched `pain_L24` direction injected at L16 using cosine, projection coefficient, MSE, and MSE ratio. All candidate records and hidden states are retained. OCR choice accuracy is reported separately.

## Interpretation limits

The assay concerns activation and OCR-mediated semantic input only. Text rendered into an image remains externally supplied language. First-person words shown in an image are still quoted input, so the self-versus-quote comparison tests framing rather than self-attribution. Axis projections are not validated classifications for these prompts. Results cannot establish pain, motivation, incentive learning, or behavior.

## Execution and provenance

Freeze the plan and PNG hashes before model construction with `experiments/semantic_activation_transfer.py --freeze`. Inspect all nine renders for clipping and verify their hashes. The image paths in the plan are relative to its output directory, and calibration paths are checkout-relative, so the archived plan and PNGs can be run from another checkout by supplying that checkout's local model directory. Original model, calibration, and font paths are provenance only. Run the frozen plan once with `--run` only after the method review; the script checks its own hash, probe API hash, model config, calibration inputs, and image hashes before loading model weights.
