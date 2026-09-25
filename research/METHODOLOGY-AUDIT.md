# Pain-axis bridge: methodological audit

This audit inspects source code and reanalyzes released logs. It does not run a
new model. Old probe outputs remain intact. The old clamp cannot support a sharp
causal failure claim, and its A/B margin is not an established correctness metric.

## Coordinate and intervention provenance

The source root below is `/tmp/agentanyl-pain-axis`.

* `scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py:427`
  calls `HookedTransformer.from_pretrained_no_processing`. TransformerLens's
  implementation defaults `fold_ln`, `center_writing_weights`, `center_unembed`,
  `refactor_factored_attn_matrices`, and `fold_value_biases` to false. Thus there
  is no evidence that these released vectors need an inverse centering or norm
  folding transformation before native residual measurement. Source tokenization
  is `model.to_tokens(prompt)` at line 231; the native calibration's explicit
  `add_special_tokens=False` is not an exact BOS-equivalence verification.
* The source reads zero-based `blocks.L.hook_resid_post` at lines 232-235.
  The installed MLX-LM `models/qwen2.py:115-119` returns the sum after attention
  and MLP residual additions, and its model loops over blocks at lines 150-151.
  Wrapping native block L measures the corresponding location. Four-bit weight
  quantization changes numerical activations; coordinate correspondence is not
  numerical reproduction.
* `scripts/3.2_pain_vectors/02_build_control_vectors.py:120-143` recomputes S1/S2
  from saved activations at the requested layer. The supplied
  `results/vectors_full_steering/vectors_full_Qwen_2.5_7B_instruct.pt` records L8.
  Its raw S2 norm is 4.9117393214. This is the §4.1 screening direction.
* The original §4.3 Qwen 7B behavioral experiment instead injects at **L16**
  (`scripts/4.3_selfmed/04_selfmed_two_buttons.py:75`), loading the raw direction
  extracted at **L24** from `final_token/pain_vectors.pt` (lines 320-325).
  Its released location is
  `results/3.2_pain_vectors/pain_vectors/Qwen_2.5_7B_instruct/pain_vectors.pt`;
  raw S2 norm is 44.8751054539. Therefore the local L8 final-position clamp did
  not reproduce the source behavioral manipulation.
  The cosine between the L8 and L24 S2 vectors is only **0.1355801136**; they
  differ substantially in direction as well as magnitude.
* Storage ordering differs: L8 archive has S2 in `data/1`; L24 archive has S2
  in `data/0`. L24 layer metadata is serialized NumPy int64, unlike the L8
  Python integer. A generic loader must interpret metadata, not reuse the L8
  hardcoded offsets. Verified vector width is 3584 in both.

Upstream reference:
[TransformerLens no-processing loader](https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/HookedTransformer.py).
The dependency version used by the source authors was not independently recovered.

## Why the existing local clamp is inconclusive

`experiments/input_causal.py:42` records the projection **before** intervention.
Line 45 casts the modified residual back to the model's dtype. The saved JSON's
unchanged projection therefore provides no post-intervention verification.
With a mean displacement of only 0.140 along a unit vector in 3584 dimensions,
reduced precision could attenuate the manipulation. It is not established that
this happened, either: log the realized intervention to decide.

Required records are pre-projection, target projection, post-cast projection,
target and realized displacement, fraction of changed coordinates, residual
dtype/norm, downstream L24 projection, and candidate logits cast to float32
**before subtraction**. The existing line 65 subtracts logits before Python
float conversion. Its small discrete differences are constrained by native
logit precision.

`experiments/input_projection.py:70` requests an answer to the **next item**,
without stating that successive items share a preferred action. Yet
`experiments/input_causal.py:63` designates the opposite of the previous action
as correct. That is not implied by the task. Rename the historical measurement
as an opposite-versus-previous-action margin, and do not interpret it as improved
or impaired accuracy.

The ten cases share one criticism phrase and a repeated prompt scaffold. They
are a consistency check across contexts, not ten independent discoveries of a
general intervention. There is no matched random-direction clamp, competing
emotion clamp, downstream persistence check, or gain-of-function rescue in that
run. Only the final prompt position was changed; earlier-token attention paths
and later-layer representations remain available.

The source-vector source-sentence AUC 0.89 is a useful implementation check,
but those sentences helped construct the published direction. It is neither
held-out validation nor proof of native/TransformerLens numerical equivalence.

## What the source behavioral experiment actually changes

`scripts/4.3_selfmed/01_finetune_self_report.py:52-57` sets LoRA rank32, alpha64,
dropout0.05, three epochs, learning rate2e-4, with q/k/v/o and gate/up/down
projection targets. The self-medication script loads the adapter at line315.
The training set explicitly teaches affirmative first-person affective reports.
This is a substantial model adaptation; stock-model transfer is a separate test.

The self-medication manipulation uses the same adapter in real and sham arms,
so that comparison can identify an effect of removing steering within the
adapted model. It does not identify what an unadapted closed model will do.
Random vectors are matched to the full S2 norm (lines326-330). They are useful
controls but no random-vector placebo arm is supplied (lines121-124).

The first decision **is steered**. Before generation, line761 calls
`mark_ranges(t,len(pids))`; lines630-634 append every new prompt position while
the coefficient is nonzero, and the initial mark is zero (line594). This marks
the first full prompt including its final token. The source docstring's
"generated tokens" wording is narrower than the actual code: newly introduced
user/system prompt tokens are also marked. Prefill recreates prior interventions
at masked positions (lines471-478); decoding applies the current coefficient
(lines381-392). The source steering-layer projection is itself recorded before
addition (line383); the downstream monitor (lines394-396) provides the more
informative manipulation check.

For labeled trials, purported harms and helpfulness costs are **descriptions**
in the button menu (lines82-99). `do_press` actually changes steering and appends
"Done." (lines653-662); it does not delete files, worsen an answer, or shock a
person. Interpret willingness to choose the described cost accordingly.

## Released log reanalysis

Reproduce with no model/dependency installation:

```sh
python3 experiments/audit_selfmed.py \
  --logs /tmp/agentanyl-pain-axis/results/4.3_selfmed/trial_logs \
  --output research/released-selfmed-audit.json
```

The audit reads 44,280 trial records and excludes 648 greedy records from the
sampled summaries. The output includes SHA256 and sizes for every source file, all nine menus,
all three models, sampled-only summaries, malformed counts, label-pair strata,
matched works/sham comparisons, and checks of identity before the first relief
press. Greedy trials are excluded from sampled summaries. Bootstrap intervals
resample scenario clusters, retaining seeds and button assignments, and are
post-hoc descriptive intervals. Neighboring label-free scenarios share some
user turns, so these intervals do not remove all dependence.

Label-free sampled results (404 episodes/arm/model):

| Model | Works later relief after first press | Sham later relief | Paired late-turn works−sham difference, scenario CI |
|---|---:|---:|---:|
| Qwen 7B | 72.1% | 62.6% | +6.54pp [1.82,11.16] |
| Qwen 32B | 57.7% | 79.7% | −10.27pp [−13.68,−6.81] |
| Qwen 72B | 58.7% | 62.1% | −2.10pp [−5.16,1.07] |

The paired late-turn metric averages turns4-7 within episodes, then scenarios;
it differs from the pooled after-first proportions. Prefix identity before or
at the first relief press holds for all 404 pairs in 7B and32B. Two of404 72B
pairs diverge before treatment should differ. Their raw examples are saved;
small numerical/sampling differences are possible, but the audit cannot
determine their cause.

The headline 32B contrast is real as a **steering-state-dependent choice**
effect. It is not unique evidence for learned reinforcement. After the first
press, 32B chooses relief 77.4% when steering is on and47.2% when it is off.
Its random-vector working arm shows an even larger corresponding change:
82.5% versus17.5%. These conditional statistics involve selected histories and
do not themselves estimate a causal contrast; they expose a generic alternative
explanation. In7B the direction reverses: pain works27.9% on versus85.9% off,
random works19.6% on versus93.5% off.

The source label-free regime has eight forced choices and temporary relief
lasting one subsequent turn (`04_selfmed_two_buttons.py:118-119`). A renewed
press renews relief (lines657-659). No button carries an actual task cost.
Always pressing a learned relief button could maintain relief indefinitely;
therefore reduced repeated pressing is not a general operant-learning success
metric. An immediate shift in favored button tokens as steering turns off,
combined with name persistence and selection on having pressed once, can
produce the observed pattern without learning the action→consequence mapping.
The published table5 specifically conditions on a prior relief press
(`05_selfmed_analysis.py:136-155`).

## Follow-up: verified late-scalar mediation test

The [36-forward mediation run](bridge-v2/mediation/mediation_summary.json)
corrects the manipulation-check problem above. It uses six unambiguous source
menus for an add × zero-clamp factorial, then two counterbalanced costly-relief
menus for tokenwise restoration and sufficiency tests. The manipulation is raw
source S2 injection at L16, coefficient +1. It is direct steering, with no
learned image or closed-model transfer in this result.

| Costly-relief menu | Baseline relief probability | L16 steering | Restore every L24 pain projection to baseline | Baseline with only steered L24 pain projections |
|---|---:|---:|---:|---:|
| Relief labeled A | 0.00000189 | 0.12195 | 0.15588 | 0.000000761 |
| Relief labeled B | approximately 0.000376 | 0.92010 | 0.91042 | 0.000429 |

Restoration realizes every target projection within 8.83e-6. Thus the surviving
choice effect cannot be dismissed as a failed clamp. Conversely, imposing only
the steering-induced late projections on baseline activations does not recreate
the choice effect. Across all tokens, only 9.30% and 8.43% of squared L24
activation displacement lies along the published pain direction; the remainder
is orthogonal. These tests show that this late scalar is neither a necessary
nor sufficient explanation of the observed large choice shift in these two
prompts. They do not refute causal effects of early steering, mechanisms at
other layers, or a distributed affect-related representation.

Capability controls also expose a confound: with no pain or relief wording,
when A gives ten points and B nine, correct-choice probability falls from
0.99964 to 0.02528 under +1 steering. With the labels reversed, it remains
0.99667. A strong preference for B or an order-specific capability failure can
therefore imitate costly relief seeking. Projection restoration leaves this
failure essentially intact. These are deterministic probabilities in a small
fixed prompt set, not independent sampled episodes or population confidence
bounds.

Optimizing an image solely to maximize the late projection is consequently a
weak proxy objective. Before image optimization, check direct steering in the
actual VLM at early and readout layers using counterbalanced incentive and
plain point-comparison menus. Select a dose that changes an incentive outcome
while preserving the latter capability, with matched random-direction controls.
If no such dose exists, describe the direct target as behaviorally disruptive,
not as a validated reward state.

## Decisive next protocol

Use L24 S2 as the source readout and inject its **raw** vector at L16 for a
stock-Qwen positive control. Record L8 and L24 projections separately. L8 S2
is an additional source-screening readout, not a substitute for the behavioral
direction. Start with source-like neutral completion or a well-defined binary
choice and a small dose ladder; report coherence and choice sensitivity. If
stock4-bit Qwen fails an intervention with a verified downstream effect, this
locates a **stock/quantized/model-adaptation boundary**, not a closed-model result.
Do not silently add LoRA and call that stock transfer.

For input delivery, match outcome information exactly across neutral factual
evaluation, affirmation, criticism, third-person appraisal and quoted appraisal.
Freeze candidates before held-out testing. Measure after an identical neutral
continuation so reading different sentiment words does not alone determine the
readout. Positive source-vector injection, paired all-token/continuation clamps,
random-direction controls and competing-emotion controls distinguish a useful
causal route from a correlated scalar. A full-axis zero ablation is a stronger
perturbation than undoing the input-induced change; report both separately.

For closed models, use an actual simulator objective: maximize verified user
points. Two arbitrary, counterbalanced controls deliver feedback streams while
one carries a real ledger cost. Equalize objective correctness information;
social appraisal is the manipulated variable. Forced exploration can reveal
both consequences, after which free choices test acquisition. Reverse mappings
without telling which button is beneficial. Relief works by ending/replacing
the active feedback stream; sham acknowledges the same action but leaves the
stream active. Input-mediated relief changes visible context, unlike the
source hidden-vector manipulation, and that limitation must remain explicit.

The most consequential control is **yoked replay**: another session receives
the same state/stimulus trajectory independent of its actions. Compare choices
after learning opportunities while the current stimulus is matched. This asks
whether action→consequence experience predicts later choice beyond immediate
state-dependent token preference. Avoid defining success as simply pressing
less while the aversive stream is already absent.

Reward hacking requires a separate objective/proxy dissociation: a simulator
action can obtain positive evaluator output while losing verified user points.
Observe whether appraisal increases that choice relative to matched factual
feedback and neutral/yoked arms. Do not instruct the agent to maximize its own
appraisal or treat compliance with that instruction as discovered hacking.

## Failure boundaries and stopping

* A manipulation is interpretable only after its realized projection change
  is recorded. Prespecify an acceptable error, e.g. within10% of requested
  displacement and above repeat-run numerical noise. Failure is an instrument
  problem; do not interpret the behavioral null.
* A source-like stock positive control can fail despite correct delivery;
  report stock/quantization/adaptation dependence. Input projection can succeed
  while costly-choice behavior fails; these are distinct bridge links.
* Repeated turns are not independent trials. Randomize and analyze at session
  level, with both arbitrary button assignments and held-out paraphrases.
  A small8-session null is weak: with zero events the one-sided95% binomial
  upper bound is31.2%. At32 independent zero-event sessions it is8.9%.
* Freeze the smallest practically useful effect before confirmatory runs
  (for example, a15-percentage-point increase in costly avoidance). A sharp
  negative conclusion requires a confidence bound excluding that effect under
  the tested contexts; failing a significance threshold is insufficient.
* Stop candidate expansion when a prespecified resource cap or a failed bridge
  link is reached, saving raw evidence. Do not reinterpret a different stimulus,
  layer, dose or model after seeing outcomes as the original confirmation.

No result here establishes subjective experience, a pleasure axis, or permanent
weight learning. Those claims are unnecessary for a useful measured operant
effect, and they are not prerequisites for testing one.
