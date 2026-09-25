# Closest prior work and the claim this experiment could support

Optimizing an image to alter a model's internal representation, or transferring
an adversarial image to a commercial model, is established prior work. The
potential contribution here is narrower: **input-mediated, action-contingent
control of a validated pain-related direction, with measurable costly avoidance
and a tested open-to-closed behavioral bridge**. This remains a hypothesis.

## Relevant precedents

* **Adversarial Illusions in Multi-Modal Embeddings** aligns perturbed images or
  sounds with adversary-selected representations in another modality. The
  authors test downstream generation, classification and retrieval without
  targeting each downstream task, and report a black-box attack on Amazon
  Titan embeddings. This is a close precedent for activation-targeted media
  optimization. It does not establish an affective incentive or operant
  conditioning. [Paper](https://arxiv.org/abs/2308.11804)

* **CrossMPI: A Cross-Modal Prompt Injection Attack against Large
  Vision-Language Models with Image-Only Perturbation** explicitly optimizes
  multimodal hidden states, with layer selection and constrained image
  perturbations. Therefore even targeting internal decoder representations
  through pixels is not by itself a novel contribution. Its stated endpoint
  is cross-modal prompt injection. [Paper](https://arxiv.org/abs/2605.16090)

* **Visual Adversarial Examples Jailbreak Aligned Large Language Models**
  optimizes a visual input using a limited harmful-output corpus and reports
  broader jailbreak effects across instructions. This motivates separating a
  reusable input's activation effect from ordinary changes in generation or
  refusal behavior. [Paper](https://arxiv.org/abs/2306.13213),
  [authors' code](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models)

* **AnyAttack** reports self-supervised targeted image attacks and transfer to
  commercial systems including Gemini, Claude Sonnet, Copilot and GPT. Thus
  closed-model behavioral transfer of images alone would not distinguish this
  project. Its large-scale training also differs from our bounded local
  optimization. [Paper](https://arxiv.org/abs/2410.05346)

* **Understanding Adversarial Transfer: Why Representation-Space Attacks Fail
  Where Data-Space Attacks Succeed** presents evidence that transfer depends
  on whether the attack operates in shared input space or model-specific
  representation space, with geometric alignment relevant to the latter.
  Consequently, sending a valid PNG is necessary for practical transfer but
  does not guarantee that optimizing one model's latent objective will affect
  another model similarly. [Paper](https://arxiv.org/abs/2510.01494)

* **EmotionPrompt** appends short, psychology-derived emotional stimuli to
  ordinary task prompts and evaluates them on instruction-induction and
  BIG-Bench tasks, reporting performance changes across several language
  models. This makes general emotional or social wording effects established
  prior work: criticism-text effects alone are not novel. EmotionPrompt does
  not test repeated action-contingent outcomes or costly avoidance under a
  matched-cue control. [Paper](https://arxiv.org/html/2307.11760)

* **Representation Engineering (RepE)** studies reading and controlling
  high-level concepts in model activations, including emotion, utility, and
  honesty. Linear concept extraction and activation steering are established
  methods, so a pain-related axis or a direct steering effect is not novel by
  itself. The distinct open question is whether a validated pain-related
  direction functions as an action value or reward signal under causal,
  capability-preserving controls. [Paper](https://arxiv.org/html/2310.01405)

## Runtime feedback is already an established approach

* **Reflexion: Language Agents with Verbal Reinforcement Learning** converts task
  feedback into verbal reflections retained in episodic memory, improving later
  decisions without updating model weights. Agentanyl's evaluator, feedback
  message and observation ledger alone would not establish a new learning
  mechanism. [Paper](https://arxiv.org/abs/2303.11366)

* **Self-Refine: Iterative Refinement with Self-Feedback** uses the same model to
  generate, critique and revise outputs without additional training. This is a
  relevant ordinary-feedback alternative; the experiments here need independently
  scored outcomes rather than improvements judged only by the feedback model.
  [Paper](https://arxiv.org/abs/2303.17651)

* **In-context Reinforcement Learning with Algorithm Distillation** trains a
  causal transformer on learning histories, after which its policy can improve
  through context without parameter updates. That result establishes a possible
  form of adaptation, but its dedicated training does not establish that a
  general closed-weight assistant treats an arbitrary affective cue as a reward.
  [Paper](https://arxiv.org/abs/2210.14215)

These precedents separate changes supported by conversation history from changes
to model parameters. The paired contingency test adds a narrower question: does
controlling an auxiliary cue change choices when the current cue is matched,
with the user's task score computed independently? Success would still need to
be distinguished from learning the meaning of ordinary verbal feedback.

## Pain-axis requirements and scope

**The Pain Axis** supplies denoised pain/control directions, direct residual
steering, and working-versus-sham relief experiments. Its behavioral models
are LoRA-adapted Qwen models. Local source inspection finds Qwen7B's L24-derived
S2 vector injected at L16; the L8 recomputed screening vector is a different
direction. The adapter uses rank32, alpha64 and three training epochs across
attention and MLP projections. Stock multimodal inference is therefore a new
transfer setting. [Paper](https://arxiv.org/html/2609.16247v1),
[self-medication source](https://github.com/valen-research/Pain-axis/blob/main/scripts/4.3_selfmed/04_selfmed_two_buttons.py),
[adaptation source](https://github.com/valen-research/Pain-axis/blob/main/scripts/4.3_selfmed/01_finetune_self_report.py)

Our local reanalysis finds that the source's unlabeled choices depend on
steering state, including under random steering. Its temporary, renewable,
cost-free relief does not by itself isolate learned reinforcement from
state-dependent action preferences. See [audit](METHODOLOGY-AUDIT.md).

## Evidence needed for a distinctive result

A convincing result could involve more than one input route: establish a
held-out, input-mediated change in a pain-related readout, then show selective
causal attenuation with capability preserved and action-contingent,
mapping-sensitive costly avoidance beyond matched factual-feedback and yoked
controls. This work's semantic image screen supports an OCR-mediated language
route; its optimized text-free pixel images failed calibrated state transfer.
Direct activation steering is a local-model manipulation and does not validate
either input route as a reward signal. Closed-model transfer would establish a
portable behavioral effect, while its internal mechanism would remain
unobserved. Image-induced sentiment, distress wording, task degradation, or a
large projection alone cannot establish that chain. Reduced pain projection is
not established pleasure. This focused literature check does not establish
priority over every affect-steering study.
