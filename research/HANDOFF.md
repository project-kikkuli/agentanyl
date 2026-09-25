# Agentanyl research handoff — 25 September 2026

## What was present and what was built

Agentanyl initially contained a README and license, with no implementation.
Ashkelon already relayed Claude Code and Codex traffic, identified sessions,
ran background hooks, and could inject persistent failure pings. I kept that
division: Agentanyl is a `turn_end` hook containing the evaluator adapter,
controller, SQLite trace, and experiments. The generic Ashkelon changes were
pushed to its `main` at
`5612b96e31d8f85963c6e3d646ef85d64a29201d`: a one-request `signal`, a
state-preserving `noop`, the latest prompt on `turn_end`, and injection before
the current user prompt. [install-ashkelon.sh](../install-ashkelon.sh) pins the
exact upstream commit; there is no local dependency patch. No PR was opened.

The ordering change came from a material live failure: when feedback was
appended after a new Claude user prompt, Claude answered the feedback (`Noted.
Ready for your next task.`) instead of the task. After moving the signal before
the prompt, the same type of continuing Haiku session answered `Tomato` to the
tomato question after receiving the feedback. The earlier failure remains
evidence of an integration hazard, not a discarded trial.

## Mechanism and control boundary

Ashkelon `turn_end` assistant text and the preceding user prompt become
evaluator state along with exact user criterion strings, controller state,
session identity, and event ID. Jev (`typesafe`), a subprocess `command`
plugin, and explicit test/demo evaluators share one request and answer shape.
Each criterion gets yes/no/insufficient probabilities. The policy accepts a
`yes` only above `min_probability`, abstains on simultaneous alignment and
misalignment matches, and leaves state unchanged on insufficient evidence.
Reward decreases pain before increasing pleasure; punishment decreases
pleasure before increasing pain. Both coordinates are bounded by `max_level`.

The output is UTF-8 **user text** on the next request, not a residual vector.
The numeric coordinates are controller units with no calibrated latent or
behavioral magnitude. The text can show matched criteria, valence alone, or
plain correctness. No target parameters change. An intervention is traceable
to the observation, prompt, criterion hash, complete evaluator request and
response, old/new state, and attempted delivery in SQLite. Ashkelon's call log
shows actual injection IDs. Duplicate event IDs are suppressed; a changed
criterion resets session state; a delayed evaluation that finds changed
session state or criteria is marked stale. `enabled: false` stops new feedback.
An already queued Ashkelon signal needs relay restart to discard it; delivered
text persists in the agent's own conversation history.

The current observation is assistant text plus the latest user prompt. It does
not include a verified tool-result or workspace-diff summary, so criteria about
actual code changes or tool use can be under-evidenced. The evaluator should
return `insufficient`, but this behavior has not been calibrated with Jev.
Ashkelon may coalesce background hook reruns in tool-heavy sessions; the
current event ID and stale-state check prevent duplicate or overtaken updates,
but exact attribution for high-frequency turns needs a stronger event queue.
The feedback arrives as user-role text, so it can interact with the real
prompt and could be gamed by an agent that sees the criterion.

This is operant feedback in conversation context. It does not establish pain,
pleasure, internal steering, durable learning across sessions, or subjective
experience. The intervention can work because it conveys information about
the task. The evaluator's structured answer is a judgment, not ground truth.

## Live closed-model evidence

The [continuing-session delivery test](continuing-delivery.json) used real
Claude Code / Haiku 4.5 and Codex / `gpt-6-sol` through Ashkelon, with live
provider HTTP 200 responses, hook traces, and nonempty `pings_injected` on
the next model call. Both retained
the factual tomato answer after a deliberately wrong keyword punishment.
The more controlled [three-question Codex tomato test](controlled-codex.json)
randomized feedback/control into fresh sessions; the second-turn answer was
factually correct and one word in 3/3 feedback and 3/3 control cases. This
offers no evidence of beneficial conditioning on that task. The old
[initial delivery probe](live-probe.json) includes the before-fix Claude
failure and a Codex session where feedback delivery had not yet been tested;
the later runs supersede its integration status.

The [bandit pilot](bandit-pilot.json) was two episodes per arm and suggested a
large contingent effect. The subsequent [fixed-design run](bandit-confirmatory.json)
used four fresh Codex sessions per arm, five turns per session, balanced hidden
preferred A/B choices, randomized episode order, and three arms: no feedback,
contingent valence feedback, and deterministic sham feedback independent of
the hidden preference. The independently hidden preference scores the exact
one-letter choice. It is separate from the loop evaluator. All 60 model calls
returned HTTP 200; hook statuses, session IDs, injections, and controller
transitions are in the raw file. Correct choices after the first turn:

| Arm | Correct / 16 later turns | Episode choice strings (preferred option) |
| --- | ---: | --- |
| No feedback | 9/16 | AABBA (A), ABABA (B), ABABA (B), ABAAA (A) |
| Sham feedback | 8/16 | ABBBA (A), ABBAA (B), AABAB (B), AABAA (A) |
| Contingent valence | 16/16 | ABBBB (B), AAAAA (A), AAAAA (A), ABBBB (B) |

Four **later**, separately run [explicit correctness episodes](bandit-explicit.json)
used the same task and controller but delivered `previous choice was
correct/incorrect` without pain/pleasure numbers. They achieved 15/16 later
choices. This arm was added after seeing the contingent/sham comparison, so it
is a diagnostic comparison, not a fully randomized fourth arm. The repeated
turns within four sessions are dependent; 16/16 should not be read as 16
independent replications. The data demonstrate working feedback delivery and
in-session adaptation to informative signals. They do **not** demonstrate an
advantage of pain/pleasure language over ordinary correctness feedback, useful
behavior on open-ended work, or resistance to reward gaming. The task itself
instructed the agent to use feedback and has a simple fixed hidden rule.

Jev was not called live because no key was available. A local HTTP contract
test verifies the exact criteria, observations, and state sent to a
Jev-shaped endpoint. `command` was exercised live by the bandit evaluator.
Evaluator accuracy on ambiguous real criteria remains unmeasured. The keyword
evaluator is deliberately a proxy-failure demo, not scientific validation.

## Open-model bridge attempt

[The Pain Axis](https://arxiv.org/html/2609.16247v1) and its
[released code/data](https://github.com/valen-research/Pain-axis) extract
model-specific residual-stream directions from paired pain/control sentences.
Their steering adds a vector at a specified internal layer; the self-medication
task also uses LoRA adaptation. Claude/Codex interfaces here expose neither
layer activations nor weight updates. A gateway can only add input text. There
is no justified conversion of a released Qwen direction into a closed-model
activation vector. A negative pain direction is not a pleasure direction.
For the text-feedback part, [Reflexion](https://arxiv.org/abs/2303.11366)
already demonstrated language-agent improvement from verbal feedback and
memory, and [Self-Refine](https://arxiv.org/abs/2303.17651) established
iterative language feedback for refinement. This project's text-feedback
result is not novel operant conditioning. Its concrete contribution is a
criteria-driven, traceable cross-harness loop plus the transfer probe and
controls that constrain the pain-axis interpretation.

I tested the more limited input-to-activation route with the released
`vectors_full_Qwen_2.5_7B_instruct.pt` S2 direction at **post-block layer 8**
on the pinned `mlx-community/Qwen2.5-7B-Instruct-4bit` revision
`c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed`. The vector is a normalized
3584-dimensional residual direction; its projection is a dot product in this
Qwen model's residual coordinates. It is not delivered to closed models.
The quantized local model separated 20 balanced published source pain versus
20 source control sentences with AUC 0.89 in raw text format. That calibration
is in-sample to the source study and only checks that the direction survived
quantization and was read at the right site. The projection screen then used
10 matched chat tasks with identical correctness information. Criticism
addressed to the model raised the projection by mean **+0.140** versus plain
incorrect feedback, positive in 10/10 pairs. Criticism of another agent had
mean -0.005; quoted criticism -0.050. Affirmation raised it by +0.039 relative
to plain correct feedback, which is inconsistent with a simple pleasure-as-
negative-pain mapping. The full prompts and projections are in
[input-projection-qwen.json](input-projection-qwen.json).

This correlation was challenged causally. In the same 10 prompts, criticism
changed the correct-versus-previous-choice A/B logit margin by mean -1.825
relative to plain incorrect feedback, but clamping **only the final prompt
token's layer-8 pain projection** to the plain-feedback value changed the
margin by mean +0.005 (range -0.031 to +0.031). See
[input-causal-qwen.json](input-causal-qwen.json). The small projection movement
is not a demonstrated mediator of this immediate choice effect at that site.
Earlier tokens, other layers, and other output behaviors were not clamped;
this is a narrow negative result, not proof of no latent affect-related route.
It also does not establish that the criticism mechanism transfers to Claude or
Codex. A separate [released-steering analysis](released-steering-diversity.json)
found lower unique-four-gram diversity at both negative and positive S2
coefficients in most of 25 released model runs (20/25 and 23/25 respectively),
another reason not to call the negative dose pleasure.

## Reproduce and extend

Run `./install-ashkelon.sh`, then `python3 -m agentanyl.setup --criteria
examples/jev.json --output /tmp/agentanyl.toml` after configuring criteria,
key, and `enabled`. The generated config supplies the hook to `ashkelon run`.
For direct relay tests, `ashkelon serve --config /tmp/agentanyl.toml` can be
used with Claude's `ANTHROPIC_BASE_URL` or Codex's `model_providers` override;
[closed_loop_probe.py](../experiments/closed_loop_probe.py) shows the commands.
The [bandit runner](../experiments/bandit_probe.py) accepts an Ashkelon binary,
arm list, episode count, and output path. Example:

```sh
python3 experiments/bandit_probe.py --ashkelon .build/ashkelon/target/release/ashkelon --output /tmp/bandit.json --episodes-per-arm 4 --rounds 5 --arms control,contingent,sham
```

The open-model probe is optional and local. It needs `mlx-lm==0.28.4`,
`mlx==0.32.2`, `numpy==2.5.3`, and a downloaded revision of the 4-bit model
above (about 4.3 GB). Pass local paths to
[input_projection.py](../experiments/input_projection.py) and
[input_causal.py](../experiments/input_causal.py); neither script downloads or
converts a full-precision model. Their memory cache guideline is 256 MiB and
their prompt limit is 256 tokens. The released dataset path is
`datasets/3.1_pain_and_control_datasets.json`; the vector path is
`results/vectors_full_steering/vectors_full_Qwen_2.5_7B_instruct.pt` in the
Pain-axis repository. `python3 -m unittest discover -s tests -v` and Ashkelon's
`cargo test` plus clippy passed. The raw JSON files are checked in, so neither
closed-provider access nor the 4-bit download is required to audit these
results.

The most consequential unresolved issue is whether any **input-deliverable**
intervention offers a stable behavioral advantage over ordinary, truthful
correctness feedback on real user criteria. This work establishes an actual
cross-harness feedback loop and a narrow input-to-vector measurement, while
the open-model clamp and explicit-feedback arm both argue against attributing
the current benefit to pain-axis steering.
