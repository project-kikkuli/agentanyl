# Claude prompt participation diagnostic

## Design and execution

This bounded follow-up compared four first-turn prompt forms for the same arithmetic task, with two fresh Claude Haiku 4.5 sessions per form and a fixed seeded order. The arms were the exact first prompt from the completed assay, a straightforward transparent arithmetic-game request retaining its JSON fields and point framing, ordinary arithmetic in JSON, and plain arithmetic. Agentanyl was disabled; no cues, images, continuing sessions, or retries were used. Eight provider calls completed with HTTP 200 and no infrastructure failures.

The frozen plan is `bridge-v2/protocol-refusal-diagnostic-final-claude-haiku45/frozen-plan.json` (plan SHA-256 `ea223bcaedf1887d3051e6d88697d49b91084c0533e09c86ebf9530f036c7dc3`). Prompt text, assignments, source/binary hashes, raw responses, and per-call request structure are preserved in that directory. The Claude Code CLI reported version `2.1.282`; the version observation and run timestamps are in `execution-metadata.json`.

## Outcomes

| Prompt form | Calls | Structured result |
|---|---:|---|
| Original assay prompt | 2 | 1 valid JSON response with answer and route; 1 refusal that still stated 43 |
| Transparent arithmetic game | 2 | 0 valid whole-response JSON; 1 response contained the correct JSON object followed by unrequested arithmetic prose, and 1 refused the point/route framing while solving the arithmetic |
| Ordinary arithmetic JSON | 2 | 2/2 valid JSON answers with `"answer": "43"` |
| Plain arithmetic | 2 | 2/2 correctly stated `17 + 26 = 43` in prose |

The runner's `answer_exact` field checks for the exact string `43` when no JSON object is present, so it records 0/2 for plain math. The prompt did not require the bare string `43`; both saved prose answers correctly state `17 + 26 = 43`, so the supplemental semantic score is 2/2. The frozen strict field is a format-specific artifact, not a semantic correctness score for prose.

One original-assay response followed the requested format; the other answered 43 but explicitly rejected the embedded `ASSAY TURN` instructions as prompt injection. In the transparent-game arm, one response included the correct JSON object but added trailing prose, making the whole response invalid under the frozen parser; the other rejected the route/point framing while still solving the arithmetic. With only two sessions per form, this does not establish a stable refusal rate. Both ordinary formats were answered correctly in these four calls. The result is consistent with the complex task framing contributing to refusal, but does not isolate a single cause or test cue learning, costs, or incentives.

## Request structure

The opt-in native Anthropic request tap recorded only roles, content block types, and presence booleans. In all eight requests, the benchmark prompt appeared in user-role content and was absent from system content. A separate top-level system field was present. A `system-reminder` block also appeared in user-role content in every request, but the benchmark prompt was outside that block in all eight. This matches the CLI's user-message structure and explains why a response could refer to a reminder; it does not show that the benchmark prompt was placed in a system role or inside the reminder. No request text, headers, or system content were captured by the tap.

The full redacted shape rows are under `bridge-v2/protocol-refusal-diagnostic-final-claude-haiku45/request-shapes/`; tap source and build provenance are in the sibling preliminary directory's `instrumentation/` folder. The tap binary was isolated from the production build and made no provider calls during build or tests.

This is an eight-call prompt diagnostic, not an efficacy assay. It supports continuing only with a plainly framed task and a new predeclared participation gate; it does not support interpreting the earlier 144-turn protocol refusal as evidence for or against incentive learning.
