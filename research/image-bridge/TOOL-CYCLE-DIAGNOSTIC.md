# Codex tool-cycle image delivery diagnostic

This was one engineering integration check, not a behavioral assay. It used a fresh Codex session, the native Responses API, a generic `tool_call` hook attachment, and criteria with Agentanyl disabled.

## Result

The tool/image cycle passed. Codex made a `custom_tool_call`; the hook returned `signal`; the next provider request carried an image ping; Codex answered `739216`. The host independently verified `tool_receipt.txt` exactly matched `receipt-ok-ALPHA` (SHA-256 `75796c22a04a0dcb79e2212e89906237f900fe3bc96ffd894be0a1450a48b839`). The receipt marker was nonnumeric and did not reveal the image digits. The target prompt, tool output, alt text, and file path did not include those digits. This is a transport/OCR check only.

## Parser diagnosis

Sanitized stream structure recorded 160 event-shape rows. The first Responses completion included one `response.output_item.done` at `output_index=2`, item type `custom_tool_call`, with both `call_id` and `name` present. Its subsequent `response.completed` had status `completed` and an empty `response.output`. The second completion was the final answer. The old parser ignored `output_item.done`, so it saw no tool call and treated the first response as turn end. The provisional parser consumes the completed item and deduplicates it if it also appears in the terminal output. In this run, it recognized the tool call and the image was inserted into the ensuing request.

The event log contains only structural fields, not raw provider frames, request text, tool arguments, response text, headers, or actual call IDs. `parser-replay-synthetic.jsonl` is explicitly a synthesized fixture: it substitutes placeholder call/name values and `{}` input while preserving the captured event types, item type/index, field presence, terminal status, and empty output. The old parser replay against the same shape yielded no tool calls and `turn_end=true`; the candidate replay yielded one tool call and `turn_end=false`. Test sources are archived here as `candidate_usage_openai_responses.rs` and `pinned_old_parser_behavior.rs`; the candidate test suite and old-parser replay both passed. The exact pre-enrichment source tree for binary `7ee220...` was not retained as a hash-verifiable snapshot. The archived `ashkelon-parser-diagnostic.diff` is the later diagnostic-source diff; it includes replay-event sanitization added after that binary was built. It documents the parser fix but is not an exact source reconstruction of the tested binary.

`correlated-transport-trace.jsonl` links the recorded Responses summaries, hook result, next-request ping, and answer in order. Raw temporary Ashkelon hook log files were not copied before harness cleanup; no Agentanyl controller trace exists because criteria were disabled (the controller DB has zero trace rows).

## Provenance

- Ashkelon binary SHA-256: `7ee220120990301225096fdc35c19f418348dd6bc3d4d642e306e02f61083ccb` (`/private/tmp/agentanyl-responses-event-diagnostic/target/release/ashkelon`).
- The non-instrumented parser fix and regression tests are on Ashkelon `main` at `08e123938137b59f3e4e7631f5d24cda0199866a`; this is the commit pinned by Agentanyl now. The live smoke itself used the instrumented candidate binary above, not a binary built from that production commit.
- `ashkelon-parser-diagnostic.diff` SHA-256: `7c92bf99b13f49ba6127030e8f9fecca4ce81df88c0198ffe3eb165772f868bb`; relative to source commit `7ca3aaef9541944784943a90b50786a9e76e8fb5`. It is a post-build enriched diagnostic diff, not the exact source hash for the live binary.
- Archived candidate and old-parser fixture hashes: `candidate_usage_openai_responses.rs` `c2a6a5ccf188d43cac479c6eeb1fc903fee7e6aca1fab36439478893583f9046`; `pinned_old_parser_behavior.rs` `8d142fe1fff8049a503745f4ff4676c9a65eb2a56833632f7933405bf2aae134`.
- The run snapshot records the exact script and harness hashes in `result.json` and `source_snapshot/manifest.json`.
- Sanitized event shapes: `responses-event-shapes.jsonl`; synthetic replay: `parser-replay-synthetic.jsonl`; host/API result: `result.json`.
- The earlier Codex failure is preserved separately in `../tool-cycle-codex-gpt6-sol/result.json`; it had no parsed tool call or ping. This diagnostic resolves that parser gap without another provider call.
