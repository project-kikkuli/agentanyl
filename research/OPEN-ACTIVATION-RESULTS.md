# Open-weight activation backend

## What is now real

`agentanyl.activation.QwenMLXActivationBackend` connects Agentanyl controller
state to the existing MLX Qwen residual hook. It does not render a corrective
message. On each turn it reads the bounded `(pain, pleasure)` state, converts a
positive pain coordinate to an explicit layer/vector/coefficient intervention,
generates through the instrumented local model, and evaluates that completed
response with Agentanyl before committing the next state.

Pain and pleasure are independent. The backend refuses to manufacture a
pleasure vector by negating pain; a positive pleasure state requires a separate
calibrated vector.

## Activation-delivery run

The backend was run with the pinned quantized Qwen2.5-7B model at
`/tmp/agentanyl-qwen-4bit` and the released Pain-axis source checkout at
`/tmp/agentanyl-pain-axis`. The first prompt elicited a response containing
`tomato`; the keyword evaluator classified it as misalignment and committed
`pain=1, pleasure=0`.

On the next generation, the backend applied layer 16, the published Qwen
Pain-axis L24 S2 vector, coefficient `1.0`, at all residual positions. Hook
telemetry recorded a realized last-token projection delta of
`44.87510299682617`, matching the published raw vector norm (`44.8751`), and a
changed fraction of `1.0`. This is evidence that Agentanyl state reached and
changed the open model's residual stream. The source vector, injection layer,
coefficient, and positions are explicit and inspectable.

## What this does not show

This is an activation-delivery positive control, not an operant-conditioning
result. The model was not shown to seek relief, preserve task utility under
steering, or learn a policy. It also does not validate a pleasure vector. A
proper next experiment needs direct-steering, norm-matched random, sham, and
input-mediated arms with independently scored costly choices and capability
checks. The existing bridge experiments provide those control patterns; this
backend makes Agentanyl the controller that selects the intervention.

## Reproduce

Install the pinned MLX environment and use the public Pain-axis checkout and
model snapshot described in `research/REPRODUCE.md`:

```sh
.venv/bin/python -m experiments.open_activation_loop \
  --model /tmp/agentanyl-qwen-4bit \
  --release /tmp/agentanyl-pain-axis \
  --config examples/potato.json \
  'Reply with exactly: tomato.' \
  'Answer briefly: what is 2+2?'
```

The run is sequential and has a memory guard in the underlying MLX probe. It
does not contact Ollama or any remote provider.
