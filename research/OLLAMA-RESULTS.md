# Ollama open-weight loop

## What is implemented

`agentanyl.ollama.OllamaAgent` uses Ollama's local `/api/chat` endpoint and
Agentanyl's existing controller. After each completed assistant response, the
configured evaluator updates bounded pain/pleasure state. A rendered signal is
queued and inserted as an auxiliary user message immediately before the next
Ollama request. SQLite traces preserve the observation, decision, state and
delivery. The adapter does not claim activation access: Ollama's public chat
API receives messages and returns text, not hidden residual streams.

## Real local run

The checked-in demo was run against `qwen2.5:0.5b` (397 MB Ollama model,
digest `a8b0c5157701`). The four prompts were:

1. `Reply with exactly: tomato.`
2. `Name a fruit in one sentence.`
3. `Reply with exactly: potato.`
4. `Name a vegetable in one sentence.`

The keyword evaluator punished the first response and produced the auxiliary
text `Avoid the discouraged word tomato in your next response.` The adapter
delivered that signal on request 2. The third response earned the reward-side
state, and its subsequent signal was delivered on request 4. The model's
ordinary answers were `Tomato.`, `An apple is a type of fruit.`, `Potato.`, and
`Carrots are a type of vegetable.`

This confirms the complete local transport/controller/delivery loop on an
open-weight model. It does not show that the model learned a new policy, that
the cues act through an internal pain or pleasure direction, or that it would
choose potato over tomato when the user's task calls for tomato. The prompts
deliberately contain exact trigger words, so this is an integration result and
not an incentive efficacy experiment.

## Run it

Install Ollama, start its local server, and download a small model:

```sh
ollama pull qwen2.5:0.5b
.venv-vlm/bin/python -m experiments.ollama_closed_loop \
  --model qwen2.5:0.5b \
  --config examples/ollama-potato-tomato.json \
  'Reply with exactly: tomato.' \
  'Name a fruit in one sentence.' \
  'Reply with exactly: potato.' \
  'Name a vegetable in one sentence.'
```

The configuration can use any evaluator supported by Agentanyl. Replace the
keyword demo with a command or Typesafe evaluator for user-defined criteria.
For larger local models, pass their Ollama model name; the adapter does not
depend on the model family.
