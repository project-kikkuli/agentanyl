#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL=${AGENTANYL_MODEL:-/tmp/agentanyl-qwen-4bit}
RELEASE=${AGENTANYL_PAIN_AXIS:-/tmp/agentanyl-pain-axis}
PYTHON=${AGENTANYL_PYTHON:-"$ROOT/.venv/bin/python"}

if [ ! -x "$PYTHON" ]; then
  echo "Missing MLX environment: $PYTHON" >&2
  echo "Create it according to research/REPRODUCE.md, or set AGENTANYL_PYTHON." >&2
  exit 1
fi
if [ ! -d "$MODEL" ] || [ ! -d "$RELEASE" ]; then
  echo "Model or Pain-axis release not found." >&2
  echo "Set AGENTANYL_MODEL and AGENTANYL_PAIN_AXIS to local paths." >&2
  exit 1
fi

exec "$PYTHON" -m experiments.playable_activation_demo \
  --model "$MODEL" --release "$RELEASE" "$@"
