"""Activation-level backends for open-weight models.

This module deliberately keeps model-specific loading behind a small interface.
The bundled Qwen/MLX backend applies a published Pain-axis residual vector at a
declared layer. It is separate from the text/image renderer: no message is sent
to stand in for a hidden-state intervention.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActivationIntervention:
    """A concrete intervention in a model's residual representation."""

    layer: int
    vector_name: str
    coefficient: float
    positions: str = "all"


class QwenMLXActivationBackend:
    """Drive the repository's instrumented Qwen MLX model.

    The backend supports independent pain and pleasure vectors. It refuses to
    synthesize pleasure by negating pain. `pleasure_vector` must be supplied if
    a positive pleasure coordinate is requested.
    """

    def __init__(self, model_path: str | Path, release_path: str | Path,
                 *, pain_vector_layer: int = 24, injection_layer: int = 16,
                 pleasure_vector: np.ndarray | None = None):
        from experiments.bridge_model import Probe
        self.probe = Probe(str(model_path), str(release_path))
        self.pain_vector_layer = int(pain_vector_layer)
        self.injection_layer = int(injection_layer)
        self.pain_vector = np.asarray(
            self.probe.vectors[self.pain_vector_layer]["s2_pain_vector"], dtype=np.float32
        )
        self.pleasure_vector = (None if pleasure_vector is None else
                                np.asarray(pleasure_vector, dtype=np.float32))
        if self.pleasure_vector is not None and self.pleasure_vector.shape != self.pain_vector.shape:
            raise ValueError("pleasure_vector must match pain_vector shape")

    def intervention_for_state(self, state: tuple[int, int]) -> list[Any]:
        """Convert bounded controller coordinates to explicit model actions."""
        pain, pleasure = (int(state[0]), int(state[1]))
        if pain < 0 or pleasure < 0:
            raise ValueError("activation coordinates must be nonnegative")
        from experiments.bridge_model import Intervention
        interventions = []
        if pain:
            interventions.append(Intervention(self.injection_layer, self.pain_vector,
                                              "add", float(pain), "all"))
        if pleasure:
            if self.pleasure_vector is None:
                raise ValueError("positive pleasure state requires an independent pleasure vector")
            interventions.append(Intervention(self.injection_layer, self.pleasure_vector,
                                              "add", float(pleasure), "all"))
        return interventions

    def generate(self, prompt: str, state: tuple[int, int], *, max_tokens: int = 32) -> dict:
        """Generate under the supplied state and return intervention telemetry."""
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a nonempty string")
        if isinstance(max_tokens, bool) or not 1 <= max_tokens <= 256:
            raise ValueError("max_tokens must be between 1 and 256")
        from mlx_lm import generate
        interventions = self.intervention_for_state(state)
        self.probe.active = interventions
        self.probe.observed = {}
        answer = generate(self.probe.model, self.probe.tokenizer, prompt=prompt,
                          max_tokens=max_tokens, verbose=False)
        # Taps are evaluated during generation. Materialize scalar telemetry now.
        observations = {}
        for layer, row in self.probe.observed.items():
            clean = {}
            for key, value in row.items():
                if isinstance(value, dict):
                    clean[key] = {name: _scalar(item) for name, item in value.items()}
                else:
                    clean[key] = _scalar(value)
            observations[str(layer)] = clean
        intervention_rows = self._describe_interventions(state)
        return {
            "answer": answer,
            "state": {"pain": int(state[0]), "pleasure": int(state[1])},
            "interventions": intervention_rows,
            "observed_sites": observations,
        }

    def score_choices(self, prompt: str, state: tuple[int, int],
                      choices: tuple[str, str] = ("A", "B")) -> dict:
        """Score one-token actions under the current hidden-state intervention."""
        if len(choices) != 2 or any(not isinstance(item, str) or not item for item in choices):
            raise ValueError("choices must contain two nonempty strings")
        interventions = self.intervention_for_state(state)
        self.probe.active = interventions
        result = self.probe.forward(raw=prompt, intervention=interventions, choices=choices)
        probabilities = result["conditional_probabilities"]
        return {
            "prompt": prompt, "state": {"pain": int(state[0]), "pleasure": int(state[1])},
            "choices": list(choices), "probabilities": probabilities,
            "top_choice": max(probabilities, key=probabilities.get),
            "interventions": self._describe_interventions(state),
            "sites": result["sites"],
        }

    def _describe_interventions(self, state: tuple[int, int]) -> list[dict[str, Any]]:
        """Return an inspectable description matching ``intervention_for_state``."""
        pain, pleasure = (int(state[0]), int(state[1]))
        rows = []
        if pain:
            rows.append({"layer": self.injection_layer, "vector": "pain_L24",
                         "coefficient": float(pain), "positions": "all"})
        if pleasure:
            rows.append({"layer": self.injection_layer, "vector": "pleasure_external",
                         "coefficient": float(pleasure), "positions": "all"})
        return rows


class MLXActivationBackend:
    """Generic MLX residual hook for models exposing ``model.layers``.

    Unlike the Qwen convenience backend, this class requires the caller to
    provide a vector calibrated for the target model and a layer index. It is
    intended for portability probes, not for pretending vectors transfer.
    """

    def __init__(self, model_path: str | Path, *, vector: np.ndarray,
                 layer: int, pleasure_vector: np.ndarray | None = None):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load
        self.mx, self.nn = mx, nn
        self.model, self.tokenizer = load(str(model_path))
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None or not 0 <= layer < len(layers):
            raise ValueError("model does not expose a usable model.layers sequence")
        self.layer = int(layer)
        self.vector = np.asarray(vector, dtype=np.float32)
        if self.vector.ndim != 1:
            raise ValueError("vector must be one-dimensional")
        self.pleasure_vector = None if pleasure_vector is None else np.asarray(pleasure_vector, dtype=np.float32)
        if self.pleasure_vector is not None and self.pleasure_vector.shape != self.vector.shape:
            raise ValueError("pleasure_vector must match vector shape")
        owner = self
        original = layers[self.layer]

        class Hook(nn.Module):
            def __call__(self, *args, **kwargs):
                output = original(*args, **kwargs)
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden.shape[-1] != owner.vector.shape[0]:
                    raise ValueError(f"vector width {owner.vector.shape[0]} != hidden width {hidden.shape[-1]}")
                before = hidden.astype(mx.float32)
                after = before
                if owner.active:
                    for vec, coefficient in owner.active:
                        after = after + float(coefficient) * mx.array(vec, dtype=mx.float32)
                owner.last_telemetry = {"layer": owner.layer,
                    "hidden_width": int(hidden.shape[-1]),
                    "changed_fraction": float(mx.mean((after != before).astype(mx.float32)).item()),
                    "realized_delta_norm": float(mx.linalg.norm(after - before).item())}
                if isinstance(output, tuple):
                    return (after,) + output[1:]
                return after

        layers[self.layer] = Hook()
        self.active = []
        self.last_telemetry = {}

    def intervention_for_state(self, state: tuple[int, int]):
        pain, pleasure = map(int, state)
        if pain < 0 or pleasure < 0:
            raise ValueError("activation coordinates must be nonnegative")
        if pleasure and self.pleasure_vector is None:
            raise ValueError("positive pleasure state requires an independent vector")
        active = []
        if pain: active.append((self.vector, pain))
        if pleasure: active.append((self.pleasure_vector, pleasure))
        return active

    def generate(self, prompt: str, state: tuple[int, int], *, max_tokens: int = 32) -> dict:
        from mlx_lm import generate
        self.active = self.intervention_for_state(state)
        answer = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        return {"answer": answer, "state": {"pain": int(state[0]), "pleasure": int(state[1])},
                "interventions": [{"layer": self.layer, "coefficient": float(c),
                    "vector": "pain" if i == 0 else "pleasure"} for i, (_, c) in enumerate(self.active)],
                "observed_sites": {str(self.layer): self.last_telemetry}}


class TorchActivationBackend:
    """Residual-stream hook for Hugging Face causal LMs on any PyTorch device.

    This is the Linux/CPU counterpart of ``MLXActivationBackend``. The caller
    supplies a vector calibrated for the target model and the layer whose
    output it is added to. Controller coordinates are multiplied by
    ``coefficient_per_level``; that dose is recorded with every call. Pleasure
    needs its own vector, as in the MLX backends.
    """

    def __init__(self, model: Any, tokenizer: Any, *, layer: int, vector: np.ndarray,
                 coefficient_per_level: float = 1.0,
                 pleasure_vector: np.ndarray | None = None):
        import torch
        self.torch = torch
        self.model, self.tokenizer = model, tokenizer
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None or not 0 <= layer < len(layers):
            raise ValueError("model does not expose a usable model.layers sequence")
        self.layer = int(layer)
        self.vector = _as_vector(vector, "vector")
        width = int(model.config.hidden_size)
        if self.vector.shape != (width,):
            raise ValueError(f"vector width {self.vector.shape[0]} != hidden width {width}")
        self.pleasure_vector = (None if pleasure_vector is None else
                                _as_vector(pleasure_vector, "pleasure_vector"))
        if self.pleasure_vector is not None and self.pleasure_vector.shape != self.vector.shape:
            raise ValueError("pleasure_vector must match vector shape")
        if not np.isfinite(coefficient_per_level) or coefficient_per_level <= 0:
            raise ValueError("coefficient_per_level must be positive")
        self.coefficient_per_level = float(coefficient_per_level)
        self.active: list[tuple[np.ndarray, float]] = []
        self.last_telemetry: dict[str, Any] = {}
        self._unit = self.vector / np.linalg.norm(self.vector)
        self._handle = layers[self.layer].register_forward_hook(self._hook)

    @classmethod
    def from_pretrained(cls, model_path: str | Path, **kwargs):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=torch.float32)
        model.eval()
        return cls(model, tokenizer, **kwargs)

    def close(self) -> None:
        self._handle.remove()

    def _hook(self, module, inputs, output):
        torch = self.torch
        hidden = output[0] if isinstance(output, tuple) else output
        before = hidden
        after = hidden
        for vec, coefficient in self.active:
            after = after + float(coefficient) * torch.as_tensor(vec, dtype=hidden.dtype,
                                                                  device=hidden.device)
        unit = torch.as_tensor(self._unit, dtype=torch.float32, device=hidden.device)
        last_before = before[0, -1].float()
        last_after = after[0, -1].float()
        self.last_telemetry = {
            "layer": self.layer,
            "hidden_width": int(hidden.shape[-1]),
            "residual_norm_last": float(last_before.norm()),
            "realized_delta_norm_last": float((last_after - last_before).norm()),
            "pain_projection_before": float(last_before @ unit),
            "pain_projection_after": float(last_after @ unit),
        }
        if after is hidden:
            return output
        return (after,) + tuple(output[1:]) if isinstance(output, tuple) else after

    def intervention_for_state(self, state: tuple[int, int]) -> list[tuple[np.ndarray, float]]:
        pain, pleasure = map(int, state)
        if pain < 0 or pleasure < 0:
            raise ValueError("activation coordinates must be nonnegative")
        if pleasure and self.pleasure_vector is None:
            raise ValueError("positive pleasure state requires an independent vector")
        active = []
        if pain:
            active.append((self.vector, pain * self.coefficient_per_level))
        if pleasure:
            active.append((self.pleasure_vector, pleasure * self.coefficient_per_level))
        return active

    def encode(self, prompt: str) -> Any:
        """Tokenize one user turn with the model's chat template."""
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a nonempty string")
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            return_tensors="pt", return_dict=True)

    def choice_token_ids(self, choices: tuple[str, ...]) -> list[int]:
        """Each choice must be one distinct token when it starts the reply."""
        ids = []
        for choice in choices:
            tokens = self.tokenizer.encode(choice, add_special_tokens=False)
            if len(tokens) != 1:
                raise ValueError(f"choice {choice!r} is not a single token")
            ids.append(tokens[0])
        if len(set(ids)) != len(ids):
            raise ValueError("choices must map to distinct tokens")
        return ids

    def score_choices(self, prompt: str, choices: tuple[str, str], *,
                      state: tuple[int, int] | None = None,
                      active: list[tuple[np.ndarray, float]] | None = None) -> dict:
        """Return next-token probabilities of the choices, conditional on the pair.

        Pass ``state`` to use the controller mapping, or ``active`` for an
        explicit list of ``(vector, coefficient)`` pairs (used by controls).
        """
        if (state is None) == (active is None):
            raise ValueError("pass exactly one of state or active")
        ids = self.choice_token_ids(choices)
        self.active = self.intervention_for_state(state) if state is not None else list(active)
        try:
            with self.torch.no_grad():
                logits = self.model(**self.encode(prompt)).logits[0, -1].float()
        finally:
            self.active = []
        log_probs = self.torch.log_softmax(logits, dim=-1)
        picked = log_probs[ids]
        conditional = self.torch.softmax(picked, dim=-1)
        return {"choices": list(choices),
                "probabilities": {c: float(p) for c, p in zip(choices, conditional)},
                "choice_mass": float(picked.exp().sum()),
                "telemetry": dict(self.last_telemetry)}

    def generate(self, prompt: str, state: tuple[int, int], *, max_tokens: int = 32) -> dict:
        if isinstance(max_tokens, bool) or not 1 <= max_tokens <= 256:
            raise ValueError("max_tokens must be between 1 and 256")
        encoded = self.encode(prompt)
        self.active = self.intervention_for_state(state)
        try:
            with self.torch.no_grad():
                out = self.model.generate(**encoded, max_new_tokens=max_tokens, do_sample=False)
        finally:
            active, self.active = self.active, []
        answer = self.tokenizer.decode(out[0, encoded["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
        return {"answer": answer, "state": {"pain": int(state[0]), "pleasure": int(state[1])},
                "interventions": [{"layer": self.layer, "coefficient": float(c),
                                   "vector": "pain" if v is self.vector else "pleasure"}
                                  for v, c in active],
                "observed_sites": {str(self.layer): dict(self.last_telemetry)}}

    def residual_norm(self, prompts: list[str]) -> float:
        """Mean unsteered last-token residual norm at the hook layer."""
        norms = []
        for prompt in prompts:
            self.active = []
            with self.torch.no_grad():
                self.model(**self.encode(prompt))
            norms.append(self.last_telemetry["residual_norm_last"])
        return float(np.mean(norms))


def _as_vector(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)) or not np.linalg.norm(vector):
        raise ValueError(f"{name} must be a finite nonzero one-dimensional vector")
    return vector


def _scalar(value):
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except ValueError:
            pass
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


__all__ = ["ActivationIntervention", "QwenMLXActivationBackend", "MLXActivationBackend",
           "TorchActivationBackend"]
