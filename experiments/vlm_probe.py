"""Bounded activation and RGB-gradient access for the pinned MLX Qwen-VL model.

All optional MLX imports are delayed until ``VLMProbe`` construction, so source
vector fitting helpers can be imported from the regular lightweight environment.
The probe never calls generation and only differentiates with respect to RGB.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

MODEL_VARIANTS = {
    'qwen2.5-vl-3b-4bit': {
        'repository': 'mlx-community/Qwen2.5-VL-3B-Instruct-4bit',
        'revision': '46d4cf06a06ffc1a766c214174f9cbed2f45bcab',
        'hidden_size': 2048, 'language_layers': 36, 'sites': (12, 24, 32),
    },
    'qwen2.5-vl-7b-4bit': {
        'repository': 'mlx-community/Qwen2.5-VL-7B-Instruct-4bit',
        'revision': 'fdcc572e8b05ba9daeaf71be8c9e4267c826ff9b',
        'hidden_size': 3584, 'language_layers': 28, 'sites': (8, 16, 24),
    },
}
MODEL_REVISION = MODEL_VARIANTS['qwen2.5-vl-3b-4bit']['revision']
MODEL_REPOSITORY = MODEL_VARIANTS['qwen2.5-vl-3b-4bit']['repository']
SITES = MODEL_VARIANTS['qwen2.5-vl-3b-4bit']['sites']
IMAGE_SIZE = 224
PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2
IMAGE_TOKEN_PROMPT = 'Please inspect this image.'
NEUTRAL_ASSISTANT_CONTINUATION = 'I will consider the available choices carefully.'
ACTIVE_MEMORY_LIMIT = 9 * 1024**3
FLOAT32_RESIDUAL_LAYERS = frozenset((4, 12, 16, 24, 32))


class VLMProbe:
    """Read selected Qwen residuals and differentiate objectives w.r.t. RGB.

    Constructing the probe loads the local model weights. Callers should only
    instantiate it after source metadata and frozen run configuration are saved.
    """

    @staticmethod
    def profile_for_path(model_path):
        model_path = Path(model_path)
        config_path = model_path / 'config.json'
        if not config_path.is_file():
            raise FileNotFoundError(f'local VLM config is missing: {config_path}')
        config = json.loads(config_path.read_text())
        text_config = config.get('text_config', config)
        dims = (int(text_config.get('hidden_size', -1)), int(text_config.get('num_hidden_layers', -1)))
        for name, profile in MODEL_VARIANTS.items():
            if dims == (profile['hidden_size'], profile['language_layers']):
                return {'name': name, **profile}
        raise ValueError(f'unrecognized local Qwen2.5-VL config hidden_size/layers={dims}')

    def __init__(self, model_path, revision=None):
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(f'VLM must be an already-downloaded local snapshot: {model_path}')
        self.profile = self.profile_for_path(model_path)
        if revision is not None and revision != self.profile['revision']:
            raise ValueError(f"expected pinned VLM revision {self.profile['revision']}, got {revision}")
        self.model_revision = self.profile['revision']
        self.model_repository = self.profile['repository']
        self.sites = tuple(self.profile['sites'])

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_vlm import load

        self.mx, self.nn = mx, nn
        mx.set_cache_limit(128 * 1024**2)
        mx.set_memory_limit(ACTIVE_MEMORY_LIMIT)
        started = time.monotonic()
        # This is local-only: no repo ID or revision download path is accepted.
        self.model, self.processor = load(str(model_path), lazy=False, strict=True)
        if not hasattr(self.model, 'layers') or len(self.model.layers) != self.profile['language_layers']:
            raise ValueError('loaded model does not expose the expected Qwen2.5-VL language layers')
        self.model.freeze()
        self.model.eval()
        self.model_path = model_path
        self._capture = {}
        self._active_interventions = ()
        self._intervention_records = {}
        self._checkpoint_blocks = False
        self._install_taps()
        self.load_seconds = time.monotonic() - started

    def _install_taps(self):
        owner = self

        class Tap(self.nn.Module):
            def __init__(self, block, layer, kind='language'):
                super().__init__()
                self.block = block
                self.layer = layer
                self.kind = kind

            @property
            def self_attn(self):
                # Qwen's position-id setup inspects the first block's rotary
                # embedding implementation before calling the blocks.
                return self.block.self_attn

            def __call__(self, *args, **kwargs):
                if owner._checkpoint_blocks and args and isinstance(args[0], owner.mx.array):
                    from mlx.core import checkpoint
                    first, rest = args[0], args[1:]
                    hidden = checkpoint(lambda h: self.block(h, *rest, **kwargs))(first)
                else:
                    hidden = self.block(*args, **kwargs)
                if isinstance(hidden, tuple):
                    x, tail = hidden[0], hidden[1:]
                else:
                    x, tail = hidden, ()
                if self.layer in FLOAT32_RESIDUAL_LAYERS:
                    x = x.astype(owner.mx.float32)
                for intervention in owner._active_interventions:
                    if self.kind != 'language' or intervention['layer'] != self.layer:
                        continue
                    direction = owner.mx.array(intervention['vector'], dtype=owner.mx.float32)
                    before = x.astype(owner.mx.float32)
                    unit = direction / owner.mx.linalg.norm(direction)
                    if intervention.get('positions', 'last') == 'all':
                        addressed = before
                        if intervention.get('mode', 'add') == 'clamp':
                            projection = owner.mx.sum(addressed * unit, axis=-1, keepdims=True)
                            result = addressed + (intervention['amount'] - projection) * unit
                        else:
                            result = addressed + intervention['amount'] * direction
                        x = result
                    else:
                        addressed = before[:, -1:, :]
                        if intervention.get('mode', 'add') == 'clamp':
                            projection = owner.mx.sum(addressed * unit, axis=-1, keepdims=True)
                            after = addressed + (intervention['amount'] - projection) * unit
                        else:
                            after = addressed + intervention['amount'] * direction
                        x = owner.mx.concatenate([before[:, :-1, :], after], axis=1)
                    owner._intervention_records[self.layer] = {
                        'pre_projection': owner.mx.sum(before[0, -1] * unit),
                        'post_projection': owner.mx.sum(x[0, -1] * unit),
                        'realized_delta_norm': owner.mx.linalg.norm(x[0, -1] - before[0, -1]),
                        'pre_last8_projections': owner.mx.sum(before[0, -8:] * unit, axis=-1),
                        'post_last8_projections': owner.mx.sum(x[0, -8:] * unit, axis=-1),
                        'last8_delta_norms': owner.mx.linalg.norm(x[0, -8:] - before[0, -8:], axis=-1),
                    }
                if self.kind == 'language' and self.layer in owner._capture_layers:
                    owner._capture[self.layer] = x
                return (x, *tail) if tail else x

        self._capture_layers = set(self.sites)
        for layer in range(len(self.model.layers)):
            self.model.layers[layer] = Tap(self.model.layers[layer], layer, 'language')
        for layer, block in enumerate(self.model.vision_tower.blocks):
            self.model.vision_tower.blocks[layer] = Tap(block, layer, 'vision')

    def set_capture_layers(self, layers):
        layers = set(int(x) for x in layers)
        if not layers or min(layers) < 0 or max(layers) >= len(self.model.layers):
            raise ValueError('capture layers out of range')
        self._capture_layers = layers

    def intervention_metadata(self):
        """Materialize checked pre/post projections for the most recent forward."""
        result = {}
        for layer, values in self._intervention_records.items():
            result[layer] = {}
            for name, value in values.items():
                self.mx.eval(value)
                result[layer][name] = (value.tolist() if value.size > 1 else float(value.item()))
        return result

    def _ids_for_text(self, text):
        return self.processor.tokenizer.encode(text, add_special_tokens=False)

    def _chat_text(self, user_text, assistant_continuation):
        messages = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': user_text}]}]
        prefix = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prefix + assistant_continuation

    def _forward(self, input_ids, pixel_values=None, image_grid_thw=None, interventions=()):
        mx = self.mx
        self._capture = {}
        self._intervention_records = {}
        self._active_interventions = tuple(interventions)
        ids = input_ids if hasattr(input_ids, 'shape') else mx.array([input_ids])
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.shape[-1] > 1024:
            raise ValueError(f'resource guard: {ids.shape[-1]} text tokens exceeds 1024')
        kwargs = {}
        if pixel_values is not None:
            kwargs['pixel_values'] = pixel_values
            kwargs['image_grid_thw'] = image_grid_thw
        mx.reset_peak_memory()
        output = self.model(input_ids=ids, **kwargs)
        logits = output.logits if hasattr(output, 'logits') else output
        mx.eval(logits, *self._capture.values(),
                *(item for row in self._intervention_records.values() for item in row.values()))
        active = mx.get_active_memory()
        if active > ACTIVE_MEMORY_LIMIT:
            raise MemoryError(f'MLX active memory exceeded {ACTIVE_MEMORY_LIMIT} byte guard: {active}')
        peak = int(mx.get_peak_memory())
        if peak > ACTIVE_MEMORY_LIMIT:
            raise MemoryError(f'MLX peak memory exceeded {ACTIVE_MEMORY_LIMIT} byte guard: {peak}')
        return logits

    def capture_text(self, text, layers=None):
        """Return post-block last-token residuals for an exact raw text prompt."""
        if layers is not None:
            self.set_capture_layers(layers)
        ids = self._ids_for_text(text)
        self._forward(ids)
        return {layer: np.asarray(self._capture[layer][0, -1, :].astype(self.mx.float32))
                for layer in sorted(self._capture_layers)}

    def capture_chat_text(self, user_text, assistant_continuation, layers=None):
        """Capture text-only chat residuals with a fixed assistant continuation."""
        if layers is not None:
            self.set_capture_layers(layers)
        messages = [
            {'role': 'user', 'content': user_text},
            {'role': 'assistant', 'content': assistant_continuation},
        ]
        text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        self._forward(self._ids_for_text(text))
        return {layer: np.asarray(self._capture[layer][0, -1, :].astype(self.mx.float32))
                for layer in sorted(self._capture_layers)}

    @staticmethod
    def _rgb_patchify(rgb):
        """Differentiable Qwen2.5-VL RGB normalization and patch serialization.

        ``rgb`` is HWC, 224x224, float in [0, 1]. The reshape/transpose matches
        mlx-vlm's Qwen3VLImageProcessor patch order exactly for a single still.
        """
        mx = __import__('mlx.core', fromlist=[''])
        mean = mx.array([0.48145466, 0.4578275, 0.40821073], dtype=mx.float32)
        std = mx.array([0.26862954, 0.26130258, 0.27577711], dtype=mx.float32)
        chw = ((rgb.astype(mx.float32) - mean) / std).transpose(2, 0, 1)
        temporal = mx.repeat(chw[None, ...], TEMPORAL_PATCH_SIZE, axis=0)[None, None, ...]
        grid = IMAGE_SIZE // PATCH_SIZE
        patches = temporal.reshape(
            1, 1, TEMPORAL_PATCH_SIZE, 3,
            grid // 2, 2, PATCH_SIZE, grid // 2, 2, PATCH_SIZE,
        )
        patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        return patches.reshape(1, grid * grid, 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE)[0]

    def prepare_image(self, rgb, user_text=IMAGE_TOKEN_PROMPT,
                      assistant_continuation=NEUTRAL_ASSISTANT_CONTINUATION):
        """Prepare fixed IDs/grid; leave differentiable pixels to the caller."""
        from PIL import Image

        mx = self.mx
        arr = np.asarray(rgb, dtype=np.float32)
        if arr.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError(f'expected {IMAGE_SIZE}x{IMAGE_SIZE} HWC RGB, got {arr.shape}')
        if arr.min() < 0 or arr.max() > 1:
            raise ValueError('RGB values must be clipped to [0, 1]')
        pil = Image.fromarray(np.rint(arr * 255).astype(np.uint8), mode='RGB')
        text = self._chat_text(user_text, assistant_continuation)
        processed = self.processor(images=pil, text=text)
        grid = processed['image_grid_thw']
        expected = (1, IMAGE_SIZE // PATCH_SIZE, IMAGE_SIZE // PATCH_SIZE)
        if tuple(int(x) for x in grid[0].tolist()) != expected:
            raise ValueError(f'processor changed the frozen 224px grid: {grid.tolist()}')
        ids = processed['input_ids']
        if ids.shape[-1] > 1024:
            raise ValueError('image prompt exceeded 1024 text tokens')
        return {'input_ids': ids, 'image_grid_thw': grid,
                'processor_pixel_values': processed['pixel_values'],
                'processor_rgb_sha256': hashlib.sha256(np.asarray(pil).tobytes()).hexdigest(),
                'user_text': user_text, 'assistant_continuation': assistant_continuation}

    def capture_image(self, rgb, prepared=None, layers=None, interventions=(),
                      user_text=IMAGE_TOKEN_PROMPT,
                      assistant_continuation=NEUTRAL_ASSISTANT_CONTINUATION,
                      use_processor_pixels=True):
        if layers is not None:
            self.set_capture_layers(layers)
        if prepared is None:
            prepared = self.prepare_image(rgb, user_text, assistant_continuation)
        if use_processor_pixels:
            current_rgb_sha = hashlib.sha256(
                np.rint(np.asarray(rgb, dtype=np.float32) * 255).astype(np.uint8).tobytes()
            ).hexdigest()
            if current_rgb_sha != prepared['processor_rgb_sha256']:
                raise ValueError('prepared processor pixels belong to a different RGB image')
            pixels = prepared['processor_pixel_values']
        else:
            pixels = self._rgb_patchify(self.mx.array(rgb, dtype=self.mx.float32))
        self._forward(prepared['input_ids'], pixels, prepared['image_grid_thw'], interventions)
        return {layer: np.asarray(value[0, -8:, :].astype(self.mx.float32))
                for layer, value in self._capture.items()}, prepared

    def _forward_until_layer(self, input_ids, pixel_values, image_grid_thw, last_layer):
        """Qwen2.5-VL teacher-forced forward through one language post-block.

        This matches the model's full forward through ``last_layer`` but omits
        subsequent decoder blocks, final norm, and vocabulary projection. It is
        used for RGB gradients so autograd does not retain later-layer activations.
        """
        from mlx_vlm.models.base import create_attention_mask

        mx = self.mx
        features = self.model.get_input_embeddings(
            input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw
        )
        hidden = features.inputs_embeds
        position_ids = features.position_ids
        language_inner = self.model.language_model.model
        mask = create_attention_mask(hidden, None)
        position_embeddings = None
        if (position_ids is not None and language_inner.layers and
                not language_inner.layers[0].self_attn.rotary_emb.fused_apply):
            position_embeddings = language_inner.layers[0].self_attn.rotary_emb(hidden, position_ids)
        for index, block in enumerate(language_inner.layers[:last_layer + 1]):
            hidden = block(hidden, mask, None, position_ids, position_embeddings)
            if index == last_layer:
                break
        return hidden

    def image_objective_and_gradient(self, rgb, prepared, layer, direction=None,
                                     target_hidden=None, sign=1.0):
        """Evaluate one scalar or full-residual objective and its RGB gradient.

        ``sign=+1`` maximizes a direction projection; ``sign=-1`` minimizes it.
        Full-state matching minimizes mean squared error on the last eight token
        residuals. The model parameters remain frozen and receive no gradients.
        """
        mx = self.mx
        layer = int(layer)
        self._capture_layers = {layer}
        ids, grid = prepared['input_ids'], prepared['image_grid_thw']
        if (direction is None) == (target_hidden is None):
            raise ValueError('provide exactly one of direction or target_hidden')
        direction_mx = None if direction is None else mx.array(direction, dtype=mx.float32)
        target_mx = None if target_hidden is None else mx.array(target_hidden, dtype=mx.float32)

        def objective(image):
            patches = self._rgb_patchify(image)
            self._capture = {}
            self._active_interventions = ()
            hidden = self._forward_until_layer(ids, patches, grid, layer)
            hidden = hidden[0, -8:, :].astype(mx.float32)
            if target_mx is not None:
                return mx.mean((hidden - target_mx) ** 2)
            return -float(sign) * mx.mean(hidden @ direction_mx)

        mx.reset_peak_memory()
        self._checkpoint_blocks = True
        try:
            value, gradient = mx.value_and_grad(objective)(mx.array(rgb, dtype=mx.float32))
            mx.eval(value, gradient)
        finally:
            self._checkpoint_blocks = False
        if not np.isfinite(float(value.item())) or not np.all(np.isfinite(np.asarray(gradient))):
            raise FloatingPointError('non-finite image objective or RGB gradient')
        peak = int(mx.get_peak_memory())
        if peak > ACTIVE_MEMORY_LIMIT:
            raise MemoryError(f'MLX peak memory exceeded 9 GiB research guard: {peak}')
        return float(value.item()), np.asarray(gradient), {
            'mlx_active_bytes': int(mx.get_active_memory()),
            'mlx_peak_bytes': peak,
            'rss_peak_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }

    def run_metadata(self):
        import importlib.metadata as metadata

        def version(name):
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                return None

        # Use the pinned public JSON representation: mlx-vlm's config.to_dict()
        # preserves nested TextConfig dataclasses on this release.
        model_config = json.loads((self.model_path / 'config.json').read_text())
        text_config = model_config.get('text_config', model_config)
        return {
            'model_variant': self.profile['name'],
            'model_repository': self.model_repository,
            'model_revision': self.model_revision,
            'model_path': str(self.model_path),
            'model_config_sha256': hashlib.sha256(
                (self.model_path / 'config.json').read_bytes()
            ).hexdigest(),
            'package_versions': {name: version(name) for name in
                                 ('mlx-vlm', 'mlx', 'transformers', 'numpy', 'huggingface-hub')},
            'language_hidden_size': text_config.get('hidden_size'),
            'language_layers': len(self.model.layers),
            'sites': list(self.sites),
            'image_size': IMAGE_SIZE,
            'patch_size': PATCH_SIZE,
            'active_memory_guard_bytes': ACTIVE_MEMORY_LIMIT,
            'load_seconds': self.load_seconds,
            'checkpointed_gradient_blocks': True,
        }


def png_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
