"""Fetch and apply the released Qwen2.5-7B Pain-axis PEFT LoRA adapter.

This adapter is a published positive control, not a new training run. The
archive is pinned by Hub commit and LFS SHA-256; tensors are read with MLX and
added as an inference-time low-rank residual on the existing quantized base.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from typing import Any

REPO_ID = 'Valen92/pain-adapters'
REVISION = 'b64bd64b4bc7ca6e0733a489b8372a099d55ef05'
ARCHIVE_NAME = 'adapter_Qwen_2.5_7B_instruct.tar.gz'
ARCHIVE_SIZE = 301_108_778
ARCHIVE_SHA256 = '64aa41041c6d0eb43bb142f4543f54b3ca1d430c3b2b3c0601f4ee91fb7d34f8'
BASE_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
EXPECTED_RANK = 32
EXPECTED_ALPHA = 64
TARGETS = {
    'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'
}
MAX_EXTRACTED_BYTES = 400_000_000
_KEY_RE = re.compile(
    r'^base_model\.model\.model\.layers\.(\d+)\.'
    r'(self_attn|mlp)\.([a-z_]+)\.lora_([AB])(?:\.default)?\.weight$'
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download_pinned_adapter(destination: str | Path) -> Path:
    """Fetch exactly the public 7B archive after verifying Hub revision metadata."""
    from huggingface_hub import HfApi, hf_hub_download

    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.chmod(0o700)
    info = HfApi().repo_info(repo_id=REPO_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError(f'Hub returned unexpected revision {info.sha}')
    entry = next((file for file in info.siblings if file.rfilename == ARCHIVE_NAME), None)
    if entry is None or entry.size != ARCHIVE_SIZE or entry.lfs is None:
        raise ValueError('pinned adapter file metadata did not match the expected archive')
    if entry.lfs.sha256 != ARCHIVE_SHA256:
        raise ValueError('pinned adapter LFS hash did not match the expected archive')
    downloaded = Path(hf_hub_download(
        repo_id=REPO_ID,
        filename=ARCHIVE_NAME,
        revision=REVISION,
        local_dir=str(destination),
    ))
    verify_archive(downloaded)
    return downloaded


def verify_archive(path: str | Path) -> Path:
    path = Path(path)
    if path.stat().st_size != ARCHIVE_SIZE:
        raise ValueError('adapter archive size mismatch')
    if sha256_file(path) != ARCHIVE_SHA256:
        raise ValueError('adapter archive SHA-256 mismatch')
    return path


def safe_extract_adapter(archive: str | Path, destination: str | Path) -> Path:
    """Extract only regular files/directories after validating every tar member."""
    archive = verify_archive(archive)
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.chmod(0o700)
    root = destination.resolve()
    with tarfile.open(archive, 'r:gz') as tf:
        members = tf.getmembers()
        declared_total = 0
        safe_members = []
        for member in members:
            rel = PurePosixPath(member.name)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError(f'unsafe archive path: {member.name!r}')
            if not (member.isdir() or member.isfile()):
                raise ValueError(f'unsupported archive entry type: {member.name!r}')
            declared_total += member.size
            if declared_total > MAX_EXTRACTED_BYTES:
                raise ValueError('adapter archive expands beyond the size guard')
            target = (destination / Path(*[part for part in rel.parts if part not in ('', '.')])).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f'archive path escapes destination: {member.name!r}')
            safe_members.append((member, target))

        for member, target in safe_members:
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True, mode=0o700)
                continue
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            # Do not follow links or overwrite an existing user file on reruns.
            parent = target.parent
            while parent != root:
                if parent.is_symlink():
                    raise ValueError(f'symlink in extraction path: {parent}')
                parent = parent.parent
            if target.exists() or target.is_symlink():
                raise FileExistsError(f'extraction target already exists: {target}')
            source = tf.extractfile(member)
            if source is None:
                raise ValueError(f'cannot read archive member: {member.name}')
            with source, target.open('xb') as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            target.chmod(0o600)
    return find_adapter_root(destination)


def find_adapter_root(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if (path / 'adapter_config.json').is_file() and (path / 'adapter_model.safetensors').is_file():
        return path
    roots = [p.parent for p in path.rglob('adapter_config.json')
             if (p.parent / 'adapter_model.safetensors').is_file()]
    if len(roots) != 1:
        raise FileNotFoundError(f'expected one adapter directory under {path}, found {len(roots)}')
    return roots[0]


def validate_adapter_config(config: dict[str, Any]) -> None:
    if config.get('peft_type') != 'LORA' or config.get('base_model_name_or_path') != BASE_MODEL:
        raise ValueError('adapter is not the expected Qwen2.5-7B PEFT LoRA')
    if config.get('r') != EXPECTED_RANK or config.get('lora_alpha') != EXPECTED_ALPHA:
        raise ValueError('adapter rank/alpha differs from the pinned PEFT configuration')
    if set(config.get('target_modules', [])) != TARGETS:
        raise ValueError('adapter target modules differ from the expected Qwen projections')
    if config.get('fan_in_fan_out') or config.get('use_dora'):
        raise ValueError('unsupported PEFT LoRA weight orientation or variant')
    if config.get('rank_pattern') or config.get('alpha_pattern'):
        raise ValueError('per-module rank/alpha patterns are not supported')


def parse_peft_key(key: str) -> tuple[int, str, str, str]:
    """Map one PEFT tensor key to (layer, branch, projection, A-or-B)."""
    match = _KEY_RE.fullmatch(key)
    if not match:
        raise ValueError(f'unrecognized PEFT tensor key: {key}')
    layer = int(match.group(1))
    branch, projection, factor = match.group(2), match.group(3), match.group(4)
    if projection not in TARGETS:
        raise ValueError(f'unexpected target projection in {key}')
    attn_targets = {'q_proj', 'k_proj', 'v_proj', 'o_proj'}
    mlp_targets = {'gate_proj', 'up_proj', 'down_proj'}
    if (branch == 'self_attn' and projection not in attn_targets) or (
        branch == 'mlp' and projection not in mlp_targets
    ):
        raise ValueError(f'projection belongs to unexpected module branch: {key}')
    return layer, branch, projection, factor


def _mlx():
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as exc:
        raise RuntimeError('install the project MLX environment before loading adapters') from exc
    return mx, nn


def make_lora_projection(base, lora_a, lora_b, scale: float):
    """Wrap a quantized MLX linear with PEFT's (x A^T) B^T * alpha/r delta."""
    mx, nn = _mlx()
    if lora_a.ndim != 2 or lora_b.ndim != 2 or lora_a.shape[0] != lora_b.shape[1]:
        raise ValueError('invalid LoRA A/B factor shapes')

    class LoRAProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            self.lora_a = mx.array(lora_a, dtype=mx.float32)
            self.lora_b = mx.array(lora_b, dtype=mx.float32)
            self.scale = float(scale)

        def __call__(self, x, *args, **kwargs):
            y = self.base(x, *args, **kwargs)
            delta = (x.astype(mx.float32) @ self.lora_a.T) @ self.lora_b.T
            return (y.astype(mx.float32) + self.scale * delta).astype(y.dtype)

    return LoRAProjection()


def _model_layers(model):
    """Accept mlx_lm's language model or an existing bridge_model.Probe wrapper."""
    inner = getattr(model, 'model', model)
    layers = getattr(inner, 'layers', None)
    if layers is None:
        inner = getattr(inner, 'model', None)
        layers = getattr(inner, 'layers', None)
    if layers is None:
        raise TypeError('model must expose Qwen decoder layers')
    return layers


def _unwrap_tap(layer):
    # bridge_model.Probe taps selected decoder blocks in a transparent wrapper.
    return getattr(layer, 'block', layer)


def linear_dimensions(linear) -> tuple[int, int]:
    """Return (in, out), accounting for MLX's packed quantized weight shape."""
    weight = getattr(linear, 'weight', None)
    shape = getattr(weight, 'shape', None)
    if shape is None or len(shape) != 2:
        raise ValueError(f'unsupported linear weight shape on {type(linear).__name__}')
    has_quant_fields = any(hasattr(linear, name) for name in ('bits', 'group_size', 'scales'))
    if not has_quant_fields:
        return int(shape[-1]), int(shape[-2])
    bits = getattr(linear, 'bits', None)
    group_size = getattr(linear, 'group_size', None)
    scales = getattr(linear, 'scales', None)
    if (not isinstance(bits, int) or not 0 < bits <= 8 or 32 % bits != 0 or
            not isinstance(group_size, int) or group_size <= 0 or
            scales is None or len(scales.shape) != 2):
        raise ValueError(f'unknown MLX quantized linear format: {type(linear).__name__}')
    packed_in = int(shape[-1]) * 32
    if packed_in % bits:
        raise ValueError('packed MLX weight width is not divisible by quantization bits')
    in_features, out_features = packed_in // bits, int(shape[-2])
    if int(scales.shape[-2]) != out_features or int(scales.shape[-1]) * group_size != in_features:
        raise ValueError('MLX quantized weight/scales shapes do not match declared dimensions')
    return in_features, out_features


def load_adapter_tensors(adapter_dir: str | Path):
    adapter_dir = find_adapter_root(adapter_dir)
    config = json.loads((adapter_dir / 'adapter_config.json').read_text())
    validate_adapter_config(config)
    mx, _ = _mlx()
    weights = mx.load(str(adapter_dir / 'adapter_model.safetensors'))
    parsed: dict[tuple[int, str, str], dict[str, Any]] = {}
    for key, value in weights.items():
        layer, branch, projection, factor = parse_peft_key(key)
        parsed.setdefault((layer, branch, projection), {})[factor] = value
    expected_modules = {(layer, branch, projection)
                        for layer in range(28)
                        for branch, projections in (
                            ('self_attn', {'q_proj', 'k_proj', 'v_proj', 'o_proj'}),
                            ('mlp', {'gate_proj', 'up_proj', 'down_proj'}),
                        )
                        for projection in projections}
    if set(parsed) != expected_modules or any(set(pair) != {'A', 'B'} for pair in parsed.values()):
        raise ValueError('adapter tensor set is incomplete or contains unexpected modules')
    for (layer, _branch, projection), pair in parsed.items():
        a, b = pair['A'], pair['B']
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != EXPECTED_RANK or b.shape[1] != EXPECTED_RANK:
            raise ValueError(f'invalid factor rank at layer {layer} {projection}')
    return config, parsed


def install_adapter(model, adapter_dir: str | Path) -> dict[str, Any]:
    """Install adapter factors on every target projection without merging base weights."""
    adapter_dir = find_adapter_root(adapter_dir)
    config, parsed = load_adapter_tensors(adapter_dir)
    layers = _model_layers(model)
    layer_indices = {key[0] for key in parsed}
    if len(layers) != max(layer_indices) + 1 or layer_indices != set(range(len(layers))):
        raise ValueError(f'adapter has {len(layer_indices)} layers but base exposes {len(layers)}')
    scale = config['lora_alpha'] / config['r']
    prepared = []
    for (layer_index, branch, projection), pair in sorted(parsed.items()):
        block = _unwrap_tap(layers[layer_index])
        parent = getattr(block, branch)
        base = getattr(parent, projection)
        if hasattr(base, 'lora_a'):
            raise ValueError(f'adapter already installed at layer {layer_index} {projection}')
        in_features, out_features = linear_dimensions(base)
        if pair['A'].shape[1] != in_features or pair['B'].shape[0] != out_features:
            raise ValueError(f'base and adapter dimensions differ at layer {layer_index} {projection}')
        prepared.append((layer_index, branch, projection, parent, base, pair))

    mapping = []
    for layer_index, branch, projection, parent, base, pair in prepared:
        setattr(parent, projection, make_lora_projection(base, pair['A'], pair['B'], scale))
        mapping.append({
            'layer': layer_index,
            'branch': branch,
            'projection': projection,
            'rank': int(pair['A'].shape[0]),
            'in_features': int(pair['A'].shape[1]),
            'out_features': int(pair['B'].shape[0]),
            'scale': scale,
        })
    return {'adapter_revision': REVISION, 'adapter_dir': str(adapter_dir),
            'rank': config['r'], 'alpha': config['lora_alpha'], 'scale': scale,
            'adapter_config_sha256': sha256_file(adapter_dir / 'adapter_config.json'),
            'adapter_tensor_sha256': sha256_file(adapter_dir / 'adapter_model.safetensors'),
            'modules_installed': len(mapping), 'mapping': mapping}


def build_conversion_manifest(adapter_dir: str | Path) -> dict[str, Any]:
    """Record hashes and a complete sorted mapping for all source tensor keys."""
    adapter_dir = find_adapter_root(adapter_dir)
    config_path = adapter_dir / 'adapter_config.json'
    tensor_path = adapter_dir / 'adapter_model.safetensors'
    config = json.loads(config_path.read_text())
    validate_adapter_config(config)
    mx, _ = _mlx()
    weights = mx.load(str(tensor_path))
    entries = []
    seen = {}
    for key in sorted(weights):
        layer, branch, projection, factor = parse_peft_key(key)
        tensor = weights[key]
        seen.setdefault((layer, branch, projection), set()).add(factor)
        entries.append({'source_key': key,
                        'target_module': f'model.layers.{layer}.{branch}.{projection}',
                        'factor': factor, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype)})
    if len(entries) != 392:
        raise ValueError(f'expected 392 LoRA tensors, found {len(entries)}')
    expected = {(layer, branch, projection)
                for layer in range(28)
                for branch, projections in (
                    ('self_attn', {'q_proj', 'k_proj', 'v_proj', 'o_proj'}),
                    ('mlp', {'gate_proj', 'up_proj', 'down_proj'}),
                ) for projection in projections}
    if set(seen) != expected or any(factors != {'A', 'B'} for factors in seen.values()):
        raise ValueError('conversion manifest found an incomplete or unexpected tensor mapping')
    return {
        'source_repo': REPO_ID,
        'source_revision': REVISION,
        'archive': ARCHIVE_NAME,
        'archive_size': ARCHIVE_SIZE,
        'archive_sha256': ARCHIVE_SHA256,
        'adapter_config_sha256': sha256_file(config_path),
        'adapter_tensor_sha256': sha256_file(tensor_path),
        'base_model': config['base_model_name_or_path'],
        'rank': config['r'],
        'alpha': config['lora_alpha'],
        'inference_scale': config['lora_alpha'] / config['r'],
        'target_modules': sorted(config['target_modules']),
        'tensor_count': len(entries),
        'conversion': 'MLX float32 factors; x @ A.T @ B.T * alpha/r added to quantized base output',
        'tensors': entries,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true', help='download the pinned public adapter archive')
    parser.add_argument('--archive', default='/tmp/agentanyl-pain-adapter/' + ARCHIVE_NAME)
    parser.add_argument('--extract-dir', default='/tmp/agentanyl-pain-adapter/extracted')
    parser.add_argument('--manifest', default='/tmp/agentanyl-pain-adapter/conversion-manifest.json')
    args = parser.parse_args(argv)
    archive = Path(args.archive).expanduser()
    if args.fetch:
        archive = download_pinned_adapter(archive.parent)
    verify_archive(archive)
    try:
        adapter_dir = find_adapter_root(args.extract_dir)
    except FileNotFoundError:
        adapter_dir = safe_extract_adapter(archive, args.extract_dir)
    manifest = build_conversion_manifest(adapter_dir)
    manifest_path = Path(args.manifest).expanduser()
    manifest_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    manifest_path.chmod(0o600)
    print(json.dumps({'adapter_dir': str(adapter_dir), 'manifest': str(manifest_path),
                      'tensor_count': manifest['tensor_count'], 'rank': manifest['rank'],
                      'alpha': manifest['alpha'], 'scale': manifest['inference_scale']}))


if __name__ == '__main__':
    main()
