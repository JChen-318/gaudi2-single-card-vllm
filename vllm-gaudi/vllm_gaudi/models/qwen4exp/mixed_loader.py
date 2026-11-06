# SPDX-License-Identifier: Apache-2.0
"""``--load-format qwen4exp_mixed`` loader for the raw ModelOpt checkpoint.

The NVIDIA export keeps its shards under ``<model>/tensors/`` and names them in
``model.safetensors.index.json``. vLLM's default loader only looks at the
checkpoint root (it found no safetensors there, fell back to ``.pt``/``.bin``
and died in ``torch.load``), which is why the Qwen4Exp port needs its own
iterator.

Unlike ``qwen4exp_plain`` this loader does **not** dequantize. The model keeps
the checkpoint dtypes (packed ``uint8`` NVFP4 planes, ``float8_e4m3fn`` block
scales, FP8 linears), and Track B decodes them per layer:

* routed-expert planes are skipped here — ``NvFp4ExpertStore.load_from_source``
  reads them directly (63 GiB of tensor traffic the loader never touches);
* the PLE n-gram shards are skipped — they are mmap'ed into host memory by
  ``attach_ple_tables`` (25 GiB), never streamed to the device;
* ``mtp.*`` (speculative draft) and, for text-only serving, ``visual.*`` are
  skipped as well.

Layout::

    <model>/config.json
    <model>/model.safetensors.index.json
    <model>/tensors/model-000XX-of-000YY.safetensors
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from vllm.logger import init_logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

logger = init_logger(__name__)

_INDEX_NAME = "model.safetensors.index.json"

# Checkpoint prefixes this loader never materializes (see the module docstring).
SKIP_PREFIXES = ("mtp.",)
SKIP_SUBSTRINGS = (
    ".mlp.experts.",       # Track B streams the packed planes itself
    ".ple.ple_embedding.",  # host mmap, never a device tensor
)


def _weight_map(root: Path) -> dict[str, str]:
    index = root / _INDEX_NAME
    if not index.is_file():
        raise FileNotFoundError(f"ModelOpt checkpoint index is missing: {index}")
    return json.loads(index.read_text())["weight_map"]


def iter_raw_weights(
    root: Path,
    *,
    skip_vision: bool = False,
) -> Iterator[tuple[str, "object"]]:
    """Yield ``(name, tensor)`` in checkpoint dtype, shard by shard."""
    from safetensors import safe_open

    shards = sorted((root / "tensors").glob("*.safetensors")) or sorted(root.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards under {root}")

    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name.startswith(SKIP_PREFIXES):
                    continue
                if any(sub in name for sub in SKIP_SUBSTRINGS):
                    continue
                if skip_vision and ("visual." in name):
                    continue
                yield name, handle.get_tensor(name)


@register_model_loader("qwen4exp_mixed")
class Qwen4ExpMixedModelLoader(BaseModelLoader):
    """Stream the ModelOpt checkpoint in its native quantized dtypes."""

    def download_model(self, model_config) -> None:
        root = Path(model_config.model)
        _weight_map(root)
        shards = sorted((root / "tensors").glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no ModelOpt safetensors shards under {root}/tensors")
        logger.info("qwen4exp_mixed: %d shards under %s", len(shards), root / "tensors")

    def load_weights(self, model, model_config) -> None:
        root = Path(model_config.model)
        skip_vision = bool(getattr(model_config, "language_model_only", False))
        # ``language_model_only`` lives on the multimodal config, not ModelConfig.
        if not skip_vision:
            multimodal = getattr(model_config, "multimodal_config", None)
            skip_vision = bool(getattr(multimodal, "language_model_only", False))
        logger.info("qwen4exp_mixed: streaming native dtypes (skip_vision=%s)", skip_vision)
        model.load_weights(iter_raw_weights(root, skip_vision=skip_vision))
