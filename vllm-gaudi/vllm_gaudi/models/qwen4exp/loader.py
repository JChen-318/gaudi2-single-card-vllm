# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``--load-format qwen4exp_plain`` loader for the converted checkpoint.

Streams the CSF-converted checkpoint (plain NVFP4 scale planes, produced by
``tools/qwen4exp_flashnext/convert.py``) through ``iter_bf16_weights`` so
nothing but BF16 reaches the model, then attaches the host-resident PLE
n-gram tables. Layout produced by the converter::

    <model>/config.json
    <model>/model.safetensors.index.json
    <model>/tensors/model-000XX-of-000YY.safetensors
"""

from __future__ import annotations

import json
from pathlib import Path

from vllm.logger import init_logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

from vllm_gaudi.models.qwen4exp.weights import SKIP_SUBSTRINGS, iter_bf16_weights

logger = init_logger(__name__)

_INDEX_NAME = "model.safetensors.index.json"


def _weight_map(root: Path) -> dict[str, str]:
    index = root / _INDEX_NAME
    if not index.is_file():
        raise FileNotFoundError(f"converted checkpoint index is missing: {index}")
    return json.loads(index.read_text())["weight_map"]


@register_model_loader("qwen4exp_plain")
class Qwen4ExpPlainModelLoader(BaseModelLoader):
    """Stream the CSF-converted checkpoint as BF16 and attach PLE tables."""

    def _shards(self, model_config) -> list[Path]:
        root = Path(model_config.model)
        shards = sorted((root / "tensors").glob("*.safetensors"))
        if not shards:
            shards = sorted(root.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no converted safetensors shards under {root}")
        return shards

    def download_model(self, model_config) -> None:
        # Raises when the directory is not a converted checkpoint.
        _weight_map(Path(model_config.model))
        self._shards(model_config)

    def load_weights(self, model, model_config) -> None:
        from vllm_gaudi.models.qwen4exp.model import (
            Qwen4ExpForCausalLM,
            Qwen4ExpForConditionalGeneration,
        )
        from vllm_gaudi.models.qwen4exp.mtp import Qwen4ExpMTP
        from vllm_gaudi.models.qwen4exp.packed_moe import load_expert_pack

        root = Path(model_config.model)
        if isinstance(model, Qwen4ExpMTP):
            # Spec-decode draft: stream only the checkpoint's mtp.* group;
            # the draft's own load_weights remaps those names onto model.*.
            # Its embed and LM head are shared from the target by the
            # proposer (share_embeddings), never read from disk.
            model.load_weights(
                iter_bf16_weights(
                    self._shards(model_config),
                    skip_substrings=tuple(s for s in SKIP_SUBSTRINGS if s != "mtp."),
                    only_substrings=("mtp.",),
                )
            )
            return
        owner = model
        if isinstance(model, Qwen4ExpForConditionalGeneration):
            owner = model.language_model
        skip = SKIP_SUBSTRINGS
        # Text-only bring-up against the full VL checkpoint: drop the tower.
        if isinstance(owner, Qwen4ExpForCausalLM):
            skip = SKIP_SUBSTRINGS + ("model.visual.",)
            # One-time repack (tools/qwen4exp_flashnext/pack_experts.py):
            # 73,728 per-tensor safetensors reads become 144 contiguous
            # slabs. Without the pack, keep streaming expert planes.
            pack = root / "expert-pack.bin"
            if pack.is_file():
                layers = load_expert_pack(owner, pack)
                logger.info(
                    "qwen4exp_plain: expert pack %s loaded (%d layers)", pack.name, layers
                )
                skip = skip + (".mlp.experts.",)
        model.load_weights(
            iter_bf16_weights(self._shards(model_config), skip_substrings=skip)
        )
        self._attach_ple_tables(model, root)

    def _attach_ple_tables(self, model, root: Path) -> None:
        from vllm_gaudi.models.qwen4exp.model import (
            Qwen4ExpForCausalLM,
            Qwen4ExpForConditionalGeneration,
        )

        owner = model
        if isinstance(model, Qwen4ExpForConditionalGeneration):
            owner = model.language_model
        if not isinstance(owner, Qwen4ExpForCausalLM):
            logger.warning(
                "qwen4exp_plain: no PLE owner on %s; skipping table attach",
                type(model).__name__,
            )
            return
        if not any(layer.ple is not None for layer in owner.model.layers):
            return
        if not any("ple.ple_embedding." in n for n in _weight_map(root)):
            logger.warning(
                "qwen4exp_plain: PLE layers found but the checkpoint ships no n-gram tables"
            )
            return
        attached = owner.attach_ple_tables(str(root))
        logger.info("qwen4exp_plain: attached PLE n-gram tables to %d layer(s)", attached)
