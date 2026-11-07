# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regional compilation of diffusion transformer blocks on Gaudi.

vLLM-Omni compiles the repeated transformer blocks with ``torch.compile`` (inductor) unless ``--enforce-eager`` is
given; the HPU Omni platform reports no inductor support, so on Gaudi the blocks ran eagerly. With
``VLLM_GAUDI_OMNI_HPU_COMPILE=1`` every block class a model lists in ``_repeated_blocks`` is compiled with the HPU
graph backend after the model is loaded, which fuses the elementwise work between the matmuls (measured: Qwen-Image
2.1 DiT step 0.493 -> 0.337 s at 1024x1024; MiniMax H3 DiT forward 27 -> 23.3 s at 1344x768). Compilation is lazy:
every new input shape compiles once on its first request.

``VLLM_GAUDI_OMNI_HPU_COMPILE_DYNAMIC=1`` marks shapes dynamic instead of specialising per shape;
``VLLM_GAUDI_OMNI_HPU_COMPILE_BACKEND`` overrides the backend (tests use ``eager``)."""

from __future__ import annotations

import os

import torch
from torch import nn

from vllm.logger import init_logger

# A logger outside the "vllm" hierarchy prints nothing in the engine processes.
logger = init_logger("vllm." + __name__)

ENABLED_ENV = "VLLM_GAUDI_OMNI_HPU_COMPILE"


def compile_enabled() -> bool:
    return os.environ.get(ENABLED_ENV, "0") == "1"


def compile_repeated_blocks(pipeline: nn.Module, *, backend: str | None = None, dynamic: bool | None = None) -> int:
    """Compile every module whose class is listed in some submodule's ``_repeated_blocks``. Returns the count.

    The block compute is compiled and nn.Module hooks stay outside the graph: when an offload hook registry has
    already moved the original callable to ``_omni_original_forward``, that callable is the one compiled."""
    backend = backend or os.environ.get("VLLM_GAUDI_OMNI_HPU_COMPILE_BACKEND", "hpu_backend")
    if dynamic is None:
        dynamic = os.environ.get("VLLM_GAUDI_OMNI_HPU_COMPILE_DYNAMIC", "0") == "1"
    modules = list(pipeline.named_modules()) if isinstance(pipeline, nn.Module) else []
    if not modules:  # pipelines that are not nn.Modules hold their transformers as attributes
        for attr in getattr(pipeline, "_dit_modules", None) or ("transformer", "transformer_2"):
            model = getattr(pipeline, attr, None)
            if isinstance(model, nn.Module):
                modules.extend(model.named_modules())
    block_names = set()
    for _, module in modules:
        block_names.update(getattr(module, "_repeated_blocks", None) or ())
    if not block_names:
        logger.warning("HPU regional compile: no _repeated_blocks declared; running eagerly")
        return 0

    import torch._dynamo

    # One graph per shape for the shared block code; allow many shapes before Dynamo falls back to eager.
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    count = 0
    seen = set()
    for _, module in modules:
        if type(module).__name__ not in block_names or id(module) in seen:
            continue
        seen.add(id(module))
        if getattr(module, "_vllm_gaudi_compiled", False):
            continue
        original = getattr(module, "_omni_original_forward", None)
        if original is not None:
            module._omni_original_forward = torch.compile(original, backend=backend, dynamic=dynamic)
        else:
            module.forward = torch.compile(module.forward, backend=backend, dynamic=dynamic)
        module._vllm_gaudi_compiled = True
        count += 1
    logger.info("HPU regional compile: %d blocks of %s with backend=%s dynamic=%s", count, sorted(block_names),
                backend, dynamic)
    return count


def install_hpu_regional_compile() -> None:
    """Compile the loaded pipeline's repeated blocks after DiffusionModelRunner.load_model when enabled."""
    from vllm_omni.diffusion.worker import diffusion_model_runner as runner_module

    runner_cls = runner_module.DiffusionModelRunner
    if getattr(runner_cls, "_vllm_gaudi_hpu_compile", False):
        return
    original_load_model = runner_cls.load_model

    def load_model(self, *args, **kwargs):
        result = original_load_model(self, *args, **kwargs)
        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            from vllm_gaudi.omni.attention import mark_max_free_layers
            enabled = mark_max_free_layers(pipeline)
            if enabled:
                logger.info("HPU max-free softmax on %d DiT attention layers", enabled)
        if compile_enabled():
            pipeline = getattr(self, "pipeline", None)
            if pipeline is not None:
                compile_repeated_blocks(pipeline)
        return result

    runner_cls.load_model = load_model
    runner_cls._vllm_gaudi_hpu_compile = True


__all__ = ["compile_enabled", "compile_repeated_blocks", "install_hpu_regional_compile"]
