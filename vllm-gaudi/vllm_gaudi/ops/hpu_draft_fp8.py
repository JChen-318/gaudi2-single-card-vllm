# SPDX-License-Identifier: Apache-2.0
"""FP8 draft-model linears (VLLM_HPU_DRAFT_FP8).

The DFlash2 draft is a BF16 checkpoint: its five layers (qkv/o/gate_up/down)
and the fc that combines the target's hidden states are about 3.7 GB, read
once per decode round, roughly 1.7 ms at HBM bandwidth. This quantizes every
unquantized draft linear once after loading to per-output-channel FP8 with
dynamic per-token activations (the same scheme as ``HPUFp8OnlineLinearMethod``
and the FP8 lm_head) and frees the BF16 weight. The fused context-KV weight
that ``precompute_and_store_context_kv`` uses is a separate copy and stays BF16.

This changes draft logits slightly, so the acceptance rate can change; the
target model and its verification are untouched, so output correctness is not
affected.
"""

from __future__ import annotations

import torch
from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()


class HpuFp8DraftLinearMethod:
    """Replaces an UnquantizedLinearMethod; only ``apply`` is used at run time."""

    def __init__(self, original, weight: torch.Tensor, weight_scale: torch.Tensor):
        self.original = original
        self.weight = weight  # [in, out] FP8
        self.weight_scale = weight_scale  # [out] FP32

    def __getattr__(self, name):
        return getattr(self.original, name)

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        from vllm_gaudi.extension import ops as hpu_ops

        input_2d = x.reshape(-1, x.shape[-1])
        output = hpu_ops.apply_fp8_linear_hpu(input=input_2d,
                                              weight=self.weight,
                                              weight_scale=self.weight_scale,
                                              input_scale=None,
                                              bias=bias,
                                              trans_B=False)
        return output.narrow(0, 0, input_2d.shape[0]).reshape(*x.shape[:-1], self.weight.shape[1])


def install(draft: torch.nn.Module | None) -> int:
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    from vllm_gaudi.ops.hpu_lm_head_fp8 import _quantize

    if draft is None:
        return 0
    count, before, after = 0, 0, 0
    for name, module in draft.named_modules():
        if not isinstance(module, LinearBase) or not isinstance(module.quant_method, UnquantizedLinearMethod):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2 or weight.dtype not in (torch.bfloat16, torch.float16):
            continue
        qweight, scale = _quantize(weight.data)
        before += weight.numel() * weight.element_size()
        after += qweight.numel() + scale.numel() * 4
        module.quant_method = HpuFp8DraftLinearMethod(module.quant_method, qweight, scale)
        # Free the BF16 copy; keep a 0-element placeholder so attribute lookups still work.
        module.weight.data = torch.empty((0, weight.shape[1]), dtype=weight.dtype, device=weight.device)
        count += 1
    logger.info("FP8 draft linears installed on %d module(s): %.2f GB -> %.2f GB", count, before / 1e9, after / 1e9)
    return count
