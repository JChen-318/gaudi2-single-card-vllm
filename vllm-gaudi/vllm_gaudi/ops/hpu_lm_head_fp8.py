# SPDX-License-Identifier: Apache-2.0
"""FP8 vocabulary projection (VLLM_HPU_LM_HEAD_FP8).

The 248K x 5120 BF16 lm_head is read twice per decode round (target logits
and DFlash2 candidates), about 1.1 ms each at HBM bandwidth. This quantizes it
once to per-output-channel FP8 with dynamic per-token activations, the same
scheme as ``HPUFp8OnlineLinearMethod``. The BF16 weight is left in place for
any other user (for example a tied embedding).

This changes logits slightly. ``VLLM_HPU_LM_HEAD_FP8_CHECK=1`` also computes
the BF16 logits on every projection and accumulates, on the device and inside
the compiled region, the rows compared, top-1 agreement, the largest
probability and logit differences and the summed KL(bf16 || fp8). The runner
logs them every 50 decode steps (diagnostic only: it costs the BF16 read).
"""

from __future__ import annotations

import os

import torch
from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()

_CHECK = os.environ.get("VLLM_HPU_LM_HEAD_FP8_CHECK", "0") not in ("", "0")
_methods: list["HpuFp8LmHeadMethod"] = []


class HpuFp8LmHeadMethod:
    """Drop-in for the lm_head's quant method; only ``apply`` is used for logits."""

    def __init__(self, original, weight: torch.Tensor, weight_scale: torch.Tensor):
        self.original = original
        self.weight = weight
        self.weight_scale = weight_scale
        # rows, top-1 same, max |dp|, max |dlogit|, sum KL
        self.stats = torch.zeros(5, dtype=torch.float32, device=weight.device) if _CHECK else None
        _methods.append(self)

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
        output = output.reshape(*x.shape[:-1], self.weight.shape[1])
        if self.stats is not None:
            reference = self.original.apply(layer, x, bias=bias).float().reshape(-1, output.shape[-1])
            quantized = output.float().reshape(-1, output.shape[-1])
            log_ref = reference.log_softmax(-1)
            log_q = quantized.log_softmax(-1)
            current = torch.stack((
                torch.tensor(float(reference.shape[0]), device=reference.device),
                (reference.argmax(-1) == quantized.argmax(-1)).float().sum(),
                (log_ref.exp() - log_q.exp()).abs().amax(),
                (reference - quantized).abs().amax(),
                (log_ref.exp() * (log_ref - log_q)).sum(),
            ))
            merged = torch.cat((self.stats[:2] + current[:2], torch.maximum(self.stats[2:4], current[2:4]),
                                self.stats[4:] + current[4:]))
            self.stats.copy_(merged)
        return output


def report() -> None:
    for index, method in enumerate(_methods):
        if method.stats is None:
            continue
        rows, same, dprob, dlogit, kl = method.stats.tolist()
        if rows:
            logger.info("lm_head FP8 check [%d]: rows=%d top1_agree=%.5f max|dp|=%.4g max|dlogit|=%.4g mean_KL=%.3g",
                        index, rows, same / rows, dprob, dlogit, kl / rows)


def _quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm_gaudi.extension import ops as hpu_ops

    rows = weight.shape[0]
    quantized = torch.empty(weight.shape, dtype=torch.float8_e4m3fn, device=weight.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=weight.device)
    step = 16384  # bounded temporaries; the full BF16 abs() alone would be 2.5 GB
    for start in range(0, rows, step):
        part = weight[start:start + step]
        part_scale = (part.abs().amax(dim=-1, keepdim=True).float() + 1e-8) / float(hpu_ops.FP8_MAX)
        quantized[start:start + step] = torch.ops.hpu.cast_to_fp8_v2(part, part_scale.reciprocal(), False, False,
                                                                     torch.float8_e4m3fn)[0]
        scale[start:start + step] = part_scale
    return quantized.t(), scale.squeeze(-1)


def install(*models: torch.nn.Module) -> int:
    """Replace every ParallelLMHead quant method in ``models``; shared weights are quantized once."""
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    done: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    count = 0
    for model in models:
        if model is None:
            continue
        for module in model.modules():
            if not isinstance(module, ParallelLMHead) or isinstance(module.quant_method, HpuFp8LmHeadMethod):
                continue
            weight = getattr(module, "weight", None)
            if weight is None or weight.dtype not in (torch.bfloat16, torch.float16) or weight.ndim != 2:
                continue
            key = weight.data_ptr()
            if key not in done:
                done[key] = _quantize(weight.data)
            module.quant_method = HpuFp8LmHeadMethod(module.quant_method, *done[key])
            count += 1
    logger.info("FP8 lm_head installed on %d head(s), %d distinct weight(s)", count, len(done))
    return count
