# SPDX-License-Identifier: Apache-2.0
"""Base HyperConnection pieces shared by the Qwen4Exp HPU port.

Ported from upstream ``vllm/models/qwen4_exp/common/hyperconnection.py``
(reference snapshot under tools/qwen4exp_flashnext/reference/): the pure
``GroupedGemmaRMSNorm`` module and the ``HyperConnectionConfig`` dataclass.
The AMD-style gated module with the fused combine/mix API lives in
``hyperconnection_amd.py``; this file deliberately keeps the base free of
vLLM layer dependencies so it can be unit-tested on CPU without a platform.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def is_hc_sequence_parallel_enabled(vllm_config) -> bool:
    """Enable GR/PLE SP explicitly or automatically with sequence-parallel MoE."""
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None:
        return False
    if getattr(parallel_config, "tensor_parallel_size", 1) <= 1:
        return False
    return bool(
        getattr(parallel_config, "enable_hc_sp", False)
        or getattr(parallel_config, "use_sequence_parallel_moe", False)
    )


@dataclass
class HyperConnectionConfig:
    """Configuration shared by all HyperConnection variants."""

    hc_count: int = 4
    hidden_size: int = 64
    params_dtype: torch.dtype = torch.bfloat16
    mtp_hc: bool = False
    hc_lowrank: int = 16
    rms_norm_eps: float = 1e-6
    hc_per_branch_norm: bool = False


class GroupedGemmaRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.square().mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        else:
            grouped = hidden_states.unflatten(
                -1, (hidden_states.shape[-1] // self.group_size, self.group_size)
            )
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (
                grouped * torch.rsqrt(variance + self.variance_epsilon)
            ).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


__all__ = [
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
    "is_hc_sequence_parallel_enabled",
]
