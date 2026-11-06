# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection (Gated Residual) for the Qwen4Exp HPU port.

Ported from upstream ``vllm/models/qwen4_exp/amd/hyperconnection.py`` with the
AMD Triton glue kernels (amd/ops/hc.py) replaced by equivalent pure-torch
implementations (same math: grouped GemmaRMSNorm, silu(x/hc),
sigmoid-gated stream mean, 2*sigmoid(inj/hc) injection). The module API
follows the AMD variant because the model port mirrors it:

    hidden_states, block_input, injection = hc.mix(hidden_states)
    ... block runs on block_input ...
    hidden_states, block_input, injection = next_hc.combine_and_mix(
        hidden_states, block_output, injection)

Weights match the checkpoint naming via a merged down+inject
MergedColumnParallelLinear (``input_mix_weight_down_block_inject``) padded to
16 rows for alignment, exactly like the reference.
"""

from __future__ import annotations

import os

# Before torch / vLLM / any HPU graph compile (see _gc_kernel_path):
# importing this module first must still make the Qwen4Exp TPC library
# visible to the graph compiler. Idempotent.
from ._gc_kernel_path import ensure_gc_kernel_path

ensure_gc_kernel_path()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from vllm.model_executor.layers.linear import (  # noqa: E402
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.models.utils import maybe_prefix  # noqa: E402

from ._materialize import in_compile_only_mode as _in_compile_only_mode  # noqa: E402
from ._materialize import materialized as _materialized  # noqa: E402
from .hyperconnection_base import (  # noqa: E402
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
    is_hc_sequence_parallel_enabled,
)

__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
    "is_hc_sequence_parallel_enabled",
]

# ---------------------------------------------------------------------------
# H13 fused TPC path (csrc/qwen4exp/kernels/qwen4exp_hc_*_bf16_gaudi2.c).
# Five custom ops replace the ~30 eager dispatches of the torch path:
#   hc_norm, hc_combine, hc_combine_norm, hc_down_epilogue, hc_gate_mix
# with the two GEMMs (down+inject, up) kept on the MME via F.linear on the
# stored weights. Disabled with VLLM_QWEN4EXP_HC_TPC=0; any load failure or
# non-HPU / non-BF16 / odd geometry falls back to the exact torch code below.
# ---------------------------------------------------------------------------
_HC_TPC_OP_NAMES = ("norm", "combine", "combine_norm", "gate_mix", "down_epilogue")
_HC_TPC_OPS: dict | bool | None = None
_HC_TPC_AVAILABLE = False


def _hc_tpc_ops() -> dict | None:
    global _HC_TPC_OPS, _HC_TPC_AVAILABLE
    if os.environ.get("VLLM_QWEN4EXP_HC_TPC", "1") == "0":
        return None
    if _HC_TPC_OPS is not None:
        return _HC_TPC_OPS or None  # type: ignore[return-value]
    try:
        from .packed_moe import _ensure_dequant_op

        _ensure_dequant_op()  # same extension .so registers the HC ops
        namespace = torch.ops.custom_op
        ops = {
            name: getattr(namespace, f"custom_qwen4exp_hc_{name}_bf16_gaudi2")
            for name in _HC_TPC_OP_NAMES
        }
    except Exception:
        ops = False
    _HC_TPC_OPS = ops
    _HC_TPC_AVAILABLE = bool(ops)
    return ops or None  # type: ignore[return-value]


def hc_grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, hc_count: int
) -> torch.Tensor:
    """Per-stream RMSNorm with Gemma (1 + w) affine; torch fallback of the
    AMD ``grouped_gemma_rmsnorm`` kernel."""
    num_tokens, dim = x.shape
    hc_dim = dim // hc_count
    grouped = x.float().unflatten(-1, (hc_count, hc_dim))
    variance = grouped.square().mean(dim=-1, keepdim=True)
    normalized = grouped * torch.rsqrt(variance + eps)
    if weight.numel() == dim:  # per-branch affine in checkpoint HC-outer layout
        weight = weight.float().unflatten(-1, (hc_count, hc_dim))
    else:  # shared affine repeats across streams
        weight = weight.float().expand(hc_count, hc_dim)
    y = normalized * (1.0 + weight)
    return y.flatten(-2).to(x.dtype)


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    return F.silu(x.float() / hc_count).to(x.dtype)


def hc_gate_mix(xn: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    """sigmoid(gate) * xn averaged over the HC streams -> [N, hc_dim]."""
    num_tokens, dim = xn.shape
    hc_dim = dim // hc_count
    g = torch.sigmoid(gate.float().unflatten(-1, (hc_count, hc_dim)))
    x = xn.float().unflatten(-1, (hc_count, hc_dim))
    return (g * x).mean(dim=-2).to(xn.dtype)


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    """out = residual + block_output * (2 * sigmoid(inj / hc)) per stream."""
    num_tokens, dim = residual.shape
    hc_dim = dim // hc_count
    inj = 2.0 * torch.sigmoid(injection_logits.float() / hc_count)
    res = residual.float().unflatten(-1, (hc_count, hc_dim))
    out = res + block_output.float().unsqueeze(-2) * inj.float().unsqueeze(-1)
    return out.flatten(-2).to(residual.dtype)


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused combine + next block's grouped norm (same boundary as unfused)."""
    combined = hc_combine(residual, block_output, injection_logits, hc_count)
    return combined, hc_grouped_gemma_rmsnorm(combined, norm_weight, eps, hc_count)


class GatedResidual(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``combine_and_mix()`` runs the pre pipeline (grouped GemmaRMSNorm ->
    merged low-rank down+inject GEMM -> scaled silu -> up GEMM -> sigmoid ->
    gated mean over the HC streams). When passed a pending block output and
    injection, it fuses their residual combine with the RMSNorm. Final mixers
    use ``use_combine=False`` and do not produce a new injection.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hc_lowrank = config.hc_lowrank
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine

        norm_size = (
            self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size
        )
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        self.pad_size = (-(self.hc_lowrank + self.hc_count)) % 16 if use_combine else 0
        linear_kwargs = dict(
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            return_bias=False,
        )
        if use_combine:
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                [self.hc_lowrank, self.hc_count]
                + ([self.pad_size] if self.pad_size else []),
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"),
                **linear_kwargs,
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                self.hyper_hidden_size,
                self.hc_lowrank,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                **linear_kwargs,
            )
        self.input_mix_weight_up = ReplicatedLinear(
            self.hc_lowrank,
            self.hyper_hidden_size,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            **linear_kwargs,
        )

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size

    def _tpc_ops(self, *tensors: torch.Tensor) -> dict | None:
        """Fused ops for this call, or None to take the exact torch path."""
        if self.hidden_size % 128 or self.hc_norm.weight.dtype != torch.bfloat16:
            return None
        for tensor in tensors:
            if tensor.device.type != "hpu" or tensor.dtype != torch.bfloat16:
                return None
        return _hc_tpc_ops()

    def _gamma2d(self) -> torch.Tensor:
        # [Sg, D] materialization of the flat norm weight (Sg == hc_count, or
        # 1 shared). A *view* of the 1-D parameter must not reach a custom op:
        # in lazy mode its graph node dies with "Empty tensor optional ...
        # Strided Params" (the same trap packed_moe.materialized_ids works
        # around). The copy is cached and rebuilt only when the weight's
        # storage or version changes (load_state_dict / .to(device)), so the
        # steady-state cost is zero launches.
        weight = self.hc_norm.weight.detach()
        key = (weight.data_ptr(), weight._version, weight.device, weight.dtype)
        cached = getattr(self, "_gamma2d_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        gamma = torch.empty(
            (weight.numel() // self.hidden_size, self.hidden_size),
            device=weight.device,
            dtype=weight.dtype,
        )
        gamma.copy_(weight.view(-1, self.hidden_size))
        # In compile-only warmup the copy never executes, so the tensor has
        # no data; caching it would poison every real forward's norm input
        # (ValidateSyncInputTensors, empty tensor_data). Only cache a
        # genuinely executed copy (the Kimi-K25 guard for the same trap).
        if not _in_compile_only_mode():
            self._gamma2d_cache = (key, gamma)
        return gamma

    def _project_and_mix_tpc(
        self, xn: torch.Tensor, ops: dict
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # F.linear on the stored weights is the same MME GEMM the vLLM linear
        # modules run at TP=1 (no bias/quant), without their Python wrapper.
        if self.use_combine:
            down_weight = self.input_mix_weight_down_block_inject.weight
        else:
            down_weight = self.input_mix_weight_down.weight
        merged = F.linear(xn, down_weight)
        # Custom-op inputs are materialized in lazy mode (see
        # _materialize): merged/gate are GEMM outputs and xn is the norm
        # op's output -- produced tensors, the empty-tensor_data form.
        lowrank, injection = ops["down_epilogue"](
            _materialized(merged), self.hc_lowrank, self.hc_count)
        gate = F.linear(lowrank, self.input_mix_weight_up.weight)
        block_input = ops["gate_mix"](_materialized(xn), _materialized(gate), self.hc_count)
        return block_input, (injection if self.use_combine else None)

    def _project_and_mix(self, xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # All linear layers here are built with return_bias=False, so forward
        # returns a bare tensor (no bias tuple to unpack).
        if self.use_combine:
            merged = self.input_mix_weight_down_block_inject(xn)
            lowrank, injection, _ = merged.split(
                [self.hc_lowrank, self.hc_count, self.pad_size], dim=-1
            )
        else:
            lowrank = self.input_mix_weight_down(xn)
            injection = None
        lowrank = hc_silu(lowrank, self.hc_count)
        gate = self.input_mix_weight_up(lowrank)
        return hc_gate_mix(xn, gate, self.hc_count), injection

    def mix(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        ops = self._tpc_ops(hidden_states)
        if ops is not None and hidden_states.dim() == 2:
            xn = ops["norm"](
                _materialized(hidden_states), self._gamma2d(), float(self.config.rms_norm_eps)
            )
            block_input, injection = self._project_and_mix_tpc(xn, ops)
            return hidden_states, block_input, injection
        xn = hc_grouped_gemma_rmsnorm(
            hidden_states, self.hc_norm.weight, self.config.rms_norm_eps, self.hc_count
        )
        block_input, injection = self._project_and_mix(xn)
        return hidden_states, block_input, injection

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if prev_block_output is None or prev_injection is None:
            return self.mix(hidden_states)
        ops = self._tpc_ops(hidden_states, prev_block_output, prev_injection)
        if ops is not None and hidden_states.dim() == 2:
            hidden_states, xn = ops["combine_norm"](
                _materialized(hidden_states),
                _materialized(prev_block_output),
                _materialized(prev_injection),
                self._gamma2d(),
                float(self.config.rms_norm_eps),
            )
            block_input, injection = self._project_and_mix_tpc(xn, ops)
            return hidden_states, block_input, injection
        hidden_states, xn = hc_combine_norm(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )
        block_input, injection = self._project_and_mix(xn)
        return hidden_states, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        ops = self._tpc_ops(hidden_states, block_output, injection)
        if ops is not None and hidden_states.dim() == 2:
            return ops["combine"](
                _materialized(hidden_states), _materialized(block_output), _materialized(injection)
            )
        return hc_combine(hidden_states, block_output, injection, self.hc_count)
