# SPDX-License-Identifier: Apache-2.0
"""H14 fused-TPC GDN fast path for Qwen4Exp (csrc/qwen4exp GDN kernels).

Three TPC launches replace the many eager dispatches of the torch GDN mixer
core, with the two input projections merged into one MME GEMM:

* ``gdn_conv`` / ``gdn_conv_rows``: causal depthwise conv1d + conv-state
  update + SiLU (decode one-token form / prefill row-block form),
* ``gdn_recur``: L2-norm of q/k, gating ``g = -exp(A_log) * softplus(a +
  dt_bias)``, ``beta = sigmoid(b)``, and the gated delta-rule recurrence over
  the [Hv, 128, 128] BF16 state, kept in TPC registers across the sequence,
* ``gdn_gated_norm``: ``rmsnorm(y) * act(z)`` before ``out_proj``.

Kill switch: ``VLLM_QWEN4EXP_GDN_TPC=0``. Unset, the switches here are on for
Qwen4Exp only (``enable_for_qwen4exp``); other models sharing the qwen3_5 GDN
layer keep their previous path. Any non-HPU tensor, unsupported
geometry/dtype, missing extension, or non-uniform sequence layout falls back
to the exact torch code in ``qwen3_5.py`` (the caller keeps that path).
All custom-op inputs are materialized contiguous tensors: un-materialized
views with non-standard strides are never passed to the HPU ops.

H25 (this unit) adds the *dispatch* half of the same fast path, behind its own
umbrella switch ``VLLM_QWEN4EXP_GDN_H25=0`` (default on for Qwen4Exp):

* the fused prefill conv now runs under the runner's ``padding_mask_flat``
  (live ``vllm serve`` always supplies one, which used to send the whole
  prefill to the torch conv),
* the short-span TPC recurrence gate follows the measured HPU crossover
  (64 tokens, was 16),
* the gated-norm dispatch picks the winning kernel per row count: TPC for
  short spans, Habana ``FusedRMSNorm`` for the mid band (prefill), torch
  above it (``hpu_fused_rmsnorm_gated`` loses from ~37k rows up).

The prefill chunk recurrence is selected in ``qwen3_5.py`` from the knobs
below: the FlashQLA gate-free reformulation for every bucketed prefill (it
removes the pairwise-decay and ``Ag`` materializations that dominate the
launch-bound HPU chunk path) plus BF16 compute from ~1k tokens up.
"""
from __future__ import annotations

import os

# Before torch / any HPU graph compile (see _gc_kernel_path): importing
# this module first must still make the Qwen4Exp TPC library visible to
# the graph compiler. Idempotent.
from ._gc_kernel_path import ensure_gc_kernel_path

ensure_gc_kernel_path()

import torch  # noqa: E402

# GDN custom-op inputs are split/chunk views in the caller
# (``mixed_qkvz.split`` -> ``mixed_qkv``, ``ba.chunk`` -> ``a``/``b``),
# exactly the view-input form that loses its tensor_data in lazy mode;
# ``_materialized`` gives them real storage (see _materialize).
from ._materialize import materialized as _materialized  # noqa: E402

_OP_NAMES = ("recur", "conv", "conv_rows", "gated_norm")
_OPS: dict | bool | None = None

# The H14/H25/H26 paths live in the GDN layer that qwen3_5.py shares with every
# Qwen3.5-family model (the 27B included), but they were measured on Qwen4Exp
# only. So an unset switch follows the model: Qwen4Exp opts in when it builds
# its text model (enable_for_qwen4exp); any other model keeps the previous GDN
# path. An explicit VLLM_QWEN4EXP_GDN_* value still wins either way.
_MODEL_DEFAULT = False


def enable_for_qwen4exp() -> None:
    """Make the unset H14/H25/H26 switches default to on (called by Qwen4Exp)."""
    global _MODEL_DEFAULT
    _MODEL_DEFAULT = True


def _switch(name: str) -> bool:
    value = os.environ.get(name, "")
    if value == "":
        return _MODEL_DEFAULT
    return value != "0"


def gdn_tpc_enabled() -> bool:
    return _switch("VLLM_QWEN4EXP_GDN_TPC")


def h25_enabled() -> bool:
    """H25 dispatch umbrella (kill switch ``VLLM_QWEN4EXP_GDN_H25=0``).

    Covers only *which* kernel runs: the padded-prefill conv unlock, the
    short-span recurrence gate, and the output-norm selection. Off restores
    the pre-H25 dispatch exactly (H14 alone, mask/shape rules unchanged).
    Unset: on for Qwen4Exp, off for other models (see ``_switch``).
    """
    return _switch("VLLM_QWEN4EXP_GDN_H25")


def _h25_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def h25_recur_max_tokens() -> int:
    """Longest prefill span that keeps the TPC recurrence.

    Measured on Gaudi2 against the *FlashQLA* chunk path (2026-11-05): TPC wins
    7.4x @16, 2.3x @32, 1.01x @64 and loses 0.46x @128 (against the pre-H25
    FP32 chunk it was 5.3x / 2.6x / 1.6x), so the gate is 64 tokens; H14 used
    16. The masked span form (see ``h25_masked_recur_enabled``) uses the same
    gate because it costs the same.
    """
    return _h25_int_env("VLLM_QWEN4EXP_GDN_H25_RECUR_MAX_TOKENS", 64)


def h25_masked_recur_enabled() -> bool:
    """Short-span TPC recurrence under the prefill padding mask (kill switch
    ``VLLM_QWEN4EXP_GDN_H25_MASKED_RECUR=0``).

    Padded rows are neutralized through the gating saturation (-inf), so the
    padded tail leaves the recurrent state untouched. Measured on Gaudi2
    against the masked chunk path (real Layer-0 weights): out |absmax| 7.6e-06,
    state 2.4e-04, padded rows exactly zero; 5.96x @T=16, 2.84x @T=32,
    1.22x @T=64.
    """
    return _switch("VLLM_QWEN4EXP_GDN_H25_MASKED_RECUR")


def h25_bf16_prefill_min_tokens() -> int:
    """Shortest prefill span whose chunk recurrence runs in BF16.

    The chunk path is latency-bound at small spans (BF16 is neutral there and
    costs ~1 BF16 ULP of accuracy), but MME-throughput-bound from ~1k tokens:
    measured per-layer T=2048 8.49 -> 5.25 ms, T=512 2.63 -> 2.78 ms.

    H26 sets the default to the measured crossover (1024 tokens) and pairs the
    BF16 bulk products with the FP32 solve/state boundary; see
    ``h26_bf16_boundary_enabled``. H25 Phase 2 had shipped this disabled
    (``1 << 30``) after long prompts collapsed in live ``vllm serve``, but that
    collapse was the padded-prefill ``conv_state`` wipe fixed in
    ``gdn_conv_prefill`` (H26 Part 1), not the chunk arithmetic: on real
    Layer-0 weights BF16 stays at |rel_l2| 6.7e-03 against the FP32 production
    configuration for every padded and unpadded shape.

    Set ``VLLM_QWEN4EXP_GDN_H25_BF16_MIN_TOKENS=0`` to force BF16 everywhere or
    a large value to keep the whole prefill on the FP32 chunk path.
    """
    return _h25_int_env("VLLM_QWEN4EXP_GDN_H25_BF16_MIN_TOKENS", 1024)


def h26_bf16_boundary_enabled() -> bool:
    """Keep the chunk triangular solve in FP32 under BF16 compute.

    The BF16 MME products (KKT, local attention, ``k^T u``, the state
    projections) dominate the chunk runtime; the unit-lower inverse is the one
    stage whose rounding is amplified by the solve products. With the exact
    blocked solver below the boundary costs 4.14 -> 4.69 ms/layer at T=2048
    (2.09x -> 1.84x against the FP32 production chunk), which keeps the whole
    recurrence on an exact FP32 solve.

    Kill switch: ``VLLM_QWEN4EXP_GDN_H26_BF16_BOUNDARY=0`` (all-BF16 solve).
    """
    return _switch("VLLM_QWEN4EXP_GDN_H26_BF16_BOUNDARY")


def h26_bf16_state_fp32() -> bool:
    """Also accumulate the chunk recurrent state in FP32 under BF16 compute.

    Off by default: measured per-layer T=2048 4.69 -> 6.35 ms for a 1-ULP
    state difference (1.95e-03 -> 9.77e-04 against a state whose absmax is
    ~1.5) that the BF16 ``ssm_state`` cache write-back discards anyway. Enable
    with ``VLLM_QWEN4EXP_GDN_H26_BF16_STATE_FP32=1`` when the state must not be
    rounded inside the prefill either.
    """
    return os.environ.get("VLLM_QWEN4EXP_GDN_H26_BF16_STATE_FP32", "0") != "0"


def h26_chunk_solver_base(chunk_size: int, current: int) -> int:
    """Solver base for the prefill chunk recurrence: exact inverse, 1.7x faster.

    ``recursive_solver_base=2`` replaces the 14 Neumann bmm iterations -- an
    *approximate* inverse (residual ~8 on Qwen3.5-9B per the solver docstring)
    -- with a blocked recursion that inverts the 2x2 diagonal blocks exactly
    and combines them through the Schur complement, i.e. the exact unit-lower
    inverse. Measured on real Layer-0 weights: 8.65 -> 4.96 ms/layer at T=2048,
    5.62 -> 3.47 at T=1280, 2.83 -> 2.93 at T=512, at |rel_l2| 2.3e-05 against
    the production Neumann path. It needs the chunk size to be a power-of-two
    multiple of the base, so any other geometry keeps ``current``.

    Kill switch: ``VLLM_QWEN4EXP_GDN_H26_SOLVER_BASE=0``.
    """
    base = _h25_int_env("VLLM_QWEN4EXP_GDN_H26_SOLVER_BASE", 2)
    if base < 2 or base & (base - 1) or chunk_size % base:
        return current
    ratio = chunk_size // base
    if ratio & (ratio - 1):
        return current
    return max(current, base)


def h25_norm_max_rows() -> int:
    """Largest row count where the Habana fused gated norm beats torch
    (measured: wins up to 24,576 rows = T 512, loses from 36,864 rows)."""
    return _h25_int_env("VLLM_QWEN4EXP_GDN_H25_NORM_MAX_ROWS", 24576)


def h25_tpc_norm_max_rows() -> int:
    """Largest row count where the TPC gated norm wins at all
    (measured: wins to ~768 rows, loses to the Habana fused norm from 3,072)."""
    return _h25_int_env("VLLM_QWEN4EXP_GDN_H25_TPC_NORM_MAX_ROWS", 1024)


def _gdn_ops() -> dict | None:
    global _OPS
    if not gdn_tpc_enabled():
        return None
    if _OPS is not None:
        return _OPS or None  # type: ignore[return-value]
    try:
        from .packed_moe import _ensure_dequant_op

        _ensure_dequant_op()  # the same extension .so registers the GDN ops
        namespace = torch.ops.custom_op
        ops = {name: getattr(namespace, f"custom_qwen4exp_gdn_{name}_bf16_gaudi2")
               for name in _OP_NAMES}
    except Exception:
        ops = False
    _OPS = ops
    return ops or None  # type: ignore[return-value]


def _safe_indices(indices: torch.Tensor, slots: int) -> torch.Tensor:
    """Map cache indices into [0, slots) (padding -1 -> the garbage slot),
    as an int32 contiguous tensor for the TPC ops."""
    idx = indices.reshape(-1)
    if idx.dtype != torch.int32:
        idx = idx.to(torch.int32)
    return torch.remainder(idx, slots).contiguous()


def merged_in_proj(layer, hidden: torch.Tensor):
    """One GEMM for in_proj_qkvz + in_proj_ba via a cached merged weight.

    The checkpoint keeps the two projections separate; the merged weight is
    a host-side concatenation cached on the module (invalidated by weight
    pointer/version), never a checkpoint rewrite. Returns
    (mixed_qkvz, ba) views of the single GEMM output, or None to fall back.
    """
    if not gdn_tpc_enabled() or hidden.device.type != "hpu":
        return None
    try:
        w_qkvz = layer.in_proj_qkvz.weight.detach()
        w_ba = layer.in_proj_ba.weight.detach()
    except Exception:
        return None
    if w_qkvz.dtype != hidden.dtype or w_ba.dtype != hidden.dtype:
        return None
    if getattr(layer, "tp_size", 1) != 1:
        return None
    quant_method = getattr(layer.in_proj_qkvz, "quant_method", None)
    if quant_method is not None and type(quant_method).__name__ != "UnquantizedLinearMethod":
        return None
    key = (w_qkvz.data_ptr(), w_qkvz._version, w_ba.data_ptr(), w_ba._version,
           w_qkvz.shape, w_ba.shape)
    merged = getattr(layer, "_h14_merged_weight", None)
    from .merged_weights import views_of

    if merged is not None and views_of(merged, w_qkvz, w_ba):
        pass  # the parameters are views of the cached concatenation: it is current wherever it lives
    elif getattr(layer, "_h14_merged_key", None) != key or merged is None:
        from .merged_weights import merge_rows

        merged = merge_rows(layer.in_proj_qkvz.weight, layer.in_proj_ba.weight)
        if layer.in_proj_qkvz.weight.data_ptr() == merged.data_ptr():
            # H31: the parameters are now views of ``merged``; key the cache on them.
            w_qkvz = layer.in_proj_qkvz.weight.detach()
            w_ba = layer.in_proj_ba.weight.detach()
            key = (w_qkvz.data_ptr(), w_qkvz._version, w_ba.data_ptr(), w_ba._version,
                   w_qkvz.shape, w_ba.shape)
        layer._h14_merged_weight = merged
        layer._h14_merged_key = key
    out = torch.nn.functional.linear(hidden, merged)
    qkvz_rows = w_qkvz.shape[0]
    return out[:, :qkvz_rows], out[:, qkvz_rows:]


def conv_weight_t(layer) -> torch.Tensor:
    """Cached tap-major conv weight [4, C] (kernel reads channels on dim 0)."""
    w = layer.conv1d.weight.detach()
    key = (w.data_ptr(), w._version, w.shape)
    cached = getattr(layer, "_h14_conv_weight_t", None)
    if getattr(layer, "_h14_conv_weight_key", None) != key or cached is None:
        cached = w.view(w.size(0), w.size(2)).t().contiguous()
        layer._h14_conv_weight_t = cached
        layer._h14_conv_weight_key = key
    return cached


def gdn_conv_decode(mixed_qkv, conv_pool, weight_t, indices):
    """Decode conv: returns (activated [B,C], next_state [B,3,C]) or None."""
    if mixed_qkv.device.type != "hpu":
        return None
    ops = _gdn_ops()
    if ops is None:
        return None
    if (mixed_qkv.dtype != torch.bfloat16 or conv_pool.dtype != torch.bfloat16
            or conv_pool.dim() != 3 or conv_pool.shape[1] != 3
            or mixed_qkv.shape[0] != indices.reshape(-1).shape[0]):
        return None
    safe = _safe_indices(indices, conv_pool.shape[0])
    # detach: a state pool written under grad mode (e.g. a priming prefill
    # outside no_grad) acquires requires_grad; the TPC op is inference-only.
    # detach shares storage, and contiguous() is a no-op for the usual pool.
    return ops["conv"](_materialized(mixed_qkv), conv_pool.detach().contiguous(), weight_t, safe)


def gdn_conv_prefill(mixed_qkv, conv_pool, weight_t, indices, has_initial, state_stops=None):
    """Prefill conv over uniform sequences: (activated [T,C], next [S,3,C]).

    ``state_stops`` is the runner's ``query_start_loc`` ([S + 1] I32, the
    cumsum of the **actual** query lengths) for a bucketed prefill. The
    kernel publishes ``next_state`` at the last *real* token of each
    sequence -- ``hpu_causal_conv1d_fn`` gathers it at
    ``(qsl[1:] - qsl[:-1]) + [0, 1, 2]`` inside ``cat(init_state, x)``, not
    at the padded end of the slot -- so a padded span needs it to avoid
    overwriting the state with the zeroed tail. ``None`` means every
    sequence fills its slot (the unpadded offline shape).
    """
    if mixed_qkv.device.type != "hpu":
        return None
    ops = _gdn_ops()
    if ops is None:
        return None
    if (mixed_qkv.dtype != torch.bfloat16 or conv_pool.dtype != torch.bfloat16
            or conv_pool.dim() != 3 or conv_pool.shape[1] != 3):
        return None
    num_seqs = indices.reshape(-1).shape[0]
    if num_seqs == 0 or mixed_qkv.shape[0] % num_seqs:
        return None
    safe = _safe_indices(indices, conv_pool.shape[0])
    if has_initial is None:
        has_i32 = torch.zeros(num_seqs, dtype=torch.int32, device=mixed_qkv.device)
    else:
        has_i32 = _materialized(has_initial.reshape(-1).to(torch.int32))
        if has_i32.shape[0] != num_seqs:
            return None
    if state_stops is None:
        tokens_per_seq = mixed_qkv.shape[0] // num_seqs
        stops = torch.arange(0,
                             (num_seqs + 1) * tokens_per_seq,
                             tokens_per_seq,
                             dtype=torch.int32,
                             device=mixed_qkv.device)
    else:
        stops = state_stops.reshape(-1)
        if stops.numel() != num_seqs + 1 or stops.device.type != "hpu":
            return None
        if stops.dtype != torch.int32:
            stops = stops.to(torch.int32)
    # ``query_start_loc`` arrives as a graph input in lazy mode (no
    # ``tensor_data`` of its own), which the custom op rejects at compile time;
    # free in eager mode (contiguous() only), one tiny copy otherwise.
    stops = _materialized(stops)
    return ops["conv_rows"](_materialized(mixed_qkv), conv_pool.detach().contiguous(),
                            weight_t, safe, has_i32, stops)


def gdn_recur(mixed_qkv, a, b, a_log, dt_bias, state_pool, indices, token_mask=None):
    """Fused recurrence: returns (out [T,Hv,128], state_out [S,Hv,128,128]).

    ``mixed_qkv`` is the conv-activated packed [q|k|v] tensor; sequences are
    uniform (T = S * tokens_per_seq consecutive rows per sequence).

    ``token_mask`` ([T, 1], 1.0 real / 0.0 padding) makes a *bucketed* prefill
    span usable: the kernel derives the gating from ``a``/``b`` internally, so
    padded rows are neutralized by driving both to -inf, which is the exact
    limit the gating saturates at (``g = -exp(A_log) * softplus(-inf) = 0``,
    ``beta = sigmoid(-inf) = 0``). A zero g means no decay and a zero beta means
    no state write, so the recurrent state after the padded tail equals the
    state after the last real token, while the padded rows' outputs are zeroed
    by the caller's mask on the conv output.
    """
    if mixed_qkv.device.type != "hpu":
        return None
    ops = _gdn_ops()
    if ops is None:
        return None
    if (mixed_qkv.dtype != torch.bfloat16 or state_pool.dtype != torch.bfloat16
            or state_pool.dim() != 4 or state_pool.shape[2:] != (128, 128)
            or a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16):
        return None
    if token_mask is not None:
        keep = _materialized(token_mask.reshape(-1, 1).to(torch.bool))
        if keep.shape[0] != a.shape[0]:
            return None
        neg = torch.full((), float("-inf"), dtype=torch.bfloat16, device=a.device)
        a = torch.where(keep, _materialized(a), neg)
        b = torch.where(keep, _materialized(b), neg)
    num_seqs = indices.reshape(-1).shape[0]
    if num_seqs == 0 or mixed_qkv.shape[0] % num_seqs:
        return None
    safe = _safe_indices(indices, state_pool.shape[0])
    a_log_f32 = a_log.detach()
    if a_log_f32.dtype != torch.float32:
        a_log_f32 = a_log_f32.float()
    out, state_out = ops["recur"](_materialized(mixed_qkv), _materialized(a), _materialized(b),
                                  a_log_f32.contiguous(), dt_bias.detach().contiguous(),
                                  state_pool.detach().contiguous(), safe)
    return out.to(torch.bfloat16), state_out


def gdn_gated_norm(x, z, weight, eps: float, activation: str):
    """rmsnorm(x) * act(z) for rows of width 128; act: sigmoid or silu."""
    if x.device.type != "hpu":
        return None
    ops = _gdn_ops()
    if ops is None:
        return None
    if activation not in ("sigmoid", "silu", "swish"):
        return None
    if x.dtype != torch.bfloat16 or z.dtype != torch.bfloat16 or x.shape[-1] != 128:
        return None
    # The TPC norm is launch/latency-bound and loses to the torch norm on
    # large row counts (measured at 24k rows); keep it for decode/short spans.
    # H25 narrows the gate to the band where it actually wins (<=1,024 rows)
    # because the Habana fused norm owns 3k-24k rows.
    max_rows = h25_tpc_norm_max_rows() if h25_enabled() else 4096
    if x.numel() // 128 > max_rows:
        return None
    act = 1 if activation in ("silu", "swish") else 0
    w = weight.detach()
    if w.dtype != torch.bfloat16:
        w = w.to(torch.bfloat16)
    return ops["gated_norm"](_materialized(x.reshape(-1, 128)), _materialized(z.reshape(-1, 128)),
                             w.contiguous(), float(eps), act)
