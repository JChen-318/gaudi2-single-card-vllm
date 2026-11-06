# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch reference runtime for the Qwen4Exp PLE layer.

Ports the math of upstream ``common/ops/ple.py`` (Triton kernels) and
``common/ngram_embedding.py`` to plain torch, preserving the kernels'
intermediate rounding boundaries (BF16 materialization at each tensor
boundary) so results match the GPU path bit-for-bit:

- ``compose_ngram_ids``: per-token hash
  ``offset[h] + (xor_i(token[t-i] * multiplier[i]) % size[h])`` with n-gram
  order ``h // heads_per_ngram + 2`` and EOS terminating older positions.
- ``ple_gate``: ``d = dot(RMSNorm_H(key), RMSNorm_H(query)) / sqrt(H)``,
  ``g = sigmoid(sign(d) * sqrt(max(|d|, 1e-6)))``, gated ``= g * value``,
  conv input ``= RMSNorm_H(gated)`` — all per HC stream of width H.
- ``ple_short_conv``: depthwise dilated conv over ``[state || chunk]`` with
  ``silu`` activation, accumulated into the gated output.

The layer keeps a per-request conv state ``[slots, C, window]`` (window =
``(kernel-1) * dilation``) so decode steps reuse the prefill tail.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ._materialize import in_compile_only_mode as _in_compile_only_mode
from ._materialize import materialized as _materialized
from .h24 import h24_enabled as _h24_enabled
from .h29 import h29_enabled as _h29_enabled
from .ple_embedding import PLEShardTable

__all__ = [
    "PLENGramEmbedding",
    "apply_ple",
    "apply_ple_fused",
    "compose_ngram_ids",
    "infer_request_spans",
    "load_ple_ngram",
    "load_ple_shard",
    "ple_gate",
    "ple_short_conv",
    "ple_short_conv_fused",
    "prepare_ple_inputs",
    "update_conv_state",
]


def infer_request_spans(positions: torch.Tensor) -> list[tuple[int, int]]:
    """Split a batch on position restarts.

    A new request starts where the logical position is 0 or moves backwards.
    A contiguous ``+1`` run stays one request (prefill or chunked prefill).
    A decode batch of unrelated positions (``[5, 9]``) is not a restart and
    must be split with an explicit ``query_start_loc``.
    """
    logical = positions if positions.ndim == 1 else positions[0]
    logical = logical.reshape(-1)
    n = int(logical.numel())
    if n == 0:
        return []
    values = [int(v) for v in logical.tolist()]
    starts = [0]
    for i in range(1, n):
        if values[i] == 0 or values[i] < values[i - 1]:
            starts.append(i)
    starts.append(n)
    return list(zip(starts[:-1], starts[1:], strict=True))


def _positions_are_contiguous(logical: torch.Tensor, start: int, end: int) -> bool:
    if end - start <= 1:
        return True
    chunk = logical[start:end]
    return bool(torch.equal(chunk[1:] - chunk[:-1], torch.ones_like(chunk[1:])))


def prepare_ple_inputs(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    query_start_loc: torch.Tensor | None,
    ngram_context: torch.Tensor | None,
    ngram_size: int,
    eos_token_id: int,
    history: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build PLE ``query_start_loc`` / ``ngram_context`` and extend ``history``.

    ``history[slot]`` is the token ids already consumed by that batch slot,
    not including the current chunk. Context is the last ``ngram_size - 1``
    of those ids, left-padded with EOS. A chunk whose first position is 0
    replaces the slot (new sequence). Batch-slot order must stay stable
    across steps; the worker's request-id map replaces this later.
    """
    ids = input_ids.reshape(-1)
    logical = positions if positions.ndim == 1 else positions[0]
    logical = logical.reshape(-1)
    if logical.numel() != ids.numel():
        raise ValueError(
            f"PLE positions ({logical.numel()}) and input_ids ({ids.numel()}) disagree"
        )
    if query_start_loc is not None and int(query_start_loc.reshape(-1)[-1]) != ids.numel():
        # Padded runner metadata does not describe this token tensor.
        query_start_loc = None
    if query_start_loc is None:
        spans = infer_request_spans(positions)
        if len(spans) == 1 and not _positions_are_contiguous(logical, 0, ids.numel()):
            raise RuntimeError(
                "PLE cannot infer request boundaries from positions; pass query_start_loc"
            )
        loc = [0] + [end for _, end in spans]
        query_start_loc = torch.tensor(loc, dtype=torch.int32)
    else:
        loc = [int(v) for v in query_start_loc.reshape(-1).tolist()]
        spans = list(zip(loc[:-1], loc[1:], strict=True))
    if ngram_context is not None:
        return query_start_loc, ngram_context

    ctx_len = int(ngram_size) - 1
    if ctx_len <= 0:
        raise ValueError("PLE ngram_size must be >= 2")
    context = torch.full((len(spans), ctx_len), int(eos_token_id), dtype=torch.int32)
    while len(history) < len(spans):
        history.append(torch.empty(0, dtype=torch.int64))
    for slot, (start, end) in enumerate(spans):
        if end <= start:
            continue
        # Trim -1 bucket padding: only the real prefix of the span enters
        # the n-gram history.
        valid = int((logical[start:end] >= 0).sum().item())
        if valid <= 0:
            continue
        end = start + valid
        restart = int(logical[start].item()) == 0
        prev = history[slot]
        if not restart and prev.numel():
            tail = prev[-ctx_len:].to(dtype=torch.int32)
            context[slot, ctx_len - tail.numel():] = tail
        chunk = ids[start:end].detach().to(device="cpu", dtype=torch.int64).contiguous()
        history[slot] = chunk if restart or prev.numel() == 0 else torch.cat((prev, chunk))
    return query_start_loc, context


def compose_ngram_ids(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_multipliers: torch.Tensor,
    head_vocab_sizes: torch.Tensor,
    head_offsets: torch.Tensor,
    eos_token_id: int,
    heads_per_ngram: int,
    to_device: bool = True,
) -> torch.Tensor:
    """Return ``[T, ctx_len * heads_per_ngram]`` int64 embedding row ids.

    ``ngram_context`` is ``[num_reqs, ctx_len]`` (previous tokens per request,
    EOS padded); ``query_start_loc`` is ``[num_reqs + 1]`` token offsets.

    The hash multiplies tokens by ~2^44 multipliers. Gaudi2 int64 storage
    round-trips, but int64 multiply is a 32-bit product, so the arithmetic
    runs on the host and only the finished ids are copied back.
    """
    input_ids = input_ids.reshape(-1)
    out_device = input_ids.device
    if out_device.type != "cpu":
        input_ids = input_ids.cpu()
        query_start_loc = query_start_loc.cpu()
        ngram_context = ngram_context.cpu()
        layer_multipliers = layer_multipliers.cpu()
        head_vocab_sizes = head_vocab_sizes.cpu()
        head_offsets = head_offsets.cpu()
    device = input_ids.device
    query_start_loc = query_start_loc.to(device)
    ngram_context = ngram_context.to(device)
    layer_multipliers = layer_multipliers.to(device)
    head_vocab_sizes = head_vocab_sizes.to(device)
    head_offsets = head_offsets.to(device)
    num_tokens = input_ids.shape[0]
    num_reqs = query_start_loc.numel() - 1
    ctx_len = ngram_context.shape[1]

    qsl = query_start_loc.to(torch.int64)
    # Request index per token: last boundary <= t.
    req = torch.searchsorted(qsl[1:], torch.arange(num_tokens, device=device), right=True)
    req = req.clamp_max(num_reqs - 1)
    chunk_pos = torch.arange(num_tokens, device=device) - qsl[req]

    multipliers = layer_multipliers.to(torch.int64)
    num_heads = ctx_len * heads_per_ngram
    head_ids = torch.arange(num_heads, device=device)
    ngram_order = head_ids // heads_per_ngram + 2  # 2 .. ngram_size

    mixed = input_ids.to(torch.int64)[:, None] * multipliers[0]
    crossed = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    for shift in range(1, ctx_len + 1):
        in_step = chunk_pos >= shift
        step_idx = (torch.arange(num_tokens, device=device) - shift).clamp_min(0)
        step_token = input_ids[step_idx].to(torch.int64)
        ctx_col = (ctx_len - shift + chunk_pos).clamp(0, ctx_len - 1)
        context_token = ngram_context.to(torch.int64)[req, ctx_col]
        candidate = torch.where(in_step, step_token, context_token)
        candidate = torch.where(crossed, torch.full_like(candidate, eos_token_id), candidate)
        crossed |= candidate == eos_token_id
        term = candidate[:, None] * multipliers[shift]
        apply = (ngram_order > shift)[None, :]
        mixed = torch.where(apply, mixed ^ term, mixed)

    sizes = head_vocab_sizes.to(torch.int64)[None, :]
    offsets = head_offsets.to(torch.int64)[None, :]
    ids = torch.remainder(mixed, sizes) + offsets
    if to_device and out_device.type != "cpu":
        ids = ids.to(out_device)
    return ids


def _grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, h: int
) -> torch.Tensor:
    """RMSNorm over the trailing ``h`` of each HC stream, Gemma (1 + w) affine."""
    xf = x.float().unflatten(-1, (x.shape[-1] // h, h))
    # Materialize the fp32 intermediates (lazy mode only; plain contiguous
    # tensors in eager). In one fused lazy segment the compiler drops these
    # rounding boundaries and the normed output flips by 1 ULP on enough
    # rows to flip the gate downstream; real storage restores them.
    variance = _materialized(xf.square().mean(dim=-1, keepdim=True))
    normed = _materialized(xf * torch.rsqrt(variance + eps))
    w = weight.float().unflatten(0, (x.shape[-1] // h, h))  # [HC, H]
    return _materialized((normed * (1.0 + w.unsqueeze(0))).flatten(-2))


# ---------------------------------------------------------------------------
# H24-2: the H13 grouped-norm TPC kernel in the PLE gate
# ---------------------------------------------------------------------------
#: ``custom_qwen4exp_hc_norm_bf16_gaudi2`` or ``False`` once the lookup failed.
_HC_NORM_OP: object | None = None
#: ``(weight, (data_ptr, _version, device, dtype))`` -> materialized ``[HC, H]``
#: gamma. The kernel takes the raw checkpoint ``w`` and adds ``1.0`` inside
#: (``y = x * rsqrt(mean(x^2)+eps) * (1 + gamma)``, see
#: ``kernels/qwen4exp_hc_norm_bf16_gaudi2.c``), so the 1-D parameter is passed
#: through unchanged, only reshaped. A *view* of the parameter must not reach a
#: custom op (lazy mode kills it with "Strided Params", the H13 trap), hence the
#: cached copy.
_GROUPED_GAMMA_CACHE: dict[int, tuple] = {}


def _hc_norm_op():
    """The H13 ``hc_norm`` custom op, or None (H24 off / not loadable)."""
    global _HC_NORM_OP
    if not _h24_enabled("norm"):
        return None
    if _HC_NORM_OP is not None:
        return _HC_NORM_OP or None
    try:
        from .packed_moe import _ensure_dequant_op

        _ensure_dequant_op()  # the same extension .so registers the HC ops
        op = torch.ops.custom_op.custom_qwen4exp_hc_norm_bf16_gaudi2
    except Exception:
        op = False
    _HC_NORM_OP = op
    return op or None


def _grouped_gamma(weight: torch.Tensor, groups: int, h: int) -> torch.Tensor | None:
    """Cached contiguous ``[groups, h]`` copy of the flat norm weight."""
    entry = _GROUPED_GAMMA_CACHE.get(id(weight))
    key = (weight.data_ptr(), weight._version, weight.device, weight.dtype)
    if entry is not None and entry[0] is weight and entry[1] == key:
        return entry[2]
    gamma = torch.empty((groups, h), device=weight.device, dtype=weight.dtype)
    gamma.copy_(weight.detach().view(groups, h))
    # Compile-only warmup never executes the copy, so the cached tensor would
    # have no data and poison the first real forward (the H13/Kimi-K25 guard).
    if not _in_compile_only_mode():
        if len(_GROUPED_GAMMA_CACHE) > 64:  # a test suite builds many layers
            _GROUPED_GAMMA_CACHE.clear()
        _GROUPED_GAMMA_CACHE[id(weight)] = (weight, key, gamma)
    return gamma


def grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, h: int
) -> torch.Tensor:
    """``_grouped_gemma_rmsnorm(...).to(x.dtype)``, fused on HPU.

    H24-2: this is the exact operation of the H13 ``hc_norm`` TPC kernel
    (grouped Gemma RMSNorm, plain F32 ``sum(x^2)`` accumulation, ``1 + w``
    affine inside the kernel), and the PLE gate calls it three times per
    forward on ``[T, hc * hidden]`` — ~8 eager launches each. The fused op is
    one launch. The only difference is the reduction order of the sum of
    squares (128-wide F32 accumulation vs torch's tree reduction), i.e. a
    1-ULP effect that the 2560-wide gate dot averages away. Anything unusual (CPU, non-BF16, ``h % 128``)
    takes the exact torch path.

    H29-1 subsumes the three gate norms into the fused gate op; this stays the
    path for ``conv_input``-style standalone calls and for the fallback.
    """
    dtype = x.dtype
    op = _hc_norm_op()
    if (
        op is not None
        and x.device.type == "hpu"
        and dtype == torch.bfloat16
        and x.dim() == 2
        and weight.dtype == torch.bfloat16
        and weight.device == x.device
        and weight.ndim == 1
        and h % 128 == 0
        and x.shape[-1] % h == 0
        and weight.numel() == x.shape[-1]
    ):
        gamma = _grouped_gamma(weight, x.shape[-1] // h, h)
        if gamma is not None:
            return op(_materialized(x), gamma, float(eps))
    return _materialized(_grouped_gemma_rmsnorm(x, weight, eps, h).to(dtype))


# ---------------------------------------------------------------------------
# H29: one launch for the whole gate, one for the whole short conv
# ---------------------------------------------------------------------------
#: ``custom_qwen4exp_ple_gate_bf16_gaudi2`` / ``False`` once the lookup failed.
_PLE_GATE_OP: object | None = None
#: ``custom_qwen4exp_ple_conv_bf16_gaudi2`` / ``False`` once the lookup failed.
_PLE_CONV_OP: object | None = None


def _ple_op(name: str):
    """Resolve one of the H29 PLE custom ops (``None`` if unavailable)."""
    try:
        from .packed_moe import _ensure_dequant_op

        _ensure_dequant_op()  # the same extension .so registers the PLE ops
        return getattr(torch.ops.custom_op, f"custom_qwen4exp_ple_{name}_bf16_gaudi2")
    except Exception:
        return None


def _ple_gate_op():
    """The H29 fused gate op, or None (H29 off / not loadable)."""
    global _PLE_GATE_OP
    if not _h29_enabled("gate"):
        return None
    if _PLE_GATE_OP is None:
        _PLE_GATE_OP = _ple_op("gate") or False
    return _PLE_GATE_OP or None


def _ple_conv_op():
    """The H29 fused short-conv op, or None (H29 off / not loadable)."""
    global _PLE_CONV_OP
    if not _h29_enabled("conv"):
        return None
    if _PLE_CONV_OP is None:
        _PLE_CONV_OP = _ple_op("conv") or False
    return _PLE_CONV_OP or None


def _ple_gate_fused(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(gated, conv_input)`` from one TPC launch, or None to take the torch path.

    The kernel is the same math the torch chain below performs, with the same
    BF16 materialization boundaries (see
    ``csrc/qwen4exp/kernels/qwen4exp_ple_gate_bf16_gaudi2.c``). It needs
    contiguous BF16 planes on the HPU, ``H % 128 == 0`` and the three gammas
    as ``[S, H]`` copies (the ``hc_norm`` strided-view contract, hence
    ``_grouped_gamma``).
    """
    op = _ple_gate_op()
    if op is None or key.device.type != "hpu" or key.dtype != torch.bfloat16:
        return None
    h = value.shape[-1]
    if (
        key.dim() != 2
        or value.dim() != 2
        or hidden.shape != key.shape
        or value.shape[0] != key.shape[0]
        or h % 128
        or key.shape[-1] % h
        or not key.numel()
    ):
        return None
    hc = hidden.shape[-1] // h
    gammas = tuple(_grouped_gamma(w, hc, h) for w in (norm_key_w, norm_query_w, norm_conv_w))
    if any(g is None for g in gammas):
        return None
    gated, conv_input = op(
        _materialized(key),
        _materialized(hidden),
        _materialized(value),
        gammas[0],
        gammas[1],
        gammas[2],
        float(eps),
    )
    return gated, conv_input


def ple_gate(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(gated, conv_input)`` for ``hidden`` [T, HC*H].

    ``key`` [T, HC*H], ``value`` [T, H]. Every intermediate tensor boundary is
    materialized in the input dtype, matching the fused kernel.
    """
    dtype = key.dtype
    h = value.shape[-1]
    hc = hidden.shape[-1] // h
    # H29-1: the whole chain below (two grouped norms + BF16 dot + sigmoid +
    # gated value + the conv-input norm) is ONE TPC launch on HPU.
    fused = _ple_gate_fused(key, value, hidden, norm_key_w, norm_query_w, norm_conv_w, eps)
    if fused is not None:
        return fused
    # Every ``.to(dtype)`` below is a documented BF16 materialization
    # boundary (see the module docstring), so in lazy mode each one is
    # materialized into real storage (``_materialized``; a plain contiguous
    # tensor in eager, hence eager values are unchanged). Fused into one
    # lazy segment, the compiler instead carries the fp32 intermediates
    # through the dot/sigmoid chain: the gate flips by 1 ULP on a tenth of
    # the (token, stream) pairs, and the conv norm amplifies each flipped
    # gated row into the conv state the next chunk reads.
    #
    # H24-2: the three norms go through ``grouped_gemma_rmsnorm`` (the fused
    # ``hc_norm`` TPC op on HPU, the exact torch chain elsewhere), and the two
    # elementwise products below are written in the input dtype: the exact
    # product/sum of two BF16 values is representable in F32, so ``a * b`` /
    # ``a + b`` round once from the exact result — the same single rounding the
    # explicit ``.float() ... .to(dtype)`` spelling performs, without the two
    # extra full-width passes. ``VLLM_QWEN4EXP_PLE_H24_CONV=0`` restores the
    # verbose spelling.
    key_n = grouped_gemma_rmsnorm(key, norm_key_w, eps, h)
    query_n = grouped_gemma_rmsnorm(hidden, norm_query_w, eps, h)
    if _h24_enabled("conv"):
        products = _materialized(key_n * query_n)
    else:
        products = _materialized((key_n.float() * query_n.float()).to(dtype))
    dot = _materialized(products.float().unflatten(-1, (hc, h)).sum(dim=-1))  # [T, HC]
    d = _materialized((dot / float(h) ** 0.5).to(dtype)).float()               # [T, HC]
    sign = torch.sign(d)
    magnitude = _materialized(torch.sqrt(torch.clamp(d.abs(), min=1e-6)).to(dtype))
    g = _materialized(torch.sigmoid(sign * magnitude.float()).to(dtype))      # [T, HC]
    if _h24_enabled("conv"):
        gated = _materialized((g[:, :, None] * value[:, None, :]).flatten(-2))
    else:
        gated = _materialized(
            (g[:, :, None] * value.float()[:, None, :]).flatten(-2).to(dtype))  # [T, HC*H]
    conv_input = grouped_gemma_rmsnorm(gated, norm_conv_w, eps, h)
    return gated, conv_input


#: ``conv_weights`` identity/version -> F32 ``[C, K]`` copy. The conv weights
#: never change after load, and the five span calls of one decode step would
#: otherwise each cast 10240 x K values.
_CONV_FP32_CACHE: dict[int, tuple] = {}


def _conv_fp32(weights: torch.Tensor) -> torch.Tensor:
    """Cached ``[C, K]`` F32 copy of the conv weights.

    Keyed on the *caller's* tensor (``conv1d.weight``, the same object every
    step), not on the ``reshape`` view ``ple_short_conv`` builds: a fresh view
    has a fresh ``id``, so a view-keyed cache would miss on every span and the
    160 KB cast would still run 5x per decode step.
    """
    entry = _CONV_FP32_CACHE.get(id(weights))
    key = (weights.data_ptr(), weights._version, weights.device, weights.dtype)
    if entry is not None and entry[0] is weights and entry[1] == key:
        return entry[2]
    out = weights.detach().float().reshape(weights.shape[0], -1)
    if not _in_compile_only_mode():
        if len(_CONV_FP32_CACHE) > 64:
            _CONV_FP32_CACHE.clear()
        _CONV_FP32_CACHE[id(weights)] = (weights, key, out)
    return out


def _conv_per_request(
    sequence: torch.Tensor,  # [state_len + length, C]  history || chunk
    weights: torch.Tensor,   # [C, K], k=0 is the oldest tap
    dilation: int,
    length: int,
    weights_f32: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal dilated taps matching the upstream kernel.

    Output position ``j`` (within the chunk) reads
    ``sequence[j + dilation * k]`` for ``k`` in ``0 .. K-1``. With history of
    length ``(K-1)*dilation`` prepended, tap ``K-1`` is the current token and
    tap ``0`` is ``(K-1)*dilation`` steps back.

    H24-2: tap ``k`` reads ``sequence[dilation*k : dilation*k + length]`` — a
    contiguous row slice of the ``[state_len + length, C]`` buffer. The previous
    spelling built ``arange(length) + dilation*k`` and ran an advanced-index
    gather per tap (3 extra launches and a materialized copy each); the slice is
    the identical rows with no kernel. The first tap initializes the F32
    accumulator instead of adding into a ``zeros`` buffer (adding to 0.0 is
    exact, so the result is unchanged), and the F32 weight copy is cached.
    """
    kernel = weights.shape[1]
    channels = weights.shape[0]
    w = weights_f32 if weights_f32 is not None else weights.float()
    if _h24_enabled("conv"):
        out = w[:, 0].unsqueeze(0) * sequence[0:length].float()
        for k in range(1, kernel):
            out += w[:, k].unsqueeze(0) * sequence[dilation * k : dilation * k + length].float()
        return out
    positions = torch.arange(length, device=sequence.device)
    out = torch.zeros(length, channels, dtype=torch.float32, device=sequence.device)
    for k in range(kernel):
        out += weights[:, k].float().unsqueeze(0) * sequence[positions + dilation * k].float()
    return out


def _host_ints(tensor: torch.Tensor | None) -> list[int] | None:
    """Python ints from a control tensor without a device round trip if possible."""
    if tensor is None:
        return None
    if tensor.device.type != "cpu":
        tensor = tensor.detach().to("cpu")
    return [int(v) for v in tensor.reshape(-1).tolist()]


def ple_short_conv(
    conv_input: torch.Tensor,
    query_start_loc: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_slots: torch.Tensor,
    *,
    dilation: int,
    has_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the dilated short conv and update the per-request state.

    ``conv_input`` [T, C]; ``conv_state`` [slots, window, C] (vLLM's default
    SD conv-state orientation, channel-fastest — the shape
    ``Qwen4ExpPLELayer.get_state_shape`` announces and the layout the fused
    H29 kernel reads) with window = ``(kernel-1)*dilation``; ``state_slots``
    [num_reqs] slot ids (given in request order); ``has_state`` [num_reqs]
    bool marks requests whose state holds real history (False on first
    prefill).

    Returns ``silu(conv)`` [T, C] in ``conv_input.dtype``.

    H29-2b: when every request of the call has the same token count and the
    slots are contiguous from the first one (100% of decode steps and every
    single-request prefill), the whole batch runs as one ``cat`` + ``K``
    slices instead of a Python loop per request: one state update for all
    slots, no ``torch.tensor(slots, device=...)`` H2D, no per-request
    ``transpose``/``zero_()``. The per-span loop stays for a mixed-length
    batch.
    """
    dtype = conv_input.dtype
    weights = conv_weights.reshape(conv_weights.shape[0], -1)  # [C, K]
    kernel = weights.shape[1]
    state_len = (kernel - 1) * dilation
    qsl = _host_ints(query_start_loc) or [0]
    reqs = [(r, qsl[r], qsl[r + 1]) for r in range(len(qsl) - 1) if qsl[r + 1] > qsl[r]]
    slots = _host_ints(state_slots) or []
    out = torch.zeros_like(conv_input, dtype=torch.float32)
    if not reqs:
        return out.to(dtype)
    streamlined = _h24_enabled("conv")
    weights_f32 = _conv_fp32(conv_weights) if streamlined else None
    length = reqs[0][2] - reqs[0][1]
    packed = (
        streamlined
        and len(reqs) == conv_input.shape[0] // max(length, 1)
        and conv_input.shape[0] == len(reqs) * length
        and all(hi - lo == length and lo == i * length for i, (_r, lo, hi) in enumerate(reqs))
        and len(slots) >= len(reqs)
        and all(slots[i] == slots[0] + i for i in range(len(reqs)))
        and (has_state is None or all(bool(v) for v in has_state.reshape(-1).tolist()[: len(reqs)]))
    )
    if packed:
        batch = len(reqs)
        base = slots[0]
        state = conv_state[base : base + batch]  # [B, window, C]
        chunk = conv_input[: batch * length].view(batch, length, conv_input.shape[1])
        # ``full`` materializes history || chunk for every slot at once, so the
        # state write below cannot alias a pending read.
        full = torch.cat([state, chunk], dim=1)  # [B, window + length, C]
        acc = weights_f32[:, 0].unsqueeze(0) * full[:, 0:length].float()
        for k in range(1, kernel):
            acc += weights_f32[:, k].unsqueeze(0) * full[
                :, dilation * k : dilation * k + length
            ].float()
        out[: batch * length] = acc.reshape(batch * length, conv_input.shape[1])
        conv_state[base : base + batch].copy_(full[:, length : length + state_len])
    else:
        for i, (r, lo, hi) in enumerate(reqs):
            slot = slots[i]
            if has_state is not None and not bool(has_state[r]):
                history = torch.zeros(
                    state_len, conv_input.shape[1], dtype=dtype, device=conv_input.device
                )
            else:
                history = conv_state[slot][:state_len]  # [state_len, C]
            sequence = torch.cat([history.to(dtype), conv_input[lo:hi]], dim=0)
            conv = _conv_per_request(sequence, weights, dilation, hi - lo, weights_f32)
            out[lo:hi] = conv
            # Newest state_len tokens, right-aligned, become the next history.
            tail = sequence[-state_len:]
            if streamlined and state_len and tail.shape[0] == state_len:
                # H24-2: ``sequence`` always holds ``state_len + (hi - lo)``
                # rows and ``hi > lo``, so ``tail`` covers the whole window and
                # the ``zero_()`` was dead work (it never survived the
                # assignment).
                conv_state[slot, -state_len:] = tail.to(conv_state.dtype)
            else:
                conv_state[slot].zero_()
                conv_state[slot, -tail.shape[0] :] = tail.to(conv_state.dtype)
    conv = out.to(dtype).float()
    activated = (conv * torch.sigmoid(conv)).to(dtype)
    return activated


#: ``conv_weights`` identity/version -> BF16 ``[K, C]`` tap-major transpose.
#: The H29 conv kernel takes the taps as its TPC dim 1, i.e. torch ``[K, C]``;
#: the transpose is built once per parameter object (as ``_conv_fp32``).
_CONV_T_CACHE: dict[int, tuple] = {}


def _conv_kernel_weight(weights: torch.Tensor) -> torch.Tensor:
    """Cached contiguous BF16 ``[K, C]`` tap-major transpose."""
    entry = _CONV_T_CACHE.get(id(weights))
    key = (weights.data_ptr(), weights._version, weights.device, weights.dtype)
    if entry is not None and entry[0] is weights and entry[1] == key:
        return entry[2]
    out = weights.detach().reshape(weights.shape[0], -1).t().contiguous()
    if not _in_compile_only_mode():
        if len(_CONV_T_CACHE) > 64:
            _CONV_T_CACHE.clear()
        _CONV_T_CACHE[id(weights)] = (weights, key, out)
    return out


def _spans_are_packed(spans: list[tuple[int, int, int]], rows: int, base_slot: int = 0) -> int:
    """Common token count of a packed span list, else 0.

    ``spans`` is ``[(slot, start, end)]``. "Packed" means every request has the
    same token count and the slots/rows are contiguous from ``base_slot`` / 0 —
    the shape the batched layer path and the H29 conv kernel require.
    """
    if not spans:
        return 0
    length = spans[0][2] - spans[0][1]
    if length <= 0 or rows != length * len(spans):
        return 0
    for i, (slot, start, end) in enumerate(spans):
        if end - start != length or start != i * length or slot != base_slot + i:
            return 0
    return length


def ple_short_conv_fused(
    conv_input: torch.Tensor,
    gated: torch.Tensor,
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    spans: list[tuple[int, int, int]],
    *,
    dilation: int,
    base_slot: int = 0,
) -> torch.Tensor | None:
    """Fused short conv + state roll + both residual adds, one TPC launch.

    Returns ``hidden + gated + silu(conv)`` for the ``len(spans) * length``
    rows the packed span list covers (and advances ``conv_state`` in place),
    or ``None`` when the call does not qualify for the kernel — non-HPU,
    non-BF16, a mixed-length batch, or H29-2 disabled. The caller then runs
    ``ple_short_conv`` and the adds itself.

    Bit-compatibility: the kernel rounds the F32 tap accumulator to BF16
    before SiLU and performs the two residual adds as separate BF16 adds,
    exactly as ``ple_short_conv`` + ``apply_ple`` do. It accumulates with FMA
    (the torch spelling rounds each product), which is a sub-F32-ULP
    difference that flips a BF16 rounding on ~0.01% of the elements;
    ``state`` comes out bit-identical.
    """
    op = _ple_conv_op()
    if op is None or conv_input.device.type != "hpu" or conv_input.dtype != torch.bfloat16:
        return None
    length = _spans_are_packed(spans, conv_input.shape[0], base_slot)
    if not length:
        return None
    rows = conv_input.shape[0]
    channels = conv_input.shape[1]
    if (
        channels % 128
        or gated.shape != conv_input.shape
        or hidden_states.shape != conv_input.shape
        or conv_state.dtype != torch.bfloat16
        or conv_state.device != conv_input.device
        or conv_state.dim() != 3
        or conv_state.shape[0] < len(spans) + base_slot
        or conv_state.shape[2] != channels
        or conv_state.shape[1] % (conv_weights.shape[-1] - 1)
        or conv_state.shape[1] // (conv_weights.shape[-1] - 1) != dilation
    ):
        return None
    state = conv_state[base_slot : base_slot + len(spans)]
    mixed, next_state = op(
        _materialized(conv_input),
        _materialized(state),
        _conv_kernel_weight(conv_weights),
        _materialized(gated),
        _materialized(hidden_states),
    )
    state.copy_(next_state)
    return mixed



def apply_ple(
    hidden_states: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_start_loc: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    state_slots: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    *,
    eps: float,
    dilation: int,
    has_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gate, dilated short-conv, and the two residual adds.

    Upstream materializes ``round(gated + silu(conv))`` before adding the
    outer multi-stream residual in fp32 and rounding once.

    H24-2 writes both sums as one BF16 add: the exact sum of two BF16 values
    rounds once in ``a + b`` exactly as it does through the F32 spelling (the
    F32 sum is exact except when the exponents differ by more than the F32
    mantissa, where both spellings still round to the same BF16). Measured
    against the verbose spelling on HPU: bit-identical.
    ``VLLM_QWEN4EXP_PLE_H24_CONV=0`` restores the F32 spelling.
    """
    dtype = hidden_states.dtype
    gated, conv_input = ple_gate(
        key, value, hidden_states, norm_key_w, norm_query_w, norm_conv_w, eps
    )
    activated = ple_short_conv(
        conv_input,
        query_start_loc,
        conv_state,
        conv_weight,
        state_slots,
        dilation=dilation,
        has_state=has_state,
    )
    if _h24_enabled("conv"):
        mixed = gated + activated
        return hidden_states + mixed
    mixed = (gated.float() + activated.float()).to(dtype)
    return (hidden_states.float() + mixed.float()).to(dtype)


def apply_ple_fused(
    hidden_states: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    spans: list[tuple[int, int, int]],
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    *,
    eps: float,
    dilation: int,
    base_slot: int = 0,
) -> torch.Tensor | None:
    """``apply_ple`` for a span list, with the H29 fusion when it applies.

    Returns the rows ``hidden + gated + silu(conv)`` (and advances
    ``conv_state``) for a packed span list whose rows tile ``hidden_states``
    from row 0 — two TPC launches for the whole batch. When the fused conv op
    cannot take this shape (or H29-2 is off) the *same* gate result is folded
    with the torch conv instead, so the gate is never computed twice.
    """
    gated, conv_input = ple_gate(
        key, value, hidden_states, norm_key_w, norm_query_w, norm_conv_w, eps
    )
    fused = ple_short_conv_fused(
        conv_input,
        gated,
        hidden_states,
        conv_state,
        conv_weight,
        spans,
        dilation=dilation,
        base_slot=base_slot,
    )
    if fused is not None:
        return fused
    if not spans or any(end <= start for _slot, start, end in spans):
        return None
    # Torch fallback: the gate above is reused, only the conv and the two
    # residual adds run here (one vectorized ``ple_short_conv`` for the batch).
    offsets = [0]
    for _slot, start, end in spans:
        offsets.append(offsets[-1] + (end - start))
    dtype = hidden_states.dtype
    activated = ple_short_conv(
        conv_input,
        torch.tensor(offsets, dtype=torch.int32),
        conv_state,
        conv_weight,
        torch.tensor([slot for slot, _a, _b in spans], dtype=torch.int64),
        dilation=dilation,
    )
    if _h24_enabled("conv"):
        return hidden_states + (gated + activated)
    mixed = (gated.float() + activated.float()).to(dtype)
    return (hidden_states.float() + mixed.float()).to(dtype)


def update_conv_state(
    conv_state: torch.Tensor, slot: int, chunk: torch.Tensor, dilation: int, kernel: int
) -> None:
    """Advance one request's state with ``chunk`` rows (no conv output)."""
    state_len = (kernel - 1) * dilation
    seq = torch.cat([conv_state[slot], chunk], dim=0)
    tail = seq[-state_len:]
    conv_state[slot].zero_()
    conv_state[slot, : tail.shape[0]] = tail


class PLENGramEmbedding:
    """16-head n-gram embedding tables (128 host shards) for one PLE layer.

    The full 51B-parameter table lives in **host RAM as packed NVFP4** --
    128 shards x 2,500,012 rows x 90 B per row (80 packed E2M1 bytes + 10
    F8_E4M3 block-scale bytes) = 320,001,536 rows = 26.82 GiB -- loaded by
    np.memmap-style streaming from the checkpoint shards without a full copy
    (``ple_embedding.safetensors_mmap``): only the pages a step touches are
    faulted in, and no 26.82 GiB second copy is ever allocated.

    The shard tables chunk ONE flat global row space: a global row id lands
    in shard ``id // shard_rows`` at row ``id % shard_rows``. The per-head
    vocab ranges (``head_offsets``/``head_vocab_sizes``) are contiguous
    sub-spans of that space and deliberately do NOT align with shard
    boundaries — heads 10-15 of the real checkpoint even span more rows than
    a uniform per-head shard count would hold, which is what proves the
    layout is flat. Rows past the last valid id are exporter padding and are
    never addressed.

    Per step (:meth:`lookup`): a CPU ``index_select`` gathers the needed
    3-gram rows (``tokens x 16`` rows of 90 B each, i.e. ``tokens x 16 x
    160`` values still packed), the gathered rows are staged in a small
    pinned buffer and moved with one async H2D, and the dequant runs **on
    the destination device** through the cached-LUT path
    (``modelopt_csf.dequant_nvfp4`` -- the same function the expert planes
    use). The device op is expressed purely as tensor ops on the gathered
    rows, so it is queued lazily by the device runtime; with a CPU device
    (or ``device=None``) the *identical* function dequantizes the host rows,
    which is the CPU twin the tests exercise. The table itself never becomes
    a device tensor.
    """

    def __init__(
        self,
        tables: list[PLEShardTable],
        layer_multipliers: torch.Tensor,
        head_vocab_sizes: torch.Tensor,
        head_offsets: torch.Tensor,
        *,
        heads_per_ngram: int,
        ngram_size: int,
        shard_rows: int,
    ) -> None:
        self.ctx_len = ngram_size - 1
        self.heads_per_ngram = heads_per_ngram
        self.num_heads = self.ctx_len * heads_per_ngram
        self.layer_multipliers = layer_multipliers
        self.head_vocab_sizes = head_vocab_sizes
        self.head_offsets = head_offsets
        self.shard_rows = shard_rows
        self.tables = tables
        self.head_dim = tables[0].embed_dim
        # One weight_scale_2 is shared by all 128 shards of the table.
        self.global_scale = tables[0].global_scale
        self._stage: tuple[torch.Tensor, torch.Tensor] | None = None

    #: One gathered row is 90 bytes: 80 packed E2M1 weights + 10 E4M3 scales.
    ROW_BYTES = 90

    def gather(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Host gather of the packed rows for ``ids`` ``[T, H]``.

        Returns ``(packed_rows [T*H, 80], scale_rows [T*H, 10])`` in flat
        token-major/head-fastest order, gathered per shard from host memory
        with the rows still packed NVFP4 (nothing decoded here).

        H24-1 (``VLLM_QWEN4EXP_PLE_H24_GATHER``) replaces the per-shard
        ``shard == s`` boolean-mask scan with one stable ``argsort`` of the
        shard ids plus contiguous per-shard segments: the mask path allocates
        and scans a full-width boolean mask, a ``nonzero`` index list and two
        masked scatter copies *per unique shard* (up to 128 of them, 8192 rows
        at prefill T=512), i.e. ~7 full-width passes per shard. The sorted path
        touches each row once. Measured 3.7-5.2x faster on the host;
        both paths read the same rows, so the result is
        bit-identical by construction.
        """
        if _h24_enabled("gather"):
            fast = self._gather_sorted(ids)
            if fast is not None:
                return fast
        return self._gather_masked(ids)

    def _gather_masked(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-H24 gather: one boolean-mask scan per unique shard."""
        flat = ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        shard = torch.div(flat, self.shard_rows, rounding_mode="floor")
        row = flat - shard * self.shard_rows
        packed = torch.empty(flat.numel(), self.head_dim // 2, dtype=torch.uint8)
        scales = torch.empty(flat.numel(), self.head_dim // 16, dtype=torch.uint8)
        for s in torch.unique(shard).tolist():
            if not 0 <= s < len(self.tables):
                raise IndexError(
                    f"PLE row id {int(flat[shard == s][0]):,} routes to shard {s}, "
                    f"but only {len(self.tables)} tables are loaded"
                )
            mask = shard == s
            rows_packed, rows_scales = self.tables[int(s)].lookup_packed(row[mask])
            packed[mask] = rows_packed
            scales[mask] = rows_scales
        return packed, scales

    def _numpy_planes(self) -> list | None:
        """``[(packed_view, scales_view)]`` numpy views of every host shard.

        Built once per attached table list: ``Tensor.numpy()`` is a zero-copy
        view for a CPU tensor, but it is a Python-level call, and 256 of them
        per lookup would cost more than the gather itself. ``self.tables`` is
        replaced wholesale by ``attach_ngram``/``load_ple_ngram``, never
        mutated in place, so its identity is the cache key (a strong reference
        is kept, so a recycled ``id`` cannot alias a stale entry).
        """
        cache = getattr(self, "_np_planes", None)
        if cache is not None and cache[0] is self.tables:
            return cache[1]
        try:
            planes = [(table.packed.numpy(), table.scales.numpy()) for table in self.tables]
        except Exception:  # pragma: no cover - non-CPU / non-contiguous table
            planes = None
        self._np_planes = (self.tables, planes)
        return planes

    def _gather_sorted(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """H24-1 gather: ``argsort`` the shard ids, copy one segment per shard.

        Returns ``None`` (caller falls back to the mask path) when the host
        tables cannot be addressed as numpy views — the mocks/tests that build
        ``PLEShardTable``-shaped objects with a custom ``lookup_packed``.
        """
        tables = self.tables
        if not tables or any(not hasattr(table, "packed") for table in tables):
            return None
        try:
            import numpy as np
        except Exception:  # pragma: no cover - numpy is a hard dependency of the loader
            return None
        flat = ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        count = int(flat.numel())
        if count == 0:
            return (torch.empty(0, self.head_dim // 2, dtype=torch.uint8),
                    torch.empty(0, self.head_dim // 16, dtype=torch.uint8))
        shard = torch.div(flat, self.shard_rows, rounding_mode="floor")
        row = flat - shard * self.shard_rows
        planes = self._numpy_planes()
        if planes is None:
            return None
        shard_np = shard.numpy()
        row_np = row.numpy()
        packed_np = np.empty((count, self.head_dim // 2), dtype=np.uint8)
        scales_np = np.empty((count, self.head_dim // 16), dtype=np.uint8)
        order = np.argsort(shard_np, kind="stable")
        shard_sorted = shard_np[order]
        row_sorted = row_np[order]
        bounds = np.flatnonzero(shard_sorted[1:] != shard_sorted[:-1]) + 1
        starts = np.concatenate(([0], bounds)).tolist()
        ends = np.concatenate((bounds, [count])).tolist()
        for lo, hi in zip(starts, ends):
            s = int(shard_sorted[lo])
            if not 0 <= s < len(planes):
                raise IndexError(
                    f"PLE row id {int(flat[order[lo]]):,} routes to shard {s}, "
                    f"but only {len(planes)} tables are loaded"
                )
            index = order[lo:hi]
            plane_packed, plane_scales = planes[s]
            packed_np[index] = plane_packed[row_sorted[lo:hi]]
            scales_np[index] = plane_scales[row_sorted[lo:hi]]
        return torch.from_numpy(packed_np), torch.from_numpy(scales_np)

    def _stage_rows(
        self, packed_rows: torch.Tensor, scale_rows: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage gathered rows in a pinned buffer and start an async H2D.

        The staging buffer is reused across steps and only grows; copying
        into pinned memory is what makes ``non_blocking=True`` on the device
        transfer safe (a pageable source would silently synchronize). The
        buffers hold one step's gathered rows (``tokens x 16`` rows of 90 B),
        never the table. Only ever called for a non-CPU ``device``; the CPU
        path skips staging entirely and dequantizes in place.
        """
        stage = self._stage
        if (stage is None or stage[0].numel() < packed_rows.numel()
                or stage[1].numel() < scale_rows.numel()):
            stage = (torch.empty(packed_rows.shape, dtype=torch.uint8).pin_memory(),
                     torch.empty(scale_rows.shape, dtype=torch.uint8).pin_memory())
            self._stage = stage
        pin_packed, pin_scales = stage
        pin_packed[: packed_rows.shape[0]].copy_(packed_rows)
        pin_scales[: scale_rows.shape[0]].copy_(scale_rows)
        rows = packed_rows.shape[0]
        dev_packed = pin_packed[:rows].to(device, non_blocking=True)
        dev_scales = pin_scales[:rows].to(device, non_blocking=True)
        return dev_packed, dev_scales

    def lookup(
        self,
        ids: torch.Tensor,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """``ids`` [T, num_heads] global row ids -> ``[T, num_heads*head_dim]``.

        Gather on the host (rows stay packed), one pinned async H2D of the
        gathered rows when the destination is a device, then the cached-LUT
        dequant on whatever device the rows now live on. Gaudi2 int64
        multiply is not exact and a partial boolean scatter is an invalid
        slice node, so the gather is assembled on the host and moved once.
        """
        packed_rows, scale_rows = self.gather(ids)
        target = None if device is None else torch.device(device)
        if target is not None and target.type != "cpu":
            packed_rows, scale_rows = self._stage_rows(packed_rows, scale_rows, target)
        out = PLEShardTable.dequant(packed_rows, scale_rows, self.global_scale, dtype=dtype)
        if target is not None and out.device != target:
            out = out.to(target)
        num_tokens = ids.shape[0]
        return out.view(num_tokens, self.num_heads * self.head_dim)


def ple_ckpt_prefix(layer_idx: int, language_prefix: str = "model.language_model") -> str:
    """Checkpoint prefix of one PLE layer's n-gram tables."""
    return f"{language_prefix}.layers.{layer_idx}.ple.ple_embedding"


def _ckpt_tensor(tensors_dir: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(tensors_dir / weight_map[name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def _resolve_shard(tensors_dir: Path, entry: str) -> Path:
    """Resolve an index entry; exports disagree on the ``tensors/`` prefix."""
    direct = tensors_dir / entry
    if direct.is_file():
        return direct
    sibling = tensors_dir.parent / entry
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(f"cannot resolve shard {entry} under {tensors_dir}")


def load_ple_shard(
    tensors_dir: Path,
    weight_map: dict[str, str],
    prefix: str,
    shard_id: int,
    *,
    global_scale: float = 1.0,
    pin_memory: bool = True,
    mmap: bool = True,
) -> PLEShardTable:
    """Load one host shard table (``shard_<id>.weight`` + scale plane).

    With ``mmap`` (default) the shard's two tensors come back as read-only
    ``np.memmap`` windows into the safetensors file -- the 128-shard real
    table (26.82 GiB) streams from the HF snapshot shards page by page and
    no full copy is ever materialized. ``pin_memory=True`` then makes exactly
    one pinned copy of the shard; leave it off for the purely file-backed
    production residency.
    """
    base = f"{prefix}.ngram_embedding.shard_{shard_id}"
    if mmap:
        from .ple_embedding import PLEShardTable as _Table, safetensors_mmap

        shard = _resolve_shard(tensors_dir, weight_map[base + ".weight"])
        scale_entry = weight_map[base + ".weight_scale"]
        if _resolve_shard(tensors_dir, scale_entry) != shard:
            raise ValueError(f"{base}: weight and scale planes live in different shards")
        return _Table(
            safetensors_mmap(shard, base + ".weight"),
            safetensors_mmap(shard, base + ".weight_scale"),
            global_scale=global_scale,
            pin_memory=pin_memory,
        )
    return PLEShardTable(
        _ckpt_tensor(tensors_dir, weight_map, base + ".weight"),
        _ckpt_tensor(tensors_dir, weight_map, base + ".weight_scale"),
        global_scale=global_scale,
        pin_memory=pin_memory,
    )


def load_ple_ngram(
    checkpoint_dir: str | Path,
    *,
    layer_idx: int,
    heads_per_ngram: int,
    ngram_size: int,
    language_prefix: str = "model.language_model",
    pin_memory: bool = True,
    mmap: bool = True,
) -> PLENGramEmbedding:
    """Load every shard table for one PLE layer (26.82 GiB, host-resident).

    Real-table load, end to end: the index is parsed once, the small metadata
    tensors (multipliers / head offsets / vocab sizes / the table's single
    ``weight_scale_2``) are read in full, and each of the 128 shards is
    np.memmap-streamed from its safetensors shard
    (:func:`ple_embedding.safetensors_mmap`) -- no second 26.82 GiB copy, no
    decode, nothing on the device. Pages fault in on first touch; with
    ``pin_memory=False`` the table stays purely file-backed.
    """
    root = Path(checkpoint_dir)
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    tensors_dir = root / "tensors"
    prefix = ple_ckpt_prefix(layer_idx, language_prefix)
    multipliers = _ckpt_tensor(tensors_dir, weight_map, prefix + ".layer_multipliers")
    offsets = _ckpt_tensor(tensors_dir, weight_map, prefix + ".ngram_heads_offsets")
    sizes = _ckpt_tensor(tensors_dir, weight_map, prefix + ".ngram_heads_vocab_sizes")
    scale_name = prefix + ".ngram_embedding.weight_scale_2"
    global_scale = (
        float(_ckpt_tensor(tensors_dir, weight_map, scale_name).reshape(-1)[0])
        if scale_name in weight_map
        else 1.0
    )
    shard_ids = sorted(
        int(name.split("shard_")[1].split(".")[0])
        for name in weight_map
        if name.startswith(prefix + ".ngram_embedding.shard_") and name.endswith(".weight")
        and "scale" not in name
    )
    tables = [
        load_ple_shard(
            tensors_dir, weight_map, prefix, shard_id,
            global_scale=global_scale, pin_memory=pin_memory, mmap=mmap,
        )
        for shard_id in shard_ids
    ]
    if not tables:
        raise FileNotFoundError(f"no PLE shards under {prefix}")
    return PLENGramEmbedding(
        tables,
        multipliers,
        sizes,
        offsets,
        heads_per_ngram=heads_per_ngram,
        ngram_size=ngram_size,
        shard_rows=tables[0].rows,
    )
