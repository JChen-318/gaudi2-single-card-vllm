# SPDX-License-Identifier: Apache-2.0
"""H30: run decode layers as compiled graphs instead of about fifty eager launches each.

Eager decode launched about 2,600 ops per token, each compiled and destroyed on the host, so the
card sat idle two thirds of the step. The kernels themselves were not the limit: a token needs
about 21 ms of device time.

``VLLM_QWEN4EXP_DECODE_GRAPH=1`` runs single-token decode through the tensor-only step functions
below. They call the same ops in the same order as the eager path, with every weight and state
pool passed as an argument, so all layers of a kind share one traced graph and one recipe:

* a GDN layer is one graph: delayed combine, GDN, MLP combine, MoE, and the state write-back;
* a QSA layer is a projection graph (combine, merged qkv/indexer GEMM, norms, rotations) and an
  output graph (gate, o_proj, MLP combine, MoE). Between them the software cache commit stays
  eager; above the token budget the block selection and gathered attention run as one more graph
  over the capacity buffers (``_qsa_select_attend``), and a completed block is pooled by
  ``_qsa_pool_block``;
* the PLE layer keeps its addition eager (host n-gram lookup); the rest of it is the GDN graph.

Compiled graphs round BF16 intermediates and order MME sums differently from eager, so the output
is not byte-identical to the eager path (about 1 ulp per layer).

The GDN state rows are written back inside the graph (``index_copy_`` on the pool inputs). The
prefill path keeps that write outside its graphs because HPU ``torch.compile`` dropped it there;
here the long-context recall gate (a fact 20k tokens back) passes with it, which a dropped write
could not. ``VLLM_QWEN4EXP_DECODE_GRAPH_STATE_INPLACE=0`` returns the rows and copies them eagerly.
"""
from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn.functional as F

from vllm_gaudi.extension.logger import logger as _hpu_logger

logger = _hpu_logger()

_ENV = "VLLM_QWEN4EXP_DECODE_GRAPH"
#: Diagnostic: also run the eager layer, log how far the graph result is from it, and keep the
#: eager result. Output stays the eager output, so a passing run proves only the logging path.
_VERIFY_ENV = "VLLM_QWEN4EXP_DECODE_GRAPH_VERIFY"
_COMPILE_ARGS: dict[str, Any] = {"backend": "hpu_backend", "fullgraph": True, "dynamic": False}


def raise_dynamo_cache_limits(limit: int = 64) -> None:
    """Every QSA key capacity and every prompt bucket is its own static graph; the default limit of 8
    would silently send the rest back to eager."""
    config = torch._dynamo.config
    for name in ("cache_size_limit", "recompile_limit"):
        if hasattr(config, name):
            setattr(config, name, max(limit, getattr(config, name)))
    for name in ("accumulated_cache_size_limit", "accumulated_recompile_limit"):
        if hasattr(config, name):
            setattr(config, name, max(limit * 8, getattr(config, name)))


def enabled() -> bool:
    return os.environ.get(_ENV, "0") not in ("", "0")


def _diff(name: str, graph: torch.Tensor, eager: torch.Tensor) -> str:
    if graph.shape != eager.shape:
        return f"{name}: shape {tuple(graph.shape)} vs {tuple(eager.shape)}"
    graph_host, eager_host = graph.float().cpu(), eager.float().cpu()
    delta = (graph_host - eager_host).abs()
    return (f"{name}: differ={int((graph_host != eager_host).sum())}/{delta.numel()} "
            f"max={float(delta.max()):.3e} ref_abs_max={float(eager_host.abs().max()):.3e}")


def _hc_project(xn, down_w, up_w, lowrank: int, streams: int):
    ops = torch.ops.custom_op
    merged = F.linear(xn, down_w)
    low, injection = ops.custom_qwen4exp_hc_down_epilogue_bf16_gaudi2(merged, lowrank, streams)
    gate = F.linear(low, up_w)
    block_input = ops.custom_qwen4exp_hc_gate_mix_bf16_gaudi2(xn, gate, streams)
    return block_input, injection


def _hc_combine_and_mix(hidden, block, injection, gamma, eps: float, down_w, up_w, lowrank: int, streams: int):
    ops = torch.ops.custom_op
    hidden, xn = ops.custom_qwen4exp_hc_combine_norm_bf16_gaudi2(hidden, block, injection, gamma, eps)
    block_input, next_injection = _hc_project(xn, down_w, up_w, lowrank, streams)
    return hidden, block_input, next_injection


def _hc_mix(hidden, gamma, eps: float, down_w, up_w, lowrank: int, streams: int):
    xn = torch.ops.custom_op.custom_qwen4exp_hc_norm_bf16_gaudi2(hidden, gamma, eps)
    block_input, next_injection = _hc_project(xn, down_w, up_w, lowrank, streams)
    return hidden, block_input, next_injection


def k4_gemv_enabled() -> bool:
    """H31 k4 decode GEMVs (``VLLM_QWEN4EXP_MOE_GEMV_K4``, default 1) when the extension has them."""
    cached = _K4_STATE.get("enabled")
    if cached is None:
        ops = torch.ops.custom_op
        cached = (os.environ.get("VLLM_QWEN4EXP_MOE_GEMV_K4", "1") != "0"
                  and hasattr(ops, "custom_qwen4exp_nvfp4_indexed_swiglu_gemv_k4_bf16_gaudi2")
                  and hasattr(ops, "custom_qwen4exp_nvfp4_indexed_gemv_k4_bf16_gaudi2"))
        _K4_STATE["enabled"] = cached
    return cached


_K4_STATE: dict[str, bool] = {}


def _k4_layout(x):
    """Each whole 512-value K tile in nibble-major order (x_k4[t*512 + j*128 + i] = x[t*512 + 4i + j]);
    a trailing partial tile stays as it is. The k4 kernels read the packed weights in this order."""
    tokens, width = x.shape
    full = width // 512 * 512
    head = x[:, :full].reshape(tokens, full // 512, 128, 4).transpose(-1, -2).reshape(tokens, full)
    if full == width:
        return head.contiguous()
    return torch.cat([head, x[:, full:]], dim=1)


from .packed_moe import rowperm_active, rowperm_inv  # noqa: E402


def _moe(hidden, gate_w, top_k: int, ids_shape: int,
         gate_packed, gate_scale, gate_scale2, up_packed, up_scale, up_scale2,
         down_packed, down_scale, down_scale2,
         shared_merged, shared_down):
    ops = torch.ops.custom_op
    logits = F.linear(hidden, gate_w)
    ids, weights = ops.custom_qwen4exp_moe_router_topk_bf16_gaudi2(logits, top_k)
    flat_ids = ids.reshape(ids_shape).to(torch.int32)
    if k4_gemv_enabled() and hidden.shape[-1] % 512 == 0 and hidden.shape[-1] <= 2560:
        # H31: 21 instead of 33 vector instructions per row and 512-value tile (device time per
        # layer: swiglu 113.8 -> 89.6 us, down 109.7 -> 100.4 us at 10 experts; same output bits on
        # the synthetic planes of tools/qwen4exp_flashnext/kernel_ab/bench_k4_h31.py). Then one epilogue for the
        # four rows of a member and the activation row in TPC local memory (K <= 2560): swiglu
        # 89.2 -> 59.5 us, down 87.1 -> 44.0 us, bits unchanged (tools/qwen4exp_flashnext/kernel_ab/bench_pf.py).
        activated = ops.custom_qwen4exp_nvfp4_indexed_swiglu_gemv_k4_bf16_gaudi2(
            flat_ids, gate_packed, gate_scale, gate_scale2, up_packed, up_scale, up_scale2,
            _k4_layout(hidden))
        # H31 (VLLM_QWEN4EXP_MOE_ROWPERM): with gate/up rows stored in the k4 order the SwiGLU
        # output already is the down input; otherwise it is permuted here (a DMA transpose).
        down_in = activated if rowperm_active() else _k4_layout(activated)
        slot_down = ops.custom_qwen4exp_nvfp4_indexed_gemv_k4_bf16_gaudi2(
            flat_ids, down_packed, down_scale, down_in, down_scale2)
    else:
        activated = ops.custom_qwen4exp_nvfp4_indexed_swiglu_gemv_bf16_gaudi2(
            flat_ids, gate_packed, gate_scale, gate_scale2, up_packed, up_scale, up_scale2, hidden)
        if rowperm_active():
            activated = activated.index_select(-1, rowperm_inv())
        slot_down = ops.custom_qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2(
            flat_ids, down_packed, down_scale, activated, down_scale2)
    routed = ops.custom_qwen4exp_moe_down_epilogue_bf16_gaudi2(slot_down, weights.to(slot_down.dtype))
    shared = ops.custom_qwen4exp_moe_shared_swiglu_bf16_gaudi2(F.linear(hidden, shared_merged))
    return routed + F.linear(shared, shared_down)


def _gdn(block_input, in_merged, qkvz_rows: int, qkv_size: int, z_size: int, head_v_dim: int,
         conv_pool, conv_weight_t, a_log, dt_bias, ssm_pool, safe_index,
         norm_w, norm_eps: float, norm_act: int, out_w):
    ops = torch.ops.custom_op
    out = F.linear(block_input, in_merged)
    mixed_qkvz, ba = out[:, :qkvz_rows], out[:, qkvz_rows:]
    mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
    b, a = ba.chunk(2, dim=-1)
    activated, next_conv = ops.custom_qwen4exp_gdn_conv_bf16_gaudi2(mixed_qkv, conv_pool, conv_weight_t, safe_index)
    core, next_ssm = ops.custom_qwen4exp_gdn_recur_bf16_gaudi2(activated, a, b, a_log, dt_bias, ssm_pool, safe_index)
    core = core.to(torch.bfloat16)
    tokens = core.shape[0]
    normed = ops.custom_qwen4exp_gdn_gated_norm_bf16_gaudi2(
        core.reshape(-1, 128), z.reshape(-1, 128), norm_w, norm_eps, norm_act)
    attn_out = F.linear(normed.reshape(tokens, -1, head_v_dim).flatten(-2), out_w)
    return attn_out, next_conv, next_ssm


def _gdn_layer_step(hidden, block, injection,
                    a_gamma, a_down, a_up, m_gamma, m_down, m_up, hc_eps: float, lowrank: int, streams: int,
                    in_merged, qkvz_rows: int, qkv_size: int, z_size: int, head_v_dim: int,
                    conv_pool, conv_weight_t, a_log, dt_bias, ssm_pool, safe_index,
                    norm_w, norm_eps: float, norm_act: int, out_w,
                    gate_w, top_k: int, ids_shape: int,
                    gate_packed, gate_scale, gate_scale2, up_packed, up_scale, up_scale2,
                    down_packed, down_scale, down_scale2, shared_merged, shared_down):
    """One whole GDN decoder layer: delayed combine, attention, MLP combine, MoE."""
    hidden, block_input, next_injection = _hc_combine_and_mix(
        hidden, block, injection, a_gamma, hc_eps, a_down, a_up, lowrank, streams)
    attn_out, next_conv, next_ssm = _gdn(
        block_input, in_merged, qkvz_rows, qkv_size, z_size, head_v_dim,
        conv_pool, conv_weight_t, a_log, dt_bias, ssm_pool, safe_index,
        norm_w, norm_eps, norm_act, out_w)
    hidden, block_input, next_injection = _hc_combine_and_mix(
        hidden, attn_out, next_injection, m_gamma, hc_eps, m_down, m_up, lowrank, streams)
    mlp_out = _moe(block_input, gate_w, top_k, ids_shape,
                   gate_packed, gate_scale, gate_scale2, up_packed, up_scale, up_scale2,
                   down_packed, down_scale, down_scale2, shared_merged, shared_down)
    return hidden, mlp_out, next_injection, next_conv, next_ssm


def _gdn_first_layer_step(hidden,
                          a_gamma, a_down, a_up, m_gamma, m_down, m_up, hc_eps: float, lowrank: int, streams: int,
                          in_merged, qkvz_rows: int, qkv_size: int, z_size: int, head_v_dim: int,
                          conv_pool, conv_weight_t, a_log, dt_bias, ssm_pool, safe_index,
                          norm_w, norm_eps: float, norm_act: int, out_w,
                          gate_w, top_k: int, ids_shape: int,
                          gate_packed, gate_scale, gate_scale2, up_packed, up_scale, up_scale2,
                          down_packed, down_scale, down_scale2, shared_merged, shared_down):
    """A GDN layer with no pending combine: layer 0, and the layer after the PLE addition."""
    hidden, block_input, next_injection = _hc_mix(hidden, a_gamma, hc_eps, a_down, a_up, lowrank, streams)
    attn_out, next_conv, next_ssm = _gdn(
        block_input, in_merged, qkvz_rows, qkv_size, z_size, head_v_dim,
        conv_pool, conv_weight_t, a_log, dt_bias, ssm_pool, safe_index,
        norm_w, norm_eps, norm_act, out_w)
    hidden, block_input, next_injection = _hc_combine_and_mix(
        hidden, attn_out, next_injection, m_gamma, hc_eps, m_down, m_up, lowrank, streams)
    mlp_out = _moe(block_input, gate_w, top_k, ids_shape,
                   gate_packed, gate_scale, gate_scale2, up_packed, up_scale, up_scale2,
                   down_packed, down_scale, down_scale2, shared_merged, shared_down)
    return hidden, mlp_out, next_injection, next_conv, next_ssm


def _gdn_layer_step_inplace(hidden, block, injection, conv_at: int, ssm_at: int, safe_long, *args):
    """``_gdn_layer_step`` that also writes the new state rows into the pools inside the graph.

    H31: the pools are the graph inputs ``args[conv_at]`` / ``args[ssm_at]`` themselves, read and
    then written. They used to arrive twice (a detached alias to read, the pool to write): with
    that aliasing a request's prefill could come out differently depending on what the previous
    request decoded, which an out-of-graph write did not show.
    """
    conv_pool, ssm_pool = args[conv_at], args[ssm_at]
    hidden, mlp_out, next_injection, next_conv, next_ssm = _gdn_layer_step(
        hidden, block, injection, *args)
    conv_pool.index_copy_(0, safe_long, next_conv)
    ssm_pool.index_copy_(0, safe_long, next_ssm)
    return hidden, mlp_out, next_injection


def _gdn_first_layer_step_inplace(hidden, conv_at: int, ssm_at: int, safe_long, *args):
    conv_pool, ssm_pool = args[conv_at], args[ssm_at]
    hidden, mlp_out, next_injection, next_conv, next_ssm = _gdn_first_layer_step(hidden, *args)
    conv_pool.index_copy_(0, safe_long, next_conv)
    ssm_pool.index_copy_(0, safe_long, next_ssm)
    return hidden, mlp_out, next_injection


def _qsa_pre_step(hidden, block, injection, gamma, hc_eps: float, down_w, up_w, lowrank: int, streams: int,
                  merged_w, qkv_rows: int, q_width: int, kv_size: int, heads: int, kv_heads: int, head_dim: int,
                  gated: bool, q_norm_w, q_eps: float, k_norm_w, k_eps: float, cos_sin, head_size: int,
                  rotary_dim: int, neox: bool, rope_fp32: bool,
                  index_q_w, index_eps: float, index_heads: int, index_kv_heads: int, index_dim: int):
    """Delayed combine, then the merged qkv + indexer GEMM with its norms and rotations."""
    hidden, block_input, next_injection = _hc_combine_and_mix(
        hidden, block, injection, gamma, hc_eps, down_w, up_w, lowrank, streams)
    tokens = block_input.shape[0]
    projected = F.linear(block_input, merged_w)
    qkv = projected[:, :qkv_rows]
    index_qk = projected[:, qkv_rows:]

    q_block = qkv[:, 0:q_width]
    if gated:
        q_view = q_block.view(tokens, heads, 2 * head_dim)[..., :head_dim]
    else:
        q_view = q_block.view(tokens, heads, head_dim)
    k_view = qkv[:, q_width:q_width + kv_size].view(tokens, kv_heads, head_dim)
    q_normed = torch.ops.hpu.rms_norm(q_view, q_norm_w, q_eps, None, True)[0]
    k_normed = torch.ops.hpu.rms_norm(k_view, k_norm_w, k_eps, None, True)[0]
    q_gate, _k, v = qkv.split([q_width, kv_size, kv_size], dim=-1)
    gate = q_gate.view(tokens, heads, 2 * head_dim)[..., head_dim:].reshape(tokens, -1) if gated else q_gate
    q = _rotate(q_normed.reshape(tokens, heads * head_dim), cos_sin, head_size, rotary_dim, neox, rope_fp32)
    k = _rotate(k_normed.reshape(tokens, kv_heads * head_dim), cos_sin, head_size, rotary_dim, neox, rope_fp32)

    q_raw, token_k = index_qk.split((index_heads * index_dim, index_kv_heads * index_dim), dim=-1)
    xf = q_raw.view(-1, index_heads, index_dim).float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    q_index = (xf * torch.rsqrt(variance + index_eps) * (1.0 + index_q_w.float())).to(q_raw.dtype)
    return (hidden, next_injection, q.view(tokens, heads, head_dim), k.view(tokens, kv_heads, head_dim),
            v.view(tokens, kv_heads, head_dim), gate, _partial_rope(q_index, cos_sin),
            token_k.reshape(-1, index_dim))


def _rotate(x, cos_sin, head_size: int, rotary_dim: int, neox: bool, fp32: bool):
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    cos, sin = cos_sin.chunk(2, dim=-1)
    shape = x.shape
    x = x.view(shape[0], -1, head_size)
    rotated = ApplyRotaryEmb.forward_static(x[..., :rotary_dim], cos, sin, neox, fp32)
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1).reshape(shape)


def _partial_rope(x, cos_sin):
    rotary_dim = cos_sin.shape[-1]
    cos, sin = cos_sin.chunk(2, dim=-1)
    rotated = x[..., :rotary_dim]
    while cos.ndim < rotated.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    x1, x2 = rotated.chunk(2, dim=-1)
    rope = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return torch.cat((rope, x[..., rotary_dim:]), dim=-1)


def _qsa_pool_block(block_keys, k_norm_w, eps: float, cos_sin):
    """``pool_indexer_blocks`` for the one block a decode step completes."""
    pooled = block_keys.float().mean(dim=1).to(block_keys.dtype)
    xf = pooled.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    normed = (xf * torch.rsqrt(variance + eps) * (1.0 + k_norm_w.float())).to(pooled.dtype)
    return _partial_rope(normed, cos_sin)


def _qsa_select_attend(q, q_idx, pooled_buf, position, key_buf, value_buf, scale: float, ratio: int, k: int):
    """Block selection plus gathered attention for one decode query above the token budget.

    ``select_qsa_indices_from_pooled`` and ``sparse_attention`` read the exact cache prefixes, whose
    lengths change every token. Here the pooled rows and the keys are the capacity buffers and the
    query position is a tensor, so the graph only changes when a buffer grows. Blocks at or past the
    query's complete-block count score -inf and are never picked, which reproduces the unpadded
    selection: in decode every complete block is visible and ``k`` never exceeds their count.
    """
    index_dim = q_idx.shape[-1]
    dots = torch.matmul(q_idx.float(), pooled_buf.float().transpose(0, 1))
    scores = torch.relu(dots).sum(dim=1) / math.sqrt(index_dim)
    visible = position.to(torch.long) + 1
    n_complete = torch.div(visible, ratio, rounding_mode="floor")
    tail_len = visible - n_complete * ratio
    block_ids = torch.arange(pooled_buf.shape[0], device=q.device)
    scores = scores.masked_fill(~(block_ids.unsqueeze(0) < n_complete.unsqueeze(1)), float("-inf"))
    picked = torch.argsort(scores, dim=-1, stable=True, descending=True)[:, :k]
    offsets = torch.arange(ratio, device=q.device)
    block_tokens = (picked.unsqueeze(-1) * ratio + offsets).reshape(1, k * ratio)
    tail_off = torch.arange(ratio - 1, device=q.device)
    tail_pos = n_complete.unsqueeze(-1) * ratio + tail_off
    tail = torch.where(tail_off.unsqueeze(0) < tail_len.unsqueeze(1), tail_pos, torch.full_like(tail_pos, -1))
    indices = torch.cat((block_tokens, tail), dim=-1)

    heads, head_dim = q.shape[1], q.shape[2]
    kv_heads = key_buf.shape[1]
    q_mm = q.view(1, kv_heads, heads // kv_heads, head_dim).to(key_buf.dtype)
    valid = indices >= 0
    safe = indices.clamp(min=0).reshape(-1)
    width = indices.shape[-1]
    k_sel = key_buf.index_select(0, safe).view(1, width, kv_heads, head_dim)
    scores = (torch.matmul(q_mm, k_sel.permute(0, 2, 3, 1)) * scale).to(torch.float32)
    scores = scores.masked_fill(~valid[:, None, None, :], torch.finfo(torch.float32).min)
    row_has = valid.any(dim=-1)
    scores = scores.masked_fill(~row_has[:, None, None, None], 0)
    attn = torch.softmax(scores, dim=-1)
    attn = attn.masked_fill(~row_has[:, None, None, None], 0)
    v_sel = value_buf.index_select(0, safe).view(1, width, kv_heads, head_dim)
    out = torch.matmul(attn.to(value_buf.dtype), v_sel.permute(0, 2, 1, 3))
    return out.reshape(1, heads, head_dim).to(q.dtype)


def pool_block(block_keys, k_norm_w, eps: float, cos_sin):
    return _compiled(_qsa_pool_block)(block_keys, k_norm_w, eps, cos_sin)


def _qsa_pool_block_at(raw_buf, token_index, k_norm_w, eps: float, cos_sin):
    """``_qsa_pool_block`` reading the block's rows from the raw-key capacity buffer by index."""
    block_keys = raw_buf.index_select(0, token_index).view(1, token_index.shape[0], raw_buf.shape[-1])
    return _qsa_pool_block(block_keys, k_norm_w, eps, cos_sin)


def pool_block_at(raw_buf, token_index, k_norm_w, eps: float, cos_sin):
    """H31: the decode pooling graph without an offset view as input.

    ``pool_block`` took ``raw_keys[b * ratio:(b + 1) * ratio]``, a view whose storage offset moves
    with every pooled block. The bridge compiles a new recipe for each new offset without a Dynamo
    recompile, so the first request at a given length paid about 45 ms every fourth decode token
    (a repeated prompt was fast because its offsets were already cached). Reading the capacity
    buffer by an index tensor keeps one recipe per buffer capacity.
    """
    return _compiled(_qsa_pool_block_at)(raw_buf, token_index, k_norm_w, eps, cos_sin)


def select_attend(q, q_idx, pooled_buf, position, key_buf, value_buf, scale: float, ratio: int, k: int):
    return _compiled(_qsa_select_attend)(q, q_idx, pooled_buf, position, key_buf, value_buf, scale, ratio, k)


_WARMED: set[str] = set()


def warm_max_tokens() -> int:
    """Longest request the decode graphs are compiled for at first use (``VLLM_QWEN4EXP_WARM_MAX_TOKENS``)."""
    return int(os.environ.get("VLLM_QWEN4EXP_WARM_MAX_TOKENS", "32768"))


def _warm_enabled(name: str) -> bool:
    if name in _WARMED or os.environ.get("VLLM_QWEN4EXP_DECODE_GRAPH_WARM", "1") == "0":
        return False
    _WARMED.add(name)
    return True


def warm_select_attend(q, q_idx, pooled_buf, position, key_buf, value_buf, scale: float, ratio: int, k: int,
                       capacities) -> None:
    """Compile ``select_attend`` for every buffer capacity a request can reach, once per process.

    H31: the capacity buffers grow in steps, and each step used to compile a new graph in the
    middle of a user's decode (about 0.3-1 s each). The first decode step above the token budget
    (the launcher's 8K warmup request) now compiles them all on zero buffers. The query position is
    a fixed value inside the smallest capacity, so no index leaves any buffer.
    """
    if not _warm_enabled("select_attend"):
        return
    safe_position = torch.full_like(position, k * ratio + ratio)
    for capacity in capacities:
        blocks = capacity // ratio
        if capacity == key_buf.shape[0] and blocks == pooled_buf.shape[0]:
            continue
        pooled = pooled_buf.new_zeros((blocks, *pooled_buf.shape[1:]))
        keys = key_buf.new_zeros((capacity, *key_buf.shape[1:]))
        values = value_buf.new_zeros((capacity, *value_buf.shape[1:]))
        select_attend(q, q_idx, pooled, safe_position, keys, values, scale, ratio, k)
        del pooled, keys, values


def warm_pool_block_at(raw_buf, token_index, k_norm_w, eps: float, cos_sin, capacities) -> None:
    """``warm_select_attend`` for the decode pooling graph."""
    if not _warm_enabled("pool_block_at"):
        return
    safe_index = torch.arange(token_index.shape[0], device=token_index.device, dtype=token_index.dtype)
    for capacity in capacities:
        if capacity == raw_buf.shape[0]:
            continue
        raw = raw_buf.new_zeros((capacity, *raw_buf.shape[1:]))
        pool_block_at(raw, safe_index, k_norm_w, eps, cos_sin)
        del raw


def verifying() -> bool:
    return os.environ.get(_VERIFY_ENV, "0") not in ("", "0")


def _qsa_post_step(hidden, attn, gate, o_w, injection,
                   m_gamma, m_down, m_up, hc_eps: float, lowrank: int, streams: int,
                   gate_w, top_k: int, ids_shape: int,
                   gate_packed, gate_scale, gate_scale2, up_packed, up_scale, up_scale2,
                   down_packed, down_scale, down_scale2, shared_merged, shared_down):
    flat = attn.reshape(attn.shape[0], -1)
    attn_out = F.linear(flat * torch.sigmoid(gate), o_w)
    hidden, block_input, next_injection = _hc_combine_and_mix(
        hidden, attn_out, injection, m_gamma, hc_eps, m_down, m_up, lowrank, streams)
    mlp_out = _moe(block_input, gate_w, top_k, ids_shape,
                   gate_packed, gate_scale, gate_scale2, up_packed, up_scale, up_scale2,
                   down_packed, down_scale, down_scale2, shared_merged, shared_down)
    return hidden, mlp_out, next_injection


_COMPILED: dict[str, Any] = {}


def _compiled(fn) -> Any:
    name = fn.__name__
    if name not in _COMPILED:
        raise_dynamo_cache_limits()
        _COMPILED[name] = torch.compile(fn, **_COMPILE_ARGS)
    return _COMPILED[name]


def _cached_merge(owner, prefix: str, key_fn, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """The merged projection weight, shared with the eager path's cache of the same name.

    ``first`` and ``second`` are the parameters; ``key_fn()`` keys the cache on their current data,
    which H31 points into the merged tensor (``merged_weights.alias_params_into``).
    """
    from .merged_weights import merge_rows, views_of

    merged = getattr(owner, f"{prefix}_weight", None)
    if merged is not None and views_of(merged, first, second):
        setattr(owner, f"{prefix}_key", key_fn())
    elif merged is None or getattr(owner, f"{prefix}_key", None) != key_fn():
        merged = merge_rows(first, second)
        setattr(owner, f"{prefix}_weight", merged)
        setattr(owner, f"{prefix}_key", key_fn())
    return merged


def _moe_args(moe, top_k_slots: int) -> tuple:
    from .packed_moe import _scale2_on, ensure_gate_up_rowperm, shared_expert_h15_parts

    experts = moe.experts
    ensure_gate_up_rowperm(experts)
    parts = shared_expert_h15_parts(moe.shared_expert, getattr(moe, "shared_expert_gate", None))
    if parts is None:
        raise RuntimeError("decode graph needs the fused shared expert")
    down, gate_w, down_w, shared_w = parts[1:]
    if getattr(down, "bias", None) is not None:
        raise RuntimeError("decode graph expects a shared expert without bias")
    merged = _cached_merge(
        moe.shared_expert, "_h15_merged",
        lambda: (gate_w.data_ptr(), getattr(gate_w, "_version", 0), shared_w.data_ptr(),
                 getattr(shared_w, "_version", 0), tuple(gate_w.shape), str(gate_w.device)),
        gate_w, shared_w)
    packed = experts.gate_proj_packed
    return (
        moe.gate.weight.detach(), int(moe.top_k), int(top_k_slots),
        packed, experts.gate_proj_scale, _scale2_on(experts, "gate_proj", packed),
        experts.up_proj_packed, experts.up_proj_scale, _scale2_on(experts, "up_proj", packed),
        experts.down_proj_packed, experts.down_proj_scale, _scale2_on(experts, "down_proj", packed),
        merged, down_w.detach(),
    )


def _hc_args(hc) -> tuple:
    down = hc.input_mix_weight_down_block_inject.weight if hc.use_combine else hc.input_mix_weight_down.weight
    return hc._gamma2d(), down.detach(), hc.input_mix_weight_up.weight.detach()


def _gdn_args(attn) -> tuple:
    from .gdn_h14 import conv_weight_t

    w_qkvz = attn.in_proj_qkvz.weight
    w_ba = attn.in_proj_ba.weight
    merged = _cached_merge(attn, "_h14_merged",
                           lambda: (w_qkvz.data_ptr(), w_qkvz._version, w_ba.data_ptr(), w_ba._version,
                                    w_qkvz.shape, w_ba.shape), w_qkvz, w_ba)
    norm_w = attn.norm.weight.detach()
    if norm_w.dtype != torch.bfloat16:
        norm_w = norm_w.to(torch.bfloat16)
    if getattr(attn.out_proj, "bias", None) is not None:
        raise RuntimeError("decode graph expects a GDN out_proj without bias")
    a_log = attn.A_log.detach()
    if a_log.dtype != torch.float32:
        a_log = a_log.float()
    act = 1 if attn.norm.activation in ("silu", "swish") else 0
    return (
        merged, int(w_qkvz.shape[0]), int(attn.qkv_size), int(attn.z_size), int(attn.head_v_dim),
        conv_weight_t(attn), a_log.contiguous(), attn.dt_bias.detach().contiguous(),
        norm_w.contiguous(), float(attn.norm.eps), act, attn.out_proj.weight.detach(),
    )


class DecodeGraph:
    """Per-model cache of the bound arguments of every decoder layer."""

    def __init__(self, config) -> None:
        self.config = config
        self.eps = float(config.rms_norm_eps)
        self._bound: dict[int, tuple] = {}
        self._qsa_bound: dict[int, tuple | None] = {}
        self._cos_sin_key: tuple | None = None
        self._cos_sin: torch.Tensor | None = None
        self._index: dict[int, tuple[int, torch.Tensor, torch.Tensor]] = {}

    def _verifying(self) -> bool:
        return os.environ.get(_VERIFY_ENV, "0") not in ("", "0")

    def _log(self, layer, *rows: str) -> None:
        logger.warning("decode_graph verify layer=%d %s", layer.layer_idx, " | ".join(rows))

    def _bind(self, layer) -> tuple:
        key = id(layer)
        bound = self._bound.get(key)
        if bound is None:
            attn_hc = _hc_args(layer.attn_hyper_connection)
            mlp_hc = _hc_args(layer.mlp_hyper_connection)
            moe = _moe_args(layer.mlp, int(layer.mlp.top_k))
            gdn = _gdn_args(layer.linear_attn) if layer.layer_type == "linear_attention" else None
            lowrank = int(layer.attn_hyper_connection.hc_lowrank)
            streams = int(layer.attn_hyper_connection.hc_count)
            bound = self._bound[key] = (attn_hc, mlp_hc, moe, gdn, lowrank, streams)
        return bound

    def _state_index(self, attn, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Device tensors for the state slot, rebuilt only when the slot changes."""
        cached = self._index.get(id(attn))
        if cached is not None and cached[0] == slot:
            return cached[1], cached[2]
        pool = attn.kv_cache[1]
        safe = torch.tensor([slot % pool.shape[0]], dtype=torch.int32, device=pool.device)
        safe_long = torch.tensor([slot % pool.shape[0]], dtype=torch.long, device=pool.device)
        self._index[id(attn)] = (slot, safe, safe_long)
        return safe, safe_long

    def run_layer(self, layer, hidden, block, injection, *, positions, query_start_loc, input_ids=None,
                  ngram_context=None):
        """One decoder layer, or None when this layer has to take the eager path."""
        self._positions, self._qsl = positions, query_start_loc
        if hidden.dtype != torch.bfloat16:
            hidden = hidden.to(torch.bfloat16)
        if layer.ple is not None:
            # The PLE addition needs the host n-gram lookup, so it stays eager; it ends the pending
            # combine, and the rest of the layer is the no-combine GDN graph.
            if (layer.layer_type != "linear_attention" or self._verifying() or input_ids is None
                    or ngram_context is None or query_start_loc is None):
                return None
            slot = _state_slot(layer.linear_attn)
            if slot is None:
                return None
            if block is not None and injection is not None:
                hidden = layer.attn_hyper_connection.combine(hidden, block, injection)
            hidden = layer.ple(hidden, input_ids, query_start_loc, ngram_context, positions=positions)
            return self.run_gdn_layer(layer, hidden, None, None, slot)
        if layer.layer_type == "linear_attention":
            slot = _state_slot(layer.linear_attn)
            if slot is None:
                return None
            return self.run_gdn_layer(layer, hidden, block, injection, slot)
        if block is None:
            return None
        return self.run_qsa_layer(layer, hidden, block, injection, positions=positions,
                                  query_start_loc=query_start_loc)

    def run_gdn_layer(self, layer, hidden, block, injection, slot: int):
        attn_hc, mlp_hc, moe, gdn, lowrank, streams = self._bind(layer)
        attn = layer.linear_attn
        conv_pool = attn.kv_cache[0]
        ssm_pool = attn.kv_cache[1]
        safe, safe_long = self._state_index(attn, slot)
        in_merged, qkvz_rows, qkv_size, z_size, head_v_dim, conv_t, a_log, dt_bias, norm_w, norm_eps, act, out_w = gdn
        hc = (*attn_hc, *mlp_hc, self.eps, lowrank, streams)
        tail = (in_merged, qkvz_rows, qkv_size, z_size, head_v_dim, conv_pool.detach(), conv_t, a_log, dt_bias,
                ssm_pool.detach(), safe, norm_w, norm_eps, act, out_w, *moe)
        if os.environ.get("VLLM_QWEN4EXP_DECODE_GRAPH_STATE_INPLACE", "1") == "1" and not self._verifying():
            # The pools themselves, not detached aliases: they are read and written in the graph.
            tail_inplace = (in_merged, qkvz_rows, qkv_size, z_size, head_v_dim, conv_pool, conv_t, a_log, dt_bias,
                            ssm_pool, safe, norm_w, norm_eps, act, out_w, *moe)
            conv_at, ssm_at = len(hc) + 5, len(hc) + 9
            if block is None:
                return _compiled(_gdn_first_layer_step_inplace)(hidden, conv_at, ssm_at, safe_long, *hc,
                                                                *tail_inplace)
            return _compiled(_gdn_layer_step_inplace)(hidden, block, injection, conv_at, ssm_at, safe_long,
                                                      *hc, *tail_inplace)
        if block is None:
            out = _compiled(_gdn_first_layer_step)(hidden, *hc, *tail)
        else:
            out = _compiled(_gdn_layer_step)(hidden, block, injection, *hc, *tail)
        hidden_out, mlp_out, next_injection, next_conv, next_ssm = out
        if self._verifying():
            eager = layer(hidden_states=hidden, prev_block_output=block, prev_injection=injection,
                          positions=self._positions, input_ids=None, query_start_loc=self._qsl, ngram_context=None)
            self._log(layer, _diff("hidden", hidden_out, eager[0]), _diff("mlp_out", mlp_out, eager[1]),
                      _diff("injection", next_injection, eager[2]),
                      _diff("conv", next_conv, conv_pool.index_select(0, safe_long)),
                      _diff("ssm", next_ssm, ssm_pool.index_select(0, safe_long)))
            return eager
        conv_pool.index_copy_(0, safe_long, next_conv)
        ssm_pool.index_copy_(0, safe_long, next_ssm)
        return hidden_out, mlp_out, next_injection

    def _qsa_args(self, attn, block_input_like: torch.Tensor) -> tuple | None:
        key = id(attn)
        if key in self._qsa_bound:
            return self._qsa_bound[key]
        merged = attn._merged_projection_weight(block_input_like)
        bound = None
        if merged is not None and attn.attn_output_gate and getattr(attn.o_proj, "bias", None) is None:
            rotary = attn.rotary_emb
            apply = rotary.apply_rotary_emb
            indexer = attn.indexer
            bound = (merged, int(attn._qkv_rows), int(attn.q_size) * 2, int(attn.kv_size), int(attn.num_heads),
                     int(attn.num_kv_heads), int(attn.head_dim), True,
                     attn.q_norm.weight.detach(), float(attn.q_norm.variance_epsilon),
                     attn.k_norm.weight.detach(), float(attn.k_norm.variance_epsilon),
                     int(rotary.head_size), int(rotary.rotary_dim), bool(apply.is_neox_style),
                     bool(apply.enable_fp32_compute),
                     indexer.q_layernorm.weight.detach(), float(indexer.q_layernorm.variance_epsilon),
                     int(indexer.index_n_heads), int(indexer.index_kv_heads), int(indexer.index_head_dim),
                     attn.o_proj.weight.detach())
        self._qsa_bound[key] = bound
        return bound

    def _step_cos_sin(self, attn, positions, like: torch.Tensor) -> torch.Tensor:
        """The step's rotary rows, composed once for all twelve QSA layers."""
        from .qsa import _cos_sin_rows

        key = (id(positions), id(attn.rotary_emb.cos_sin_cache))
        if self._cos_sin_key != key:
            self._cos_sin_key = key
            self._cos_sin = _cos_sin_rows(attn.rotary_emb, positions, like)
        return self._cos_sin

    def run_qsa_layer(self, layer, hidden, block, injection, *, positions, query_start_loc):
        attn_hc, mlp_hc, moe, _gdn_unused, lowrank, streams = self._bind(layer)
        attn = layer.self_attn
        qsa = self._qsa_args(attn, block)
        if qsa is None:
            return None
        merged, qkv_rows, q_width, kv_size, heads, kv_heads, head_dim, gated, q_w, q_eps, k_w, k_eps, \
            head_size, rotary_dim, neox, fp32, idx_w, idx_eps, idx_heads, idx_kv_heads, idx_dim, o_w = qsa
        cos_sin = self._step_cos_sin(attn, positions, block)
        pre = _compiled(_qsa_pre_step)(
            hidden, block, injection, attn_hc[0], self.eps, attn_hc[1], attn_hc[2], lowrank, streams,
            merged, qkv_rows, q_width, kv_size, heads, kv_heads, head_dim, gated, q_w, q_eps, k_w, k_eps,
            cos_sin, head_size, rotary_dim, neox, fp32, idx_w, idx_eps, idx_heads, idx_kv_heads, idx_dim)
        hidden_out, next_injection, q, k, v, gate, q_idx, raw_k = pre
        if self._verifying():
            e_hidden, e_block, e_inj = layer.attn_hyper_connection.combine_and_mix(hidden, block, injection)
            projected = F.linear(e_block, merged)
            e_q, e_k, e_v, e_gate = attn._project_qkv_gate(projected[:, :qkv_rows], positions)
            e_q_idx, e_raw_k = attn.indexer.project_qk_from_proj(projected[:, qkv_rows:], positions, attn.rotary_emb)
            e_q, e_k, e_v = (e_q.view(1, heads, head_dim), e_k.view(1, kv_heads, head_dim),
                             e_v.view(1, kv_heads, head_dim))
            self._log(layer, _diff("pre.hidden", hidden_out, e_hidden), _diff("pre.inj", next_injection, e_inj),
                      _diff("q", q, e_q), _diff("k", k, e_k), _diff("v", v, e_v), _diff("gate", gate, e_gate),
                      _diff("q_idx", q_idx, e_q_idx), _diff("raw_k", raw_k, e_raw_k))
            hidden_out, next_injection, q, k, v, gate, q_idx, raw_k = (e_hidden, e_inj, e_q, e_k, e_v, e_gate,
                                                                       e_q_idx, e_raw_k)
        attended = attn._attend_cached(q, k, v, q_idx, raw_k, positions, query_start_loc)
        out = _compiled(_qsa_post_step)(
            hidden_out, attended, gate, o_w, next_injection, mlp_hc[0], mlp_hc[1], mlp_hc[2], self.eps, lowrank,
            streams, *moe)
        if self._verifying():
            flat = attended.reshape(attended.shape[0], -1) * torch.sigmoid(gate)
            e_attn = F.linear(flat, o_w)
            e_hidden, e_block, e_inj = layer.mlp_hyper_connection.combine_and_mix(hidden_out, e_attn, next_injection)
            e_out = layer.mlp(e_block)
            self._log(layer, _diff("post.hidden", out[0], e_hidden), _diff("post.mlp_out", out[1], e_out),
                      _diff("post.inj", out[2], e_inj))
            return e_hidden, e_out, e_inj
        return out


_HOST_STATE_INDICES: list[torch.Tensor | None] = [None]


def publish_host_state_indices(indices: torch.Tensor | None) -> None:
    """The runner's CPU copy of the decode state indices, ``[groups, batch]``.

    GDN reads its slot from these instead of the device tensor, whose read would drain the
    launch queue on every layer.
    """
    _HOST_STATE_INDICES[0] = indices


def _state_slot(attn) -> int | None:
    indices = _HOST_STATE_INDICES[0]
    if indices is None or indices.numel() == 0:
        return None
    if indices.dim() == 1:
        return int(indices[0])
    group = getattr(attn, "_decode_graph_group", None)
    if group is None:
        group = attn._decode_graph_group = int(attn.cache_group_idx)
    return int(indices[group, 0])


_MODEL_GRAPHS: dict[int, DecodeGraph] = {}


def for_step(model, hidden: torch.Tensor) -> DecodeGraph | None:
    """The graph runner when this step is a single-token decode, otherwise None."""
    if not enabled() or hidden.shape[0] != 1 or hidden.device.type != "hpu":
        return None
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return None
    metadata = get_forward_context().attn_metadata
    if metadata is None or bool(getattr(metadata, "is_prompt", True)):
        return None
    if _HOST_STATE_INDICES[0] is None:
        return None
    graph = _MODEL_GRAPHS.get(id(model))
    if graph is None:
        graph = _MODEL_GRAPHS[id(model)] = DecodeGraph(model.config)
    # H31: the rotary rows are keyed on ``id(positions)`` for the twelve QSA layers of one step. The
    # step's positions tensor is freed afterwards, and the next step's tensor can get the same id, so
    # the key must not outlive the step (it used to, and could hand a step the previous position).
    graph._cos_sin_key = None
    graph._cos_sin = None
    k4_gemv_enabled()  # resolve the op lookup here, outside the traced step functions
    return graph
