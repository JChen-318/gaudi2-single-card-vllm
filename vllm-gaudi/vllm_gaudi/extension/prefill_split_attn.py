"""Chunked-prefill attention over a long cached context as a fixed set of kernel calls (VLLM_HPU_PREFILL_SPLIT_ATTN).

FusedSDPA with the full ``[query, context + query]`` bias is about twice as slow as running the same attention in
8K-key slices without a bias, but a Python loop over slices unrolls into a different graph for every context length,
and the prompt graphs are compiled with a dynamic context dimension. Here the slices of the context are stacked
into the batch dimension instead, so one kernel call covers all of them and the slice count stays a symbolic size:

- clean slices: every slice of the context except the last, no bias (one batched call);
- edge slice: the last context slice with its bias (bucket padding sits right before the query tokens and, for the
  shapes accepted by ``eligible``, never reaches further back than one slice);
- diagonal: the chunk against itself with its causal bias;
- the partial results are merged with their row maxima and normalisers (online softmax).
"""
import torch

SLICE = 8192


def eligible(query, key, attn_bias, is_causal, max_query, min_prefix, slice_size=SLICE) -> bool:
    """Layout ``[batch, heads, tokens, head_dim]``. Shapes for which padding is confined to the last slice: the
    largest prompt query (long prompts run as chunks of it), contexts on the fine bucket grid (multiples of the
    slice, at least ``min_prefix`` tokens, see VLLM_HPU_PROMPT_FINE_CTX_STEP)."""
    if attn_bias is None or not is_causal or query.shape[0] != 1:
        return False
    q_len = query.shape[-2]
    prefix = key.shape[-2] - q_len
    return q_len == max_query and prefix >= max(min_prefix, 2 * slice_size) and prefix % slice_size == 0


def _sdpa(q, k, v, mask, scale, softmax_mode):
    """FusedSDPA forward on GQA-shaped inputs: normalised output (input dtype; the stacked slices make it the
    largest tensor here, so it is not widened), row max and the normaliser (float32). The normaliser of the "fast"
    mode carries a constant factor; it is the same for every call, so it cancels in the merge."""
    out, m, linv = torch.ops.hpu.sdpa_recomp_fwd(q, k, v, mask, 0.0, scale, False, True, softmax_mode, None,
                                                 'right')[:3]
    return out, m.to(torch.float32), linv.to(torch.float32)


def split_attention(query, key, value, attn_bias, scale, softmax_mode='fast', slice_size=SLICE, sdpa=_sdpa):
    """query ``[1, Hq, q, D]``; key/value ``[1, Hkv, prefix + q, D]`` (context then chunk); attn_bias
    ``[1, 1, q, prefix + q]``. Returns ``[1, Hq, q, D]`` in the query dtype."""
    batch, q_heads, q_len, head_dim = query.shape
    kv_heads = key.shape[1]
    group = q_heads // kv_heads
    prefix = key.shape[-2] - q_len
    clean = prefix - slice_size
    n = clean // slice_size

    qg = query.reshape(batch, kv_heads, group, q_len, head_dim)
    kg = key.reshape(batch, kv_heads, 1, prefix + q_len, head_dim)
    vg = value.reshape(batch, kv_heads, 1, prefix + q_len, head_dim)
    bias = attn_bias.reshape(batch, 1, 1, q_len, prefix + q_len)

    def stacked(t):  # [1, Hkv, 1, clean, D] -> [n, Hkv, 1, slice, D]
        t = t[..., :clean, :].reshape(batch, kv_heads, n, slice_size, head_dim)
        return t.permute(0, 2, 1, 3, 4).reshape(batch * n, kv_heads, 1, slice_size, head_dim)

    q_rep = qg.expand(batch * n, kv_heads, group, q_len, head_dim).contiguous()
    o1, m1, l1 = sdpa(q_rep, stacked(kg), stacked(vg), None, scale, softmax_mode)
    o2, m2, l2 = sdpa(qg, kg[..., clean:prefix, :].contiguous(), vg[..., clean:prefix, :].contiguous(),
                      bias[..., clean:prefix].contiguous(), scale, softmax_mode)
    o3, m3, l3 = sdpa(qg, kg[..., prefix:, :].contiguous(), vg[..., prefix:, :].contiguous(),
                      bias[..., prefix:].contiguous(), scale, softmax_mode)

    m_max = torch.maximum(torch.maximum(m2, m3), m1.amax(dim=0, keepdim=True))
    w1 = torch.exp(m1 - m_max) / l1
    w2 = torch.exp(m2 - m_max) / l2
    w3 = torch.exp(m3 - m_max) / l3
    out = ((w1 * o1).sum(dim=0, keepdim=True) + w2 * o2 + w3 * o3) / (w1.sum(dim=0, keepdim=True) + w2 + w3)
    return out.reshape(batch, q_heads, q_len, head_dim).to(query.dtype)
