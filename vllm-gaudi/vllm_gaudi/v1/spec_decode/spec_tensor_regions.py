# SPDX-License-Identifier: Apache-2.0
"""Tensor-only speculative regions; request RNG and CPU scheduling stay outside."""

import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

from vllm_gaudi.v1.sample import fast_topk_topp


def constrain_logits(logits, temperature, top_k, top_p):
    batch = temperature.shape[0]
    width = logits.shape[0] // batch
    temperature = torch.where(temperature <= 0, torch.ones_like(temperature), temperature)
    logits.div_(temperature[:, None].expand(batch, width).reshape(-1, 1))
    k = None if top_k is None else top_k[:, None].expand(batch, width).reshape(-1)
    p = None if top_p is None else top_p[:, None].expand(batch, width).reshape(-1)
    if k is not None and fast_topk_topp.is_active():
        return fast_topk_topp.fast_top_k_top_p(logits, k, p)
    return apply_top_k_top_p_pytorch(logits, k, p)


def correct_random_tokens(draft_ids,
                          cumulative_lengths,
                          proposal_probs,
                          target_logits,
                          bonus_ids,
                          temperature,
                          uniforms,
                          noise,
                          all_random,
                          synthetic_rates=None):
    """Recover only the first rejected position without changing any RNG draws.

    Softmax/acceptance still evaluate all rectangular rows. Exponential noise
    remains one complete vocabulary vector per request, shared across positions.
    Later rejected positions never appear in the returned prefix, so their
    recovery distributions need not be materialized or reduced.
    """
    batch, width = draft_ids.shape
    vocab = target_logits.shape[-1]
    draft_ids = draft_ids.to(torch.int64)
    probs = target_logits.softmax(-1, dtype=torch.float32).view(batch, width, vocab)
    lengths = cumulative_lengths - torch.cat((torch.zeros_like(cumulative_lengths[:1]), cumulative_lengths[:-1]))
    lengths = lengths.to(torch.long)
    pos = torch.arange(width, device=draft_ids.device, dtype=lengths.dtype)
    valid = pos[None, :] < lengths[:, None]
    safe_ids = draft_ids.clamp(0, vocab - 1)
    valid_ids = (draft_ids >= 0) & (draft_ids < vocab)
    target_p = probs.gather(-1, safe_ids[:, :, None]).squeeze(-1)
    proposal = None if proposal_probs is None else proposal_probs.view(batch, width, vocab).to(torch.float32)
    proposal_p = torch.ones_like(target_p) if proposal is None else proposal.gather(-1, safe_ids[:, :,
                                                                                                 None]).squeeze(-1)
    ratio = torch.where(proposal_p > 0, target_p / proposal_p, torch.zeros_like(target_p))
    accepted = uniforms < (ratio if synthetic_rates is None else synthetic_rates[:width][None, :])
    accepted = accepted & valid & valid_ids
    rejected = ~accepted & valid
    target_argmax = None
    if not all_random:
        target_argmax = probs.argmax(-1)
        rejected = torch.where((temperature <= 0)[:, None], (draft_ids != target_argmax) & valid, rejected)
    any_rejected = rejected.any(1)
    first = rejected.to(torch.int32).argmax(1).to(torch.long)
    selected_p = probs.gather(1, first[:, None, None].expand(batch, 1, vocab)).squeeze(1)
    selected_mask = pos[None, :] == first[:, None]
    # HPU cannot lower B1 int64 gather with a one-element output. IDs fit
    # int32; selecting from the seven-element row with a mask is exact.
    selected_ids = torch.where(selected_mask, safe_ids.to(torch.int32), 0).sum(1, keepdim=True).to(torch.long)
    if proposal is None:
        recovery = selected_p.clone().scatter_(-1, selected_ids, 0.0)
        selected_valid = (valid_ids & selected_mask).any(1, keepdim=True)
        recovery = torch.where(selected_valid, recovery, selected_p)
    else:
        selected_q = proposal.gather(1, first[:, None, None].expand(batch, 1, vocab)).squeeze(1)
        recovery = torch.clamp(selected_p - selected_q, min=0.0)
    recovered = (recovery / noise).argmax(-1)
    recovery_position = any_rejected[:, None] & (pos[None, :] == first[:, None])
    chosen = torch.where(recovery_position, recovered[:, None], draft_ids)
    if not all_random:
        chosen = torch.where((temperature <= 0)[:, None], target_argmax, chosen)
    length = torch.where(any_rejected, first + 1, lengths)
    prefix = torch.where((pos[None, :] < length[:, None]) & valid, chosen, -1).to(torch.int32)
    output = torch.cat((prefix, torch.full_like(prefix[:, :1], -1)), 1)
    output_pos = torch.arange(width + 1, device=draft_ids.device, dtype=lengths.dtype)
    bonus_mask = ~any_rejected[:, None] & (output_pos[None, :] == lengths[:, None])
    return torch.where(bonus_mask, bonus_ids.to(torch.int32).view(batch, 1), output)


_constraints_region = constrain_logits
_correction_region = correct_random_tokens


def configure_regions(compile_fn):
    """Install compiled tensor callables once, before warmup, never mid-request."""
    global _constraints_region, _correction_region
    _constraints_region = compile_fn(constrain_logits)
    # Synapse 1.24.1 cannot lower the complete random correction graph:
    # argmax_multi_dim_fwd_f32 fails in GC_COMPLEX_GUID at B1/T7/V248320.
    # Keep the reduced BxV recovery math but submit its supported eager ops.
    # Do not catch/retry a recipe failure after HPU work was submitted.
    _correction_region = torch.compiler.disable(correct_random_tokens)


def apply_constraints(*args):
    return _constraints_region(*args)


def apply_correction(*args):
    return _correction_region(*args)
