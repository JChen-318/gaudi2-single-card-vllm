# SPDX-License-Identifier: Apache-2.0
"""Top-k/top-p without sorting the vocabulary (VLLM_HPU_FAST_TOPK_TOPP).

Upstream ``apply_top_k_top_p_pytorch`` sorts every row of the 248K vocabulary
(about 2 ms per round on Gaudi2). With a small ``k`` only the largest ``k``
values matter. They are found in two exact stages: the per-chunk maxima pick
the ``C`` chunks that can hold a top-``C`` value, and a top-``C`` over those
chunks returns the values in descending order.

The result is the same filtered logits as upstream, with one tie policy:
values tied with the k-th value are kept (as upstream), and values tied with
the last token kept by top-p are all kept (upstream keeps an arbitrary subset,
decided by an unstable sort). Rounding of the top-p cumulative sum can differ
in the last bit. The caller enables this only when every row has
``1 <= k <= CAPACITY`` and ``0 < p <= 1``; other batches keep the sort.
"""

from __future__ import annotations

import torch

CAPACITY = 64
CHUNK = 256

# Set by the runner once per step from host-side sampling parameters.
_ACTIVE = False


def set_active(active: bool) -> None:
    global _ACTIVE
    _ACTIVE = bool(active)


def is_active() -> bool:
    return _ACTIVE


def chunked_topk(scores: torch.Tensor, count: int, chunk: int = CHUNK) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact top-``count`` values (descending) and int32 indices along the last dim of a 2-D tensor."""
    rows, vocab = scores.shape
    chunks = (vocab + chunk - 1) // chunk
    if chunks * chunk != vocab:
        scores = torch.nn.functional.pad(scores, (0, chunks * chunk - vocab), value=float("-inf"))
    blocks = scores.view(rows, chunks, chunk)
    picked = min(count, chunks)
    _, chunk_ids = blocks.amax(dim=-1).topk(picked, dim=-1)
    gathered = blocks.gather(1, chunk_ids.unsqueeze(-1).expand(rows, picked, chunk)).reshape(rows, picked * chunk)
    values, flat = gathered.topk(count, dim=-1)
    # Gaudi cannot compile a gather of int64 data; ids fit int32.
    chunk_ids = chunk_ids.to(torch.int32)
    flat = flat.to(torch.int32)
    indices = chunk_ids.gather(1, (flat // chunk).to(torch.int64)) * chunk + flat % chunk
    return values, indices


def fast_top_k_top_p(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
    if k is None:
        raise ValueError("fast_top_k_top_p requires top-k")
    values, _ = chunked_topk(logits, CAPACITY)
    k = k.to(torch.long).clamp(1, CAPACITY).unsqueeze(1)
    threshold = values.gather(1, k - 1)
    if p is not None:
        in_list = values >= threshold
        # Kept tokens tied with the k-th value but beyond the candidate list.
        extra = (logits >= threshold).sum(-1, keepdim=True) - in_list.sum(-1, keepdim=True)
        top = values[:, :1]
        weights = torch.where(in_list, (values - top).exp(), torch.zeros_like(values))
        total = weights.sum(-1, keepdim=True) + extra.clamp_min(0).to(values.dtype) * (threshold - top).exp()
        probs = weights / total
        before = probs.cumsum(-1) - probs
        # Upstream keeps a token when the probability mass ranked above it is
        # below p (ascending cumsum > 1 - p); the largest token always stays.
        kept = in_list & (before < p.to(values.dtype).unsqueeze(1))
        last = kept.sum(-1, keepdim=True).clamp_min(1)
        threshold = torch.maximum(threshold, values.gather(1, last - 1))
    return logits.masked_fill(logits < threshold, float("-inf"))


def install() -> None:
    """Route the upstream sampler's top-k/top-p through the fast path while active."""
    from vllm.v1.sample.ops import topk_topp_sampler

    original = topk_topp_sampler.apply_top_k_top_p
    if getattr(original, "_hpu_fast_topk_topp", False):
        return

    def apply_top_k_top_p(logits, k, p):
        if k is not None and _ACTIVE:
            return fast_top_k_top_p(logits, k, p)
        return original(logits, k, p)

    apply_top_k_top_p._hpu_fast_topk_topp = True
    topk_topp_sampler.apply_top_k_top_p = apply_top_k_top_p


def eligible(top_k, top_p, no_top_k: bool, no_top_p: bool) -> bool:
    """Host check on the CPU copies of the per-request parameters."""
    if no_top_k or top_k is None or len(top_k) == 0:
        return False
    if (top_k < 1).any() or (top_k > CAPACITY).any():
        return False
    if not no_top_p and top_p is not None and ((top_p <= 0).any() or (top_p > 1).any()):
        return False
    return True
