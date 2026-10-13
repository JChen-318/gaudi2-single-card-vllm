# SPDX-License-Identifier: Apache-2.0
"""Bounded top-k/top-p with whole-device fallback for unsupported boundaries.

Candidates are normalized AFTER the same top-k threshold mask as upstream.
Request RNG is deliberately not touched by this filter. Ties, nonfinite rows,
large/no top-k and near top-p rounding boundaries take the original full sort.
"""

import torch
from vllm.v1.sample.ops.topk_topp_sampler import (
    TopKTopPSampler,
    apply_top_k_top_p_pytorch,
    random_sample,
)

CAPACITY = 64
BOUNDARY_MARGIN = 1e-4


@torch.compiler.disable
def stable_random_sample(probs, generators, use_fp64_gumbel):
    # Keep RNG execution outside graph capture. In this runtime, unseeded
    # compiled exponential draws consume a different global generator state.
    return random_sample(probs, generators, use_fp64_gumbel)


class HPUStableTopKTopPSampler(TopKTopPSampler):
    """Original filter/distribution; only the RNG graph boundary is explicit."""

    def __init__(self, logprobs_mode="raw_logprobs", use_fp64_gumbel=False):
        super().__init__(logprobs_mode, use_fp64_gumbel)
        self.forward = self.forward_stable

    def forward_stable(self, logits, generators, k, p):
        logits = apply_top_k_top_p_pytorch(logits, k, p)
        to_return = None
        if self.logprobs_mode == "processed_logits":
            to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            to_return = logits.log_softmax(-1, dtype=torch.float32)
        probs = logits.softmax(-1, dtype=torch.float32)
        return stable_random_sample(probs, generators, self.use_fp64_gumbel), to_return


def candidate_filter(logits, k, p):
    values, indices = logits.topk(CAPACITY, dim=-1, largest=True, sorted=False)
    values, order = values.sort(dim=-1)
    indices = indices.gather(1, order)
    safe_k = k.clamp(1, CAPACITY - 1).to(torch.long)
    threshold = values.gather(1, (CAPACITY - safe_k)[:, None])
    filtered = values.masked_fill(values < threshold, -float("inf"))
    live = values >= threshold
    # A strict guard ensures no unrepresented threshold ties outside candidates.
    safe = (
        (k >= 1)
        & (k < CAPACITY)
        & (k < logits.shape[1])
        & (values[:, 0] < threshold[:, 0])
        & torch.isfinite(values).all(1)
        & ~torch.isnan(logits).any(1)
    )
    tied = (values[:, 1:] == values[:, :-1]) & live[:, 1:] & live[:, :-1]
    safe = safe & ~tied.any(1)
    if p is not None:
        cumulative = filtered.softmax(-1).cumsum(-1)
        boundary = 1 - p[:, None]
        ambiguous = ((cumulative - boundary).abs() <= BOUNDARY_MARGIN) & live
        safe = safe & ~ambiguous.any(1) & (p > 0) & (p <= 1)
        remove = cumulative <= boundary
        # Keep the largest token, exactly like upstream ascending sort.
        remove = torch.cat((remove[:, :-1], torch.zeros_like(remove[:, :1])), 1)
        filtered = filtered.masked_fill(remove, -float("inf"))
    output = torch.full_like(logits, -float("inf")).scatter(1, indices, filtered)
    return output, safe.all()


def bounded_top_k_top_p(logits, k, p):
    if k is None or logits.shape[-1] <= CAPACITY:
        return apply_top_k_top_p_pytorch(logits, k, p)
    candidate, safe = candidate_filter(logits, k, p)

    # No item()/CPU read; only the selected branch runs on the target device.
    def fast(original, chosen, kvals, *pvals):
        return chosen.clone()

    def fallback(original, chosen, kvals, *pvals):
        return apply_top_k_top_p_pytorch(
            original.clone(), kvals, pvals[0] if pvals else None
        )

    operands = (logits, candidate, k) if p is None else (logits, candidate, k, p)
    return torch.cond(safe, fast, fallback, operands)


class HPUBoundedTopKTopPSampler(TopKTopPSampler):
    """Bonus sampling with unchanged full-vocabulary exponential RNG draws."""

    def __init__(self, logprobs_mode="raw_logprobs", use_fp64_gumbel=False):
        super().__init__(logprobs_mode, use_fp64_gumbel)
        self.forward = self.forward_bounded

    def forward_bounded(self, logits, generators, k, p):
        logits = bounded_top_k_top_p(logits, k, p)
        to_return = None
        if self.logprobs_mode == "processed_logits":
            to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            to_return = logits.log_softmax(-1, dtype=torch.float32)
        probs = logits.softmax(-1, dtype=torch.float32)
        return stable_random_sample(probs, generators, self.use_fp64_gumbel), to_return
