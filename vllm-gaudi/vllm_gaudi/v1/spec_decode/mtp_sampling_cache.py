# SPDX-License-Identifier: Apache-2.0
"""Reuse immutable selected scalars within an unchanged input batch."""


def select_parameters(batch, indices, skip_copy, enabled):
    signature = (tuple(indices), batch.all_greedy, batch.no_top_p, batch.no_top_k,
                 id(batch.temperature), id(batch.top_p), id(batch.top_k))
    if not enabled or not skip_copy:
        batch._mtp_sampling_scalar_cache = {}
    cache = getattr(batch, '_mtp_sampling_scalar_cache', {})
    if enabled and skip_copy and signature in cache:
        return cache[signature]
    # Always own the selected tensors: no view may observe a later batch update.
    selected = (None if batch.all_greedy else batch.temperature[indices],
                None if batch.no_top_p else batch.top_p[indices],
                None if batch.no_top_k else batch.top_k[indices])
    if enabled:
        if len(cache) >= 8:
            cache = {}
        cache[signature] = selected
        batch._mtp_sampling_scalar_cache = cache
    return selected
