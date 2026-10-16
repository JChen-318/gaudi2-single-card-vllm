# SPDX-License-Identifier: Apache-2.0
"""Contiguous K/V mirror for single-request long-context decode (VLLM_HPU_DECODE_KV_MIRROR_PAGES).

Dense shared-prefix decode attention (extension/dense_decode_attn.py) gathers the request's pages out of the paged
cache in every layer of every round: for a 180K context that is ~350 MB read and written per layer, about 40% of
the layer's attention time. Between two rounds only the last one or two pages change.

Here every target full-attention layer gets a persistent buffer laid out like the cache, ``[pages * page, kv_heads,
head]``, holding the decoding request's pages in logical order. The decode graph copies the (at most two) pages
holding this round's verification tokens from the cache right after writing them, then attends over the buffer's
first ``bucket`` pages directly. The host keeps track of which logical pages the buffer holds and copies whatever
is missing before the forward (a new request, or pages written while the mirror was not in use).

Used only when the decode batch is one request (padded batch 1) and its page bucket lies in
``[min_pages, pages]``; whether a graph uses the mirror is therefore a function of its shape, so the warmup
compiles exactly the graphs serving needs. Page validity rules:

- a mirror forward with first tail page ``l0`` needs pages ``[0, l0)`` valid and leaves ``[0, l0 + 1)`` valid
  (the next round starts at most one page later, and its own tail covers the rest);
- a forward that writes the request's pages without the mirror (two-request decode, a bucket outside the range)
  invalidates everything from the page it starts writing at;
- the request's physical page ids are compared with the ones copied, so a re-allocated page is copied again.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch

REFRESH_CHUNK = 512  # pages per refresh call (fixed shape: one compiled graph)


def copy_pages(key_cache, value_cache, key_mirror, value_mirror, src, dst, page: int):
    """mirror page dst[i] <- cache page src[i] (both caches)."""
    for cache, mirror in ((key_cache, key_mirror), (value_cache, value_mirror)):
        mirror.unflatten(0, (-1, page)).index_copy_(0, dst, cache.unflatten(0, (-1, page)).index_select(0, src))


def tail_pages(start: int, page: int, bucket_pages: int) -> np.ndarray:
    """Logical pages holding the verification tokens of a round that starts at ``start`` (8 tokens never span
    more than two pages); the second one is clamped into the bucket (copying a padding page is harmless)."""
    first = start // page
    return np.asarray([first, min(first + 1, bucket_pages - 1)], dtype=np.int64)


class DecodeKVMirror:

    def __init__(self, pages: int, min_pages: int, page: int, device, copy_fn: Optional[Callable] = None):
        self.pages = int(pages)
        self.min_pages = max(1, int(min_pages))
        self.page = int(page)
        self.device = device
        self._copy = copy_fn or copy_pages
        self.layers: dict[int, list] = {}  # group id -> [(key_cache, value_cache, key_mirror, value_mirror)]
        self.owner: Optional[str] = None
        self.valid: dict[int, np.ndarray] = {}  # group id -> physical page ids of mirror pages [0, n) known valid
        self.stats = {"refresh_pages": 0, "refreshes": 0, "rounds": 0}

    # -- setup -------------------------------------------------------------
    def bytes_per_layer(self, kv_heads: int, head: int, itemsize: int) -> int:
        return 2 * self.pages * self.page * kv_heads * head * itemsize

    def attach(self, group_id: int, impl, key_cache: torch.Tensor, value_cache: torch.Tensor) -> None:
        """Allocate the layer's mirror (zeros: masked keys must stay finite) and hand it to the attention impl."""
        shape = (self.pages * self.page, *key_cache.shape[1:])
        key_mirror = torch.zeros(shape, dtype=key_cache.dtype, device=key_cache.device)
        value_mirror = torch.zeros(shape, dtype=value_cache.dtype, device=value_cache.device)
        impl._kv_mirror = (key_mirror, value_mirror)
        self.layers.setdefault(group_id, []).append((key_cache, value_cache, key_mirror, value_mirror))

    def eligible(self, padded_batch: int, num_reqs: int, bucket_pages: int) -> bool:
        return padded_batch == 1 and num_reqs == 1 and self.min_pages <= bucket_pages <= self.pages

    def reset(self) -> None:
        self.owner = None
        self.valid = {}

    # -- per round ---------------------------------------------------------
    def plan(self, req_ids: Sequence[str], starts: Sequence[int], padded_batch: int, bucket_pages: int,
             page_table: Callable[[int, str, int], np.ndarray], warmup: bool = False) -> Optional[np.ndarray]:
        """Called once per decode forward that is about to be launched. Returns the tail pages (the mirror
        is used) or None (the gather path is used). ``page_table(group_id, req_id, n)`` returns the request's
        first ``n`` physical pages in that group."""
        if not self.eligible(padded_batch, len(req_ids), bucket_pages):
            if self.owner is not None:
                for req_id, start in zip(req_ids, starts):
                    if req_id == self.owner:
                        self._truncate(int(start) // self.page)
            return None
        start = int(starts[0])
        tail = tail_pages(start, self.page, bucket_pages)
        if warmup:
            return tail
        req_id = req_ids[0]
        if req_id != self.owner:
            self.owner = req_id
            self.valid = {}
        first = int(tail[0])
        for gid, layers in self.layers.items():
            table = np.asarray(page_table(gid, req_id, first + 1), dtype=np.int64)
            known = self.valid.get(gid, np.empty(0, dtype=np.int64))
            n = min(len(known), first)
            same = known[:n] == table[:n]
            upto = n if bool(same.all()) else int(np.argmin(same))
            if upto < first:
                self._refresh(layers, table[upto:first], np.arange(upto, first, dtype=np.int64))
            self.valid[gid] = table[:first + 1].copy()
        self.stats["rounds"] += 1
        return tail

    def note_write(self, req_id: str, start: int) -> None:
        """The request's KV from ``start`` on was written without the mirror (e.g. a prefill chunk)."""
        if req_id == self.owner:
            self._truncate(int(start) // self.page)

    def _truncate(self, pages: int) -> None:
        for gid, known in self.valid.items():
            if len(known) > pages:
                self.valid[gid] = known[:max(pages, 0)]

    def _refresh(self, layers, src: np.ndarray, dst: np.ndarray) -> None:
        self.stats["refreshes"] += 1
        self.stats["refresh_pages"] += len(src)
        for begin in range(0, len(src), REFRESH_CHUNK):
            s = src[begin:begin + REFRESH_CHUNK]
            d = dst[begin:begin + REFRESH_CHUNK]
            if len(s) < REFRESH_CHUNK:  # pad with repeats of the last pair: same source, same destination
                s = np.concatenate([s, np.full(REFRESH_CHUNK - len(s), s[-1], dtype=np.int64)])
                d = np.concatenate([d, np.full(REFRESH_CHUNK - len(d), d[-1], dtype=np.int64)])
            s_dev = torch.from_numpy(s).to(self.device)
            d_dev = torch.from_numpy(d).to(self.device)
            for key_cache, value_cache, key_mirror, value_mirror in layers:
                self._copy(key_cache, value_cache, key_mirror, value_mirror, s_dev, d_dev, self.page)
