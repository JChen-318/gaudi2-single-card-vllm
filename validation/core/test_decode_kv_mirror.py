"""Contiguous K/V mirror for single-request decode (VLLM_HPU_DECODE_KV_MIRROR_PAGES): the host planner's page
bookkeeping and the dense attention reading the mirror, simulated over many rounds against the gather path.

Every simulated round writes the verification tokens' K/V into a scattered paged cache, asks the planner for the
tail (copying missing pages), and checks that attention over the mirror is bit-identical to gathering the pages.
Covers spec-decode advances, page-boundary crossings, two-request rounds (no mirror) in between, re-allocated
pages, a new request, a prefill rewrite and a planned round that is never executed (lookahead discarded).

    PYTHONPATH=vllm-gaudi python -m pytest -q validation/core/test_decode_kv_mirror.py
"""
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import torch

SRC = Path(os.environ.get("VLLM_GAUDI_SRC", Path(__file__).resolve().parents[2] / "vllm-gaudi" / "vllm_gaudi"))


def _load(path, name):
    """Both modules only need torch / numpy; load them from their files without importing the plugin."""
    spec = importlib.util.spec_from_file_location(name, SRC / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dense = _load("extension/dense_decode_attn.py", "dense_decode_attn_mirror")
mirror_mod = _load("v1/worker/decode_kv_mirror.py", "decode_kv_mirror_under_test")

PAGE, KV, GROUP, HEAD, WIDTH = 16, 2, 3, 32, 8
SLOTS_PAGES = 400  # physical pages in the cache; the last one is the padding page
PAD = SLOTS_PAGES - 1
SCALE = HEAD**-0.5


class Sim:
    """One layer group: paged cache, page tables per request, a mirror, the planner."""

    def __init__(self, mirror_pages=96, min_pages=4, seed=0):
        self.gen = torch.Generator().manual_seed(seed)
        self.kc = torch.zeros(SLOTS_PAGES * PAGE, KV, HEAD, dtype=torch.bfloat16)
        self.vc = torch.zeros_like(self.kc)
        self.free = list(torch.randperm(PAD, generator=self.gen).numpy())
        self.tables = {}
        self.planner = mirror_mod.DecodeKVMirror(mirror_pages, min_pages, PAGE, "cpu")

        class Impl:
            pass

        self.impl = Impl()
        self.planner.attach(0, self.impl, self.kc, self.vc)

    def table(self, gid, req, count):
        return np.asarray(self.tables[req][:count], dtype=np.int64)

    def ensure_pages(self, req, tokens):
        need = (tokens + PAGE - 1) // PAGE
        table = self.tables.setdefault(req, [])
        while len(table) < need:
            table.append(int(self.free.pop()))

    def write(self, req, start, count):
        self.ensure_pages(req, start + count)
        pos = np.arange(start, start + count)
        slots = torch.as_tensor(np.asarray(self.tables[req])[pos // PAGE] * PAGE + pos % PAGE)
        self.kc.index_copy_(0, slots, torch.randn(count, KV, HEAD, generator=self.gen).to(torch.bfloat16))
        self.vc.index_copy_(0, slots, torch.randn(count, KV, HEAD, generator=self.gen).to(torch.bfloat16))

    def buffers(self, req, start, bucket):
        pages = (start + WIDTH + PAGE - 1) // PAGE
        block_list = torch.full((bucket,), PAD, dtype=torch.long)
        block_list[:pages] = torch.as_tensor(self.tables[req][:pages])
        groups = torch.full((bucket,), -1, dtype=torch.long)
        groups[:pages] = 0
        visible = start + torch.arange(WIDTH).view(1, -1) + 1 - torch.arange(bucket).view(-1, 1) * PAGE
        usage = visible.clamp(0, PAGE)
        bias = torch.zeros(bucket, WIDTH, PAGE, dtype=torch.bfloat16)
        bias.masked_fill_(torch.arange(PAGE).view(1, 1, -1) >= usage.unsqueeze(-1), -math.inf)
        return block_list, bias, groups

    def decode(self, req, start, bucket, check=True):
        """One executed single-request round: write, plan, attend (mirror vs gather)."""
        self.write(req, start, WIDTH)
        tail = self.planner.plan([req], [start], 1, bucket, self.table)
        block_list, bias, groups = self.buffers(req, start, bucket)
        q = torch.randn(1, WIDTH, KV * GROUP * HEAD, generator=self.gen).to(torch.bfloat16)
        ref = dense.dense_query_block_attention(q, self.kc, self.vc, block_list, bias, groups, PAGE, SCALE)
        if tail is None:
            return None
        km, vm = self.impl._kv_mirror
        out = dense.dense_query_block_attention(q, self.kc, self.vc, block_list, bias, groups, PAGE, SCALE,
                                                mirror=(km, vm, torch.as_tensor(tail)))
        if check:
            assert torch.equal(out, ref), f"mirror differs from gather at start {start}"
        return tail


def bucket_for(start):
    pages = (start + WIDTH + PAGE - 1) // PAGE
    return (pages + 7) // 8 * 8


def test_tail_pages():
    assert list(mirror_mod.tail_pages(0, 16, 8)) == [0, 1]
    assert list(mirror_mod.tail_pages(20, 16, 8)) == [1, 2]
    assert list(mirror_mod.tail_pages(120, 16, 8)) == [7, 7]  # clamped into the bucket


def test_single_request_rounds_match_gather():
    sim = Sim()
    sim.write("a", 0, 300)  # prefill
    sim.planner.note_write("a", 0)
    start, rng = 300, np.random.default_rng(1)
    for _ in range(120):
        assert sim.decode("a", start, bucket_for(start)) is not None
        start += int(rng.integers(1, WIDTH + 1))
    # one refresh (the whole context on the first mirror round), then only tail copies
    assert sim.planner.stats["refreshes"] == 1
    assert sim.planner.stats["refresh_pages"] == 300 // PAGE


def test_two_request_rounds_invalidate_from_their_start():
    sim = Sim()
    sim.write("a", 0, 200)
    start = 200
    for _ in range(10):
        sim.decode("a", start, bucket_for(start))
        start += 5
    # three two-request rounds: "a" writes without the mirror
    for _ in range(3):
        sim.write("a", start, WIDTH)
        assert sim.planner.plan(["a", "b"], [start, 50], 2, 64, sim.table) is None
        start += 7
    before = sim.planner.stats["refresh_pages"]
    for _ in range(10):
        sim.decode("a", start, bucket_for(start))
        start += 6
    copied = sim.planner.stats["refresh_pages"] - before
    assert 0 < copied <= 3  # only the pages written by the two-request rounds


def test_small_and_large_buckets_use_gather():
    sim = Sim(mirror_pages=32, min_pages=8)
    sim.write("a", 0, 40)
    assert sim.decode("a", 40, 4) is None  # below min_pages
    sim.write("a", 48, 500)
    assert sim.decode("a", 548, 40) is None  # beyond the mirror
    assert sim.decode("a", 100, 16) is not None  # in range again: refreshed


def test_reallocated_pages_and_new_request():
    sim = Sim()
    sim.write("a", 0, 160)
    start = 160
    for _ in range(5):
        sim.decode("a", start, bucket_for(start))
        start += 8
    # the scheduler moves page 3 of "a" (same tokens, new physical page)
    old = sim.tables["a"][3]
    new = int(sim.free.pop())
    sim.kc.view(-1, PAGE, KV, HEAD)[new] = sim.kc.view(-1, PAGE, KV, HEAD)[old]
    sim.vc.view(-1, PAGE, KV, HEAD)[new] = sim.vc.view(-1, PAGE, KV, HEAD)[old]
    sim.kc.view(-1, PAGE, KV, HEAD)[old] = 0
    sim.vc.view(-1, PAGE, KV, HEAD)[old] = 0
    sim.tables["a"][3] = new
    before = sim.planner.stats["refresh_pages"]
    sim.decode("a", start, bucket_for(start))
    assert sim.planner.stats["refresh_pages"] - before == start // PAGE - 3
    # another request takes over the mirror
    sim.write("b", 0, 90)
    sim.decode("b", 90, bucket_for(90))
    assert sim.planner.owner == "b"


def test_prefill_rewrite_and_discarded_round():
    sim = Sim()
    sim.write("a", 0, 128)
    start = 128
    for _ in range(4):
        sim.decode("a", start, bucket_for(start))
        start += 8
    # a planned (and tail-copied) round that never runs: the next real round starts at the same place
    sim.planner.plan(["a"], [start], 1, bucket_for(start), sim.table)
    sim.decode("a", start, bucket_for(start))
    start += 8
    # recompute from token 32 (e.g. after preemption) without the mirror
    sim.write("a", 32, start - 32)
    sim.planner.note_write("a", 32)
    for _ in range(4):
        sim.decode("a", start, bucket_for(start))
        start += 3


def test_refresh_chunks(monkeypatch):
    monkeypatch.setattr(mirror_mod, "REFRESH_CHUNK", 5)
    sim = Sim()
    sim.write("a", 0, 333)
    sim.decode("a", 333, bucket_for(333))
    assert sim.planner.stats["refresh_pages"] == 333 // PAGE


def test_warmup_plans_without_state():
    sim = Sim()
    tail = sim.planner.plan(["w"], [100], 1, 16, sim.table, warmup=True)
    assert list(tail) == [6, 7]
    assert sim.planner.owner is None and sim.planner.stats["refreshes"] == 0
