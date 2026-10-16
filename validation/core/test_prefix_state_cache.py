"""GDN prefix state cache (VLLM_HPU_PREFIX_STATE_CACHE) against the real vLLM KV-cache manager.

Needs the vLLM core package (vllm-core/ of this repository or an installed vLLM) and its dependencies; no HPU.
Run with ``VLLM_PLUGINS=none`` so no platform plugin is loaded:

    VLLM_PLUGINS=none PYTHONPATH=vllm-gaudi python -m pytest -q validation/core
"""
import os
from types import SimpleNamespace

import pytest
import torch

os.environ["VLLM_HPU_PREFIX_STATE_CACHE"] = "1"
os.environ["VLLM_HPU_PREFIX_STATE_INTERVAL"] = "64"  # 4 blocks of 16

kv_cache_manager = pytest.importorskip("vllm.v1.core.kv_cache_manager")
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec  # noqa: E402
from vllm.v1.request import Request  # noqa: E402

from vllm_gaudi.v1.core import prefix_state_cache as psc  # noqa: E402

B = 16
INTERVAL = 64
init_none_hash(sha256)
psc.install()


def make_manager(num_blocks=400, use_eagle=False):
    attn = FullAttentionSpec(block_size=B, num_kv_heads=1, head_size=1, dtype=torch.float32)
    draft = FullAttentionSpec(block_size=B, num_kv_heads=2, head_size=1, dtype=torch.float32)
    mamba = MambaSpec(block_size=B, shapes=((1, 1), ), dtypes=(torch.float32, ), mamba_cache_mode="align",
                      num_speculative_blocks=7)
    config = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(["attn"], attn),
        KVCacheGroupSpec(["draft"], draft),
        KVCacheGroupSpec(["gdn0"], mamba),
        KVCacheGroupSpec(["gdn1"], mamba),
    ])
    manager = kv_cache_manager.KVCacheManager(config, max_model_len=8192, enable_caching=True, use_eagle=use_eagle,
                                              num_prefill_lookahead=int(use_eagle), scheduler_block_size=B,
                                              hash_block_size=B)
    if use_eagle:
        # what the patched Scheduler.__init__ does for DFlash2 (use_eagle() is true for it)
        stub = SimpleNamespace(has_mamba_layers=True, kv_cache_manager=manager, block_size=B, hash_block_size=B,
                               cache_config=SimpleNamespace(enable_prefix_caching=True, mamba_cache_mode="align",
                                                            block_size=B))
        psc._init_scheduler(stub)
    return manager


def make_request(req_id, tokens):
    params = SamplingParams(max_tokens=16)
    params.update_from_generation_config({}, eos_token_id=100)
    return Request(request_id=req_id, prompt_token_ids=list(tokens), mm_features=None, sampling_params=params,
                   pooling_params=None, block_hasher=get_request_block_hasher(B, sha256))


def fresh_registry(slots=8):
    psc._registry = psc.SnapshotRegistry(slots)
    return psc._registry


def stub_scheduler(requests):
    # cache_config.block_size deliberately wrong: the engine's copy can be reset after the scheduler is built
    return SimpleNamespace(block_size=B, cache_config=SimpleNamespace(block_size=128),
                           requests={r.request_id: r for r in requests})


def step_output(new=(), resumed=(), scheduled=None):
    return SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(req_id=r) for r in new],
                           scheduled_cached_reqs=SimpleNamespace(req_ids=list(resumed), resumed_req_ids=set(resumed)),
                           num_scheduled_tokens=dict(scheduled or {}))


def run_prefill(manager, scheduler, request, budget=40, first_step_new=True):
    """Schedule a request to the end of its prompt with the real split and allocator; returns the plans."""
    blocks, hit, _ = manager.get_computed_blocks(request)
    plans = []
    first = True
    while request.num_computed_tokens + (hit if first else 0) < request.num_prompt_tokens:
        start_hit = hit if first else 0
        remaining = request.num_prompt_tokens - request.num_computed_tokens - start_hit
        n = psc._aligned_split(scheduler, request, min(budget, remaining), start_hit)
        assert n > 0
        allocated = manager.allocate_slots(request, n, start_hit, blocks if first else None)
        assert allocated is not None
        if first:
            request.num_computed_tokens = start_hit
        plan = psc.plan_step(scheduler, step_output(new=[request.request_id] if first and first_step_new else (),
                                                    scheduled={request.request_id: n}))
        plans.append((request.num_computed_tokens, n, plan))
        request.num_computed_tokens += n
        first = False
    return hit, plans


def test_positions():
    assert psc.final_boundary(100, 16) == 96
    assert psc.final_boundary(96, 16) == 80
    assert psc.snapshot_positions(0, 40, 100, 16, 64) == []
    assert psc.snapshot_positions(40, 80, 100, 16, 64) == [64]
    assert psc.snapshot_positions(70, 100, 100, 16, 64) == [96]
    assert psc.is_snapshot_position(64, 100, 16, 64) and psc.is_snapshot_position(96, 100, 16, 64)
    assert not psc.is_snapshot_position(80, 100, 16, 64) and not psc.is_snapshot_position(0, 100, 16, 64)


def test_registry_lru_and_pins():
    reg = psc.SnapshotRegistry(2)
    sa = reg.acquire("a", set())
    sb = reg.acquire("b", set())
    assert reg.acquire("a", set()) is None  # kept, touched: "b" is now least recent
    sc = reg.acquire("c", set())
    assert sc == sb and "b" not in reg and "a" in reg
    assert reg.acquire("d", {sa, sc}) is None  # every slot pinned
    assert reg.hit_blocks(["x", "a", "y"], 3) == 2 and reg.hit_blocks(["x", "a", "y"], 1) == 0


def test_split_stops_at_snapshot_positions():
    sched = stub_scheduler([])
    req = make_request("r", range(150))
    # interval stop at 64, final-boundary stop at 144 (last block boundary before 150)
    assert psc._aligned_split(sched, req, 40) == 40
    req.num_computed_tokens = 40
    assert psc._aligned_split(sched, req, 40) == 24
    req.num_computed_tokens = 130
    assert psc._aligned_split(sched, req, 20) == 14
    req.num_computed_tokens = 144
    assert psc._aligned_split(sched, req, 6) == 6


@pytest.mark.parametrize("use_eagle", [False, True])
def test_multi_turn_reuse(use_eagle):
    manager = make_manager(use_eagle=use_eagle)
    reg = fresh_registry()
    turn1 = list(range(1000, 1000 + 10 * B + 5))
    a = make_request("a", turn1)
    sched = stub_scheduler([a])
    hit, plans = run_prefill(manager, sched, a)
    assert hit == 0
    saves = [(start, plan["save"]) for start, _, plan in plans if plan and plan["save"]]
    assert [s for s, _ in saves] == [64, 128, 160]  # interval, interval, final boundary of 165
    manager.free(a)

    # turn 2 appends to turn 1: resumes from the final-boundary snapshot
    b = make_request("b", turn1 + list(range(5000, 5000 + 3 * B)))
    sched.requests["b"] = b
    blocks, hit, _ = manager.get_computed_blocks(b)
    assert hit == 160
    attn, draft, gdn0, gdn1 = blocks.blocks
    assert len(attn) == len(draft) == len(gdn0) == len(gdn1) == 10
    assert all(blk.is_null for blk in gdn0 + gdn1) and not any(blk.is_null for blk in attn + draft)
    hit, plans = run_prefill(manager, sched, b)
    start, _, plan = plans[0]
    assert start == 160 and plan["restore"] == [("b", reg.slot(b.block_hashes[9]))]
    manager.free(b)

    # a prompt that diverges inside block 7 resumes from the 64-token snapshot (block 4), not later
    c_tokens = turn1[:7 * B + 3] + [7] * (4 * B)
    c = make_request("c", c_tokens)
    sched.requests["c"] = c
    _, hit, _ = manager.get_computed_blocks(c)
    assert hit == 64
    hit, plans = run_prefill(manager, sched, c)
    assert plans[0][2]["restore"][0][0] == "c"


def test_no_snapshot_means_no_hit():
    manager = make_manager()
    fresh_registry()
    tokens = list(range(2000, 2000 + 6 * B))
    a = make_request("a", tokens)
    sched = stub_scheduler([a])
    run_prefill(manager, sched, a)
    manager.free(a)
    fresh_registry()  # snapshots gone, attention blocks still cached
    b = make_request("b", tokens + [1, 2, 3])
    _, hit, _ = manager.get_computed_blocks(b)
    assert hit == 0


def test_eviction_reduces_hit():
    manager = make_manager()
    reg = fresh_registry(slots=2)
    tokens = list(range(3000, 3000 + 10 * B + 5))
    a = make_request("a", tokens)
    sched = stub_scheduler([a])
    run_prefill(manager, sched, a)  # saves 64, 128, 160 into two slots: 64 evicted
    manager.free(a)
    assert len(reg) == 2
    c = make_request("c", tokens[:7 * B + 3] + [9] * 20)
    _, hit, _ = manager.get_computed_blocks(c)
    assert hit == 0  # the 64 snapshot was evicted; 128 is past the divergence


def test_resumed_request_restores():
    reg = fresh_registry()
    a = make_request("a", range(4000, 4000 + 5 * B))
    reg.acquire(a.block_hashes[3], set())
    a.num_computed_tokens = 64
    plan = psc.plan_step(stub_scheduler([a]), step_output(resumed=["a"], scheduled={"a": 10}))
    assert plan["restore"] == [("a", reg.slot(a.block_hashes[3]))] and plan["save"] == []


def test_hit_without_snapshot_is_an_error():
    fresh_registry()
    a = make_request("a", range(5000, 5000 + 5 * B))
    a.num_computed_tokens = 32
    with pytest.raises(RuntimeError):
        psc.plan_step(stub_scheduler([a]), step_output(new=["a"], scheduled={"a": 10}))


def test_scheduler_patches_installed():
    assert Scheduler._mamba_block_aligned_split is psc._aligned_split
    assert getattr(Scheduler, "_hpu_prefix_state_installed", False)


def test_snapshot_store_copies_base_rows():
    groups, max_reqs, slots_per_req = 3, 2, 8
    rows = groups * max_reqs * slots_per_req + 2
    conv = torch.randn(rows, 5, 3)
    ssm = torch.randn(rows, 2, 4, 4)
    runner = SimpleNamespace(_compact_gdn_state_tensors=[conv, ssm], _num_gdn_groups=groups, _gdn_max_reqs=max_reqs,
                             _gdn_state_slots_per_req=slots_per_req, _gdn_req_to_base_slot={"a": 1, "b": 0})
    store = psc.GdnSnapshotStore(num_slots=4)
    base_rows = psc.GdnSnapshotStore.compact_rows(1, groups, max_reqs, slots_per_req)
    assert base_rows == [9, 25, 41]
    saved = [t[base_rows].clone() for t in (conv, ssm)]
    store.apply(runner, SimpleNamespace(hpu_prefix_state={"restore": [], "save": [("a", 2)]}))
    other = [t.clone() for t in (conv, ssm)]
    conv[base_rows] = 0
    ssm[base_rows] = 0
    store.apply(runner, SimpleNamespace(hpu_prefix_state={"restore": [("b", 2)], "save": []}))
    b_rows = psc.GdnSnapshotStore.compact_rows(0, groups, max_reqs, slots_per_req)
    for t, s, o in zip((conv, ssm), saved, other):
        assert torch.equal(t[b_rows], s)
        untouched = [r for r in range(rows) if r not in b_rows and r not in base_rows]
        assert torch.equal(t[untouched], o[untouched])


def decode(manager, request, steps, seed=0):
    """DFlash2-like rounds: 1 + 7 tokens scheduled, 0..7 drafts accepted."""
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        assert manager.allocate_slots(request, 8, num_lookahead_tokens=7) is not None
        accepted = int(torch.randint(0, 8, (1, ), generator=gen))
        request.num_computed_tokens += accepted + 1
        request.append_output_token_ids([7000 + request.num_output_tokens + i for i in range(accepted + 1)])


@pytest.mark.parametrize("use_eagle", [False, True])
def test_reuse_after_speculative_decode(use_eagle):
    manager = make_manager(use_eagle=use_eagle)
    reg = fresh_registry()
    turn1 = list(range(6000, 6000 + 9 * B + 7))
    a = make_request("a", turn1)
    sched = stub_scheduler([a])
    run_prefill(manager, sched, a)
    decode(manager, a, 40)
    output = list(a.output_token_ids)
    manager.free(a)
    # the next turn re-sends the conversation: hit at the last boundary before turn 1's prompt end
    b = make_request("b", turn1 + output + [5] * 30)
    sched.requests["b"] = b
    _, hit, _ = manager.get_computed_blocks(b)
    assert hit == psc.final_boundary(len(turn1), B) == 9 * B
    hit, plans = run_prefill(manager, sched, b)
    assert plans[0][2]["restore"] == [("b", reg.slot(b.block_hashes[8]))]
    decode(manager, b, 20, seed=1)
    manager.free(b)


def test_two_requests_share_a_step():
    """A restore and a save in one step must not pick the same slot, even with every slot in use."""
    reg = fresh_registry(slots=2)
    a = make_request("a", range(8000, 8000 + 6 * B))
    b = make_request("b", range(9000, 9000 + 9 * B))
    reg.acquire(a.block_hashes[3], set())  # a resumes at 64
    reg.acquire(("other", ), set())
    a.num_computed_tokens = 64
    b.num_computed_tokens = 64  # b reaches its interval snapshot in the same step
    plan = psc.plan_step(stub_scheduler([a, b]), step_output(new=["a"], scheduled={"a": 16, "b": 16}))
    (_, restore_slot), = plan["restore"]
    (_, save_slot), = plan["save"]
    assert restore_slot != save_slot and a.block_hashes[3] in reg
