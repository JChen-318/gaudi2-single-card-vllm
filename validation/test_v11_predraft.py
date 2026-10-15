"""DFlash2 pre-draft (VLLM_HPU_DFLASH2_PREDRAFT).

1. ``predraft_meta`` host inputs.
2. The device bridge fed by ``predraft_meta`` addresses exactly the same context and query
   KV slots as the host drafter path for every accepted length (needs FlashInfer-Gaudi's
   ``dflash2.py``: set ``FLASHINFER_DFLASH2_PY``, otherwise skipped).
3. Lookahead bookkeeping: the pre-draft rides with the pending round, survives a subset,
   is consumed only by a pure decode step, and the verifier can switch it off.
"""
import itertools
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import v6_support  # noqa: E402

kiro11 = v6_support._load("v1/worker/kiro_v11.py", "vllm_gaudi.v1.worker.kiro_v11")
v6_support._load("v1/spec_decode/dflash_host_prepare.py", "vllm_gaudi.v1.spec_decode.dflash_host_prepare")
cpu_inputs = v6_support._load("v1/spec_decode/dflash_cpu_inputs.py", "vllm_gaudi.v1.spec_decode.dflash_cpu_inputs")
fast_topk = v6_support._load("v1/sample/fast_topk_topp.py", "vllm_gaudi.v1.sample.fast_topk_topp")
sys.modules["vllm_gaudi.v1.sample"].fast_topk_topp = fast_topk
import test_lookahead as tl  # noqa: E402

DFLASH2 = os.environ.get("FLASHINFER_DFLASH2_PY", "")
BS, WIDTH, MASK, PAD_BLOCK, PAD_SLOT = 16, 8, 99, 0, 0


# ---- 1. host inputs ---------------------------------------------------------------------------
def test_meta_rows_and_static_page_count():
    meta, blocks = kiro11.predraft_meta([10, 300], 2, WIDTH, BS, 64, 40)
    assert meta[0].tolist() == [10, 300] and meta[1].tolist() == [1, 1]
    # first page from the earliest query start (start + 1) minus the window radius
    assert meta[2].tolist() == [0, (300 + 1 - 63) // BS]
    assert meta[3].tolist() == [(10 + 7) // BS, (300 + 7) // BS]
    assert blocks == max(meta[3][i] - meta[2][i] + 1 for i in range(2))
    padded, _ = kiro11.predraft_meta([10], 2, WIDTH, BS, None, 40)
    assert padded[:, 1].tolist() == [0, 0, 0, -1]  # inactive padding row
    _, bucketed = kiro11.predraft_meta([300], 1, WIDTH, BS, None, 40, cpu_inputs.bucket_context_blocks)
    assert bucketed == 32  # 19 pages -> bucket 32
    _, capped = kiro11.predraft_meta([3000], 1, WIDTH, BS, None, 40, cpu_inputs.bucket_context_blocks)
    assert capped == 40  # never wider than the table


# ---- 2. bridge equivalence --------------------------------------------------------------------
def _device_bridge():
    if not DFLASH2 or not Path(DFLASH2).is_file():
        pytest.skip("FLASHINFER_DFLASH2_PY not set")
    return v6_support.extract(DFLASH2, ["_require_integer", "build_attention_bias", "prepare_device_inputs"],
                              {"torch": torch, "Sequence": list})["prepare_device_inputs"]


def _visible(block_list_row, bias_row, first_block, query):
    """{key position: physical slot} the query attends to in the paged past."""
    out = {}
    past = bias_row.shape[-1] - WIDTH
    for j in range(past):
        if bias_row[0, query, j] == 0:
            page = int(block_list_row[j // BS])
            assert page != PAD_BLOCK, "a visible past position must be resident"
            out[first_block * BS + j] = page * BS + j % BS
    return out


@pytest.mark.parametrize("window,causal,bucket", [(64, False, True), (None, False, True), (64, True, False),
                                                  (None, True, False)])
def test_bridge_matches_host_path(window, causal, bucket):
    prepare = _device_bridge()
    rng = np.random.default_rng(7)
    bucket_fn = cpu_inputs.bucket_context_blocks if bucket else None
    for starts, padded in [([5], 1), ([47], 1), ([120, 9], 2), ([300], 2), ([0, 63], 2)]:
        for counts in itertools.product(range(1, WIDTH + 1), repeat=len(starts)):
            if len(starts) == 2 and (counts[0] + counts[1]) % 3:
                continue  # keep the pair grid small
            table = torch.from_numpy(rng.permutation(np.arange(1, 1 + padded * 40)).reshape(padded, 40).astype(
                np.int32))
            sampled = torch.full((padded, WIDTH), -1, dtype=torch.int32)
            for row, count in enumerate(counts):
                sampled[row, :count] = torch.from_numpy(rng.integers(1, 1000, count).astype(np.int32))
            meta, blocks = kiro11.predraft_meta(starts, padded, WIDTH, BS, window, table.shape[1], bucket_fn)
            (accepted, ctx_pos, ctx_slots, ids, qpos, qslots, block_list, ctx_lens,
             bias) = prepare(sampled, meta[0], table, meta[1].to(torch.bool), meta[2], meta[3], block_size=BS,
                             num_query_per_req=WIDTH, mask_token_id=MASK, pad_block_id=PAD_BLOCK,
                             pad_slot_id=PAD_SLOT, max_model_len=1 << 20, max_context_blocks=blocks, window=window,
                             causal=causal, bias_dtype=torch.float32)
            rows = list(range(len(starts)))
            query_starts = [s + c for s, c in zip(starts, counts)]
            bonus = [int(sampled[r, c - 1]) for r, c in zip(rows, counts)]
            host = cpu_inputs.prepare_query_cpu_inputs(table, rows, query_starts, bonus, batch_size=padded,
                                                       query_len=WIDTH, block_size=BS, mask_token_id=MASK,
                                                       pad_block=PAD_BLOCK, pad_slot=PAD_SLOT, window=window,
                                                       causal=causal, dtype=torch.float32, bucket_blocks=bucket_fn)
            context = cpu_inputs.prepare_context_cpu_inputs(table, rows, starts, list(counts), WIDTH, BS, PAD_BLOCK,
                                                            PAD_SLOT)
            assert accepted[:len(rows)].tolist() == list(counts)
            host_first = [0 if window is None else max(0, q - (window - 1)) // BS for q in query_starts]
            host_blocks = host["block_list"].reshape(padded, -1)
            dev_blocks = block_list.reshape(padded, -1)
            ctx_pos, ctx_slots = ctx_pos.reshape(padded, WIDTH), ctx_slots.reshape(padded, WIDTH)
            offset = 0
            for row, count in zip(rows, counts):
                assert torch.equal(ids[row], host["input_ids"][row])
                assert torch.equal(qpos[row], host["positions"][row])
                assert torch.equal(qslots[row], host["slot_mapping"][row])
                for query in range(WIDTH):
                    assert (_visible(dev_blocks[row], bias[row], int(meta[2, row]), query) == _visible(
                        host_blocks[row], host["attn_bias"][row], host_first[row], query))
                assert torch.equal(bias[row, 0, :, -WIDTH:], host["attn_bias"][row, 0, :, -WIDTH:])
                # Context KV writes: accepted rows hit the host slots, rejected rows the pad block only.
                assert ctx_pos[row, :count].tolist() == context[1, offset:offset + count].tolist()
                assert ctx_slots[row, :count].tolist() == context[2, offset:offset + count].tolist()
                rejected = ctx_slots[row, count:]
                assert bool(((rejected >= PAD_SLOT) & (rejected < PAD_SLOT + BS)).all())
                offset += count


# ---- 3. lookahead bookkeeping -----------------------------------------------------------------
class PreRunner(tl.Runner):

    def __init__(self, req_ids, computed):
        super().__init__(req_ids, computed)
        self._v6_control = SimpleNamespace(active={"lookahead": True, "fast_topk": True, "presample": True,
                                                   "delta": True, "predraft": True, "guard_skip": True})
        self.plans, self.predrafts = [], []
        for name, value in (("all_greedy", False), ("all_random", True), ("no_top_k", False), ("no_top_p", False)):
            setattr(self.input_batch, name, value)

    def _dflash2_predraft_plan(self, starts, padded):
        self.plans.append((list(starts), padded))
        return torch.zeros(4, padded, dtype=torch.int64), 4

    def _dflash2_predraft(self, decode_data, outputs, sampled, starts, plan):
        assert sampled is not None and plan[1] == 4
        drafts = torch.tensor(starts, dtype=torch.int32).unsqueeze(1) + torch.arange(tl.K, dtype=torch.int32)
        self.predrafts.append(drafts)
        return drafts


def _launch(req_ids, computed, accepted, verify=0, guard_skip=None):
    original = tl.Runner
    tl.Runner = PreRunner
    try:
        runner_holder = {}

        def make(req_ids_, computed_):
            runner = PreRunner(req_ids_, computed_)
            runner._guard_skip = guard_skip
            runner_holder["r"] = runner
            return runner

        tl.Runner = make
        original_init = tl.kiro.DFlashLookahead.__init__

        def init(self, runner, presample=True, *args, **kwargs):
            original_init(self, runner, presample, predraft_verify=verify)

        tl.kiro.DFlashLookahead.__init__ = init
        try:
            runner, la = tl._launch(req_ids, computed, accepted)
        finally:
            tl.kiro.DFlashLookahead.__init__ = original_init
    finally:
        tl.Runner = original
    return runner, la


def test_launch_carries_predraft_and_full_reuse_consumes_it_once():
    runner, la = _launch(["a", "b"], [100, 300], [3, 5])
    assert runner.plans == [([103, 305], 2)] and la.stats["predrafted"] == 1
    expected = la.pending["predraft"]
    tl._advance(runner, [3, 5], ["a", "b"])
    out = la.take(tl._sched(la, ["a", "b"]), SimpleNamespace(token_ids=torch.zeros(2 * tl.T, 1)), 2, [])
    assert out is not None
    assert torch.equal(la.pop_predraft(), expected) and la.pop_predraft() is None
    assert la.stats["predraft_used"] == 1


def test_predraft_not_used_with_a_prefill_in_the_step():
    runner, la = _launch(["a"], [100], [4])
    tl._advance(runner, [4], ["a"])
    out = la.take(tl._sched(la, ["a"]), SimpleNamespace(token_ids=torch.zeros(tl.T, 1)), 1, ["new"])
    assert out is not None and la.pop_predraft() is None


def test_subset_selects_the_surviving_rows():
    runner, la = _launch(["a", "b"], [100, 300], [3, 5])
    full = la.pending["predraft"]
    tl._advance(runner, [3, 5], ["b"])
    la.take(tl._sched(la, ["b"]), SimpleNamespace(token_ids=torch.zeros(tl.T, 1)), 1, [])
    assert torch.equal(la.pop_predraft(), full[1:2])


def test_verifier_disables_on_disagreement():
    runner, la = _launch(["a"], [100], [4], verify=2)
    assert la.needs_predraft_check()
    host = torch.full((1, tl.K), 7, dtype=torch.int32)
    la.check_predraft(torch.zeros(1, tl.K, dtype=torch.int32), host)
    assert not la.predraft_disabled and la.needs_predraft_check()
    la.check_predraft(torch.zeros(1, tl.K, dtype=torch.int32), host)
    assert la.predraft_disabled and not la.needs_predraft_check()
    tl._advance(runner, [4], ["a"])
    la.take(tl._sched(la, ["a"]), SimpleNamespace(token_ids=torch.zeros(tl.T, 1)), 1, [])
    assert la.pop_predraft() is None


def test_verifier_accepts_agreement():
    verifier = kiro11.PreDraftVerifier(2)
    same = torch.arange(tl.K, dtype=torch.int32).reshape(1, tl.K)
    assert verifier.compare(same, same.clone()) and verifier.compare(same, same.clone())
    assert verifier.left == 0 and verifier.equal == 2


class RecordingSkip:

    def __init__(self):
        self.keys, self.inside = [], False

    def round(self, key):
        import contextlib
        skip = self

        @contextlib.contextmanager
        def ctx():
            skip.keys.append(key)
            skip.inside = True
            yield True
            skip.inside = False

        return ctx()


def test_forward_sampling_and_predraft_run_inside_the_guard_skip_round():
    skip = RecordingSkip()
    seen = []
    original_forward = tl.Runner._execute_model_generic
    original_sample = tl.Runner._dflash_lookahead_sample

    def forward(self, *args, **kwargs):
        seen.append(("forward", skip.inside))
        return original_forward(self, *args, **kwargs)

    def sample(self, *args, **kwargs):
        seen.append(("sample", skip.inside))
        return original_sample(self, *args, **kwargs)

    tl.Runner._execute_model_generic, tl.Runner._dflash_lookahead_sample = forward, sample
    try:
        runner, la = _launch(["a"], [100], [4], guard_skip=skip)
    finally:
        tl.Runner._execute_model_generic, tl.Runner._dflash_lookahead_sample = original_forward, original_sample
    assert seen == [("forward", True), ("sample", True)] and len(skip.keys) == 1
    key = skip.keys[0]
    assert key[2] == 4 and key[-1] is fast_topk.is_active()  # predraft pages and fast top-k state
    assert la.pending["predraft"] is not None
