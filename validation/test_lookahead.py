"""Fake-runner checks for the B<=2 lookahead: launch bookkeeping, full reuse, subset reuse, drop, mismatch."""
import sys
import types
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import v6_support as v6_local_test  # noqa: E402

kiro = v6_local_test._load("v1/worker/kiro_v6.py", "kiro_v6_lookahead")

T = 8
K = 7
H = 4
V = 11


class Table:

    def __init__(self, rows):
        self.cpu = torch.arange(rows * 16, dtype=torch.int32).reshape(rows, 16)
        self.num_blocks_per_row = np.array([16] * rows, dtype=np.int32)

    def get_cpu_tensor(self):
        return self.cpu


class Batch:

    def __init__(self, req_ids, computed):
        n = len(req_ids)
        self.req_ids = list(req_ids)
        self.req_id_to_index = {r: i for i, r in enumerate(req_ids)}
        self.num_reqs = n
        self.num_computed_tokens_cpu = np.array(computed, dtype=np.int64)
        self.token_ids_cpu = np.zeros((n, 4096), dtype=np.int64)
        self.num_tokens_no_spec = np.array(computed, dtype=np.int64) + 1
        self.num_tokens = self.num_tokens_no_spec.copy()
        self.block_table = [Table(n)]


class Runner:

    def __init__(self, req_ids, computed):
        self.input_batch = Batch(req_ids, computed)
        self.requests = {r: SimpleNamespace(sampling_params=SimpleNamespace(max_tokens=10000), output_token_ids=[1])
                         for r in req_ids}
        self.speculative_config = SimpleNamespace(num_speculative_tokens=K)
        self.max_model_len = 100000
        self.attn_block_size = 128
        self._step_timer = None
        self._v6_control = SimpleNamespace(active={"lookahead": True, "fast_topk": True, "presample": True})
        self.kv_caches = None
        self.prepared = []

    def _get_attention_group_id_for_hybrid(self):
        return 0

    def _prepare_inputs(self, predicted, prefills, count):
        b = self.input_batch
        self.prepared.append((dict(predicted.num_scheduled_tokens), b.num_computed_tokens_cpu.copy()))
        tokens = torch.zeros(count * T, 1, dtype=torch.int32)
        spec = SimpleNamespace(target_logits_indices=torch.arange(count * T))
        return None, (SimpleNamespace(token_ids=tokens, position_ids=None, attn_metadata=None, logits_indices=None,
                                      spec_decode_metadata=spec), None)

    def _execute_model_generic(self, token_ids, *args, **kwargs):
        rows = token_ids.shape[0]
        base = token_ids.float().repeat(1, H)
        return base, [base + 1], base + 2, token_ids.float().repeat(1, V), None

    def _dflash_lookahead_sample(self, decode_data, logits, req_ids):
        count = decode_data.token_ids.shape[0] // T
        return torch.arange(count, dtype=torch.int32).unsqueeze(1).repeat(1, T + 1)


def _launch(req_ids, computed, accepted):
    runner = Runner(req_ids, computed)
    la = kiro.DFlashLookahead(runner, presample=True)
    la._export_drafts = lambda drafts: setattr(la, "_drafts_cpu", drafts.clone()) or setattr(
        la, "_drafts_event", SimpleNamespace(synchronize=lambda: None))
    b = runner.input_batch
    sampled = []
    for row, n in enumerate(accepted):
        b.num_tokens_no_spec[row] = computed[row] + n + 1
        b.num_tokens[row] = b.num_tokens_no_spec[row]
        b.token_ids_cpu[row, computed[row] + n] = 500 + row
        sampled.append([0] * (n - 1) + [500 + row])
    runner._draft_token_ids = torch.arange(len(req_ids) * K, dtype=torch.int32).reshape(len(req_ids), K) + 100
    before = b.token_ids_cpu.copy(), b.num_computed_tokens_cpu.copy(), b.num_tokens.copy()
    la.maybe_launch(len(req_ids), 0, sampled, False, False)
    assert la.pending is not None
    assert np.array_equal(before[0], b.token_ids_cpu) and np.array_equal(before[1], b.num_computed_tokens_cpu)
    assert np.array_equal(before[2], b.num_tokens)
    # predicted computed tokens were visible to the prepare
    assert list(runner.prepared[0][1]) == [c + n for c, n in zip(computed, accepted)]
    drafts = la.take_drafts()
    assert drafts == runner._draft_token_ids.tolist()
    return runner, la


def _advance(runner, accepted, keep):
    """Simulate the scheduler's next step for the kept requests (in order)."""
    old = runner.input_batch
    computed = {r: int(old.num_computed_tokens_cpu[i]) + accepted[i] for i, r in enumerate(old.req_ids)}
    new = Batch(keep, [computed[r] for r in keep])
    for i, r in enumerate(keep):
        new.token_ids_cpu[i, computed[r]] = 500 + old.req_id_to_index[r]
        new.block_table[0].cpu[i] = old.block_table[0].cpu[old.req_id_to_index[r]]
    for r in list(runner.requests):
        if r not in keep:
            del runner.requests[r]
    runner.input_batch = new


def _sched(la, keep):
    return SimpleNamespace(num_scheduled_tokens={r: T for r in keep},
                           scheduled_spec_decode_tokens={r: la_drafts(la, r) for r in keep})


def la_drafts(la, r):
    return la.pending["drafts_host"][r] if la.pending else None


def test_full_reuse_pair():
    runner, la = _launch(["a", "b"], [100, 300], [3, 5])
    _advance(runner, [3, 5], ["a", "b"])
    sched = _sched(la, ["a", "b"])
    data = SimpleNamespace(token_ids=torch.zeros(2 * T, 1))
    out = la.take(sched, data, 2, [])
    assert out is not None and la.stats["subset"] == 0 and la.stats["mismatch"] == 0
    outputs, sampled = out
    assert outputs[3].shape == (2 * T, V) and sampled.shape == (2, T + 1)


def test_subset_when_first_request_finishes():
    runner, la = _launch(["a", "b"], [100, 300], [3, 5])
    full = la.pending["outputs"]
    _advance(runner, [3, 5], ["b"])
    sched = _sched(la, ["b"])
    data = SimpleNamespace(token_ids=torch.zeros(T, 1))
    outputs, sampled = la.take(sched, data, 1, [])
    assert la.stats["subset"] == 1
    assert torch.equal(outputs[0], full[0][T:2 * T]) and torch.equal(outputs[1][0], full[1][0][T:2 * T])
    assert torch.equal(outputs[3], full[3][T:2 * T])
    assert sampled.tolist() == [[1] * (T + 1)]


def test_drop_when_all_finished_and_keep_when_unscheduled():
    runner, la = _launch(["a"], [100], [4])
    pending = la.pending
    _advance(runner, [4], ["a"])
    assert la.take(_sched(la, []), SimpleNamespace(token_ids=torch.zeros(T, 1)), 0, []) is None
    assert la.pending is pending  # kept: not scheduled as a decode
    del runner.requests["a"]
    assert la.take(_sched(la, []), SimpleNamespace(token_ids=torch.zeros(T, 1)), 0, []) is None
    assert la.pending is None and la.stats["dropped"] == 1


def test_mismatch_disables():
    runner, la = _launch(["a"], [100], [4])
    _advance(runner, [3], ["a"])  # scheduler saw a different acceptance
    out = la.take(_sched(la, ["a"]), SimpleNamespace(token_ids=torch.zeros(T, 1)), 1, [])
    assert out is None and la.disabled and la.stats["mismatch"] == 1


def test_no_launch_near_max_tokens_or_with_prefill():
    runner = Runner(["a"], [100])
    la = kiro.DFlashLookahead(runner)
    runner.requests["a"].sampling_params.max_tokens = 5
    runner._draft_token_ids = torch.zeros(1, K, dtype=torch.int32)
    runner.input_batch.num_tokens_no_spec[0] = 104
    la.maybe_launch(1, 0, [[1, 2, 3]], False, False)
    assert la.pending is None
    la.maybe_launch(1, 1, [[1, 2, 3]], False, False)
    assert la.pending is None


def test_peek_full_reuse_requires_presample_and_same_rows():
    runner, la = _launch(["a", "b"], [100, 300], [3, 5])
    la.pending["decode_data"] = "DD"
    _advance(runner, [3, 5], ["a", "b"])
    sched = _sched(la, ["a", "b"])
    assert la.peek_full_reuse(sched, 2, []) == "DD"
    la.pending["sampled"] = None
    assert la.peek_full_reuse(sched, 2, []) is None
    runner2, la2 = _launch(["a", "b"], [100, 300], [3, 5])
    la2.pending["decode_data"] = "DD"
    _advance(runner2, [3, 5], ["b"])
    assert la2.peek_full_reuse(_sched(la2, ["b"]), 1, []) is None  # subset: prepare normally


def test_stage_before_drafts_then_launch():
    runner = Runner(["a"], [100])
    la = kiro.DFlashLookahead(runner, presample=True)
    la._export_drafts = lambda drafts: setattr(la, "_drafts_cpu", drafts.clone()) or setattr(
        la, "_drafts_event", SimpleNamespace(synchronize=lambda: None))
    b = runner.input_batch
    b.num_tokens_no_spec[0] = 100 + 3 + 1
    b.token_ids_cpu[0, 103] = 500
    runner._draft_token_ids = None  # the drafter has not run yet
    staged = la.stage(1, 0, [[0, 0, 500]], False, False)
    assert staged is not None and la.pending is None and len(runner.prepared) == 1
    runner._draft_token_ids = torch.arange(K, dtype=torch.int32).reshape(1, K) + 100
    la.maybe_launch(1, 0, [[0, 0, 500]], False, False)
    assert la.pending is not None and len(runner.prepared) == 1  # staged prepare reused
    assert la.pending["outputs"][0].shape[0] == T
