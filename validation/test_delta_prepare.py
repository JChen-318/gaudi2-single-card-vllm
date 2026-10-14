"""DeltaPrepare against a reference 'full prepare' built from decode_shared_prefix (the v4 builders)."""
import sys
import types
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import v6_support as v6_local_test  # noqa: E402

utils = types.ModuleType("vllm_gaudi.utils")
utils.async_h2d_copy = lambda source, device=None, **_: source.clone()
sys.modules.setdefault("vllm_gaudi", types.ModuleType("vllm_gaudi"))
sys.modules["vllm_gaudi.utils"] = utils
kiro = v6_local_test._load("v1/worker/kiro_v6.py", "kiro_v6_delta_prepare")
dsp = v6_local_test._load("v1/worker/decode_shared_prefix.py", "decode_shared_prefix_v8_test")

T = 8
BS = 128
PAD = 9999
GROUPS = [0, 2, 3, 5]  # attention groups (first = 0); other kv groups are mamba


@dataclass
class Meta:
    slot_mapping: torch.Tensor
    block_list: torch.Tensor
    block_groups: torch.Tensor
    block_usage: torch.Tensor
    num_accepted_tokens: torch.Tensor
    seq_lens_tensor: torch.Tensor
    load_indices_tensor: torch.Tensor
    direct_gdn_state: bool = True
    attention_groups: tuple | None = None


@dataclass
class Data:
    num_decodes: int
    token_ids: torch.Tensor
    position_ids: torch.Tensor
    attn_metadata: Meta
    logits_indices: torch.Tensor
    spec_decode_metadata: object


class Table:

    def __init__(self, cpu):
        self.cpu = cpu

    def get_cpu_tensor(self):
        return self.cpu


def _runner(req_ids, seed):
    g = torch.Generator().manual_seed(seed)
    batch = SimpleNamespace(req_ids=list(req_ids), req_id_to_index={r: i for i, r in enumerate(req_ids)},
                            block_table={gid: Table(torch.randperm(5000, generator=g)[:len(req_ids) * 40]
                                                    .reshape(len(req_ids), 40).to(torch.int32)) for gid in range(6)})
    return SimpleNamespace(input_batch=batch, attn_block_size=BS, device="cpu", use_async_scheduling=False,
                           _legacy_attention_group_ids=tuple(GROUPS), _get_attention_group_id_for_hybrid=lambda: 0,
                           _shared_prefix_decode_enabled=lambda: True, defragmenter=None,
                           _gdn_num_accepted_by_req={r: 1 for r in req_ids})


def _full(runner, req_ids, starts, lasts):
    """What the runner's prepare produces for full T-token blocks (shared prefix, shared masks)."""
    batch = runner.input_batch
    count = len(req_ids)
    lengths = [T] * count
    rows = [batch.req_id_to_index[r] for r in req_ids]

    def table(gid):
        return batch.block_table[gid].get_cpu_tensor().numpy()[rows]

    def bucket(total):
        return ((total + 15) // 16) * 16

    blocks, owners, usage = dsp.shared_prefix_buffers(starts, lengths, T, BS, table(0), PAD, bucket, torch.bfloat16)
    main = Meta(slot_mapping=dsp.group_slots(starts, lengths, count, T, BS, table(0), PAD),
                block_list=blocks, block_groups=owners, block_usage=usage,
                num_accepted_tokens=torch.tensor([runner._gdn_num_accepted_by_req[r] for r in req_ids],
                                                 dtype=torch.int32),
                seq_lens_tensor=torch.full((count, ), T, dtype=torch.int32),
                load_indices_tensor=torch.arange(count * T, dtype=torch.int32))
    groups = [None] * 6
    groups[0] = replace(main, attention_groups=None)
    for gid in GROUPS[1:]:
        gb, _, _ = dsp.shared_prefix_buffers(starts, lengths, T, BS, table(gid), PAD, bucket, torch.bfloat16)
        groups[gid] = replace(main, slot_mapping=dsp.group_slots(starts, lengths, count, T, BS, table(gid), PAD),
                              block_list=gb, attention_groups=None)
    main.attention_groups = tuple(groups)
    positions = torch.tensor([s + t for s in starts for t in range(T)], dtype=torch.int64).reshape(-1, 1)
    tokens = torch.zeros(count * T, 1, dtype=torch.int32)
    tokens[torch.arange(count) * T, 0] = torch.tensor(lasts, dtype=torch.int32)
    spec = SimpleNamespace(num_draft_tokens=[T - 1] * count, cu_num_draft_tokens=torch.arange(1, count + 1) * (T - 1),
                           cu_num_sampled_tokens=torch.arange(1, count + 1) * T,
                           target_logits_indices=torch.arange(count * (T - 1)), bonus_logits_indices=torch.arange(count),
                           logits_indices=torch.arange(count * T), draft_token_ids=torch.zeros(count * (T - 1), 1))
    return Data(num_decodes=count, token_ids=tokens, position_ids=positions, attn_metadata=main,
                logits_indices=torch.arange(count * T, dtype=torch.int32), spec_decode_metadata=spec)


@pytest.mark.parametrize("req_ids,starts0", [(["a"], [100]), (["a", "b"], [1000, 300]), (["a"], [120])])
def test_delta_matches_full_over_rounds(req_ids, starts0):
    runner = _runner(req_ids, 1)
    delta = kiro.DeltaPrepare(runner, verify=10**6)
    starts = list(starts0)
    g = np.random.default_rng(0)
    deltas = fulls = 0
    for round_index in range(120):
        lasts = [int(x) for x in g.integers(0, 1000, len(req_ids))]
        for r, n in zip(req_ids, g.integers(1, T + 1, len(req_ids))):
            runner._gdn_num_accepted_by_req[r] = int(n)
        built = delta.build(req_ids, starts, lasts, T)
        full = _full(runner, req_ids, starts, lasts)
        if built is None:
            fulls += 1
            data = full
        else:
            deltas += 1
            assert delta.check(built, full), f"round {round_index}"
            data = built
            # groups share the main group's usage tensor
            for gid in GROUPS[1:]:
                assert data.attn_metadata.attention_groups[gid].block_usage is data.attn_metadata.block_usage
        delta.remember(req_ids, starts, T, data)
        starts = [s + int(n) for s, n in zip(starts, g.integers(1, T + 1, len(req_ids)))]
    assert not delta.disabled and deltas > 80 and fulls >= 1


def test_page_table_change_forces_full():
    runner = _runner(["a"], 2)
    delta = kiro.DeltaPrepare(runner, verify=0)
    full = _full(runner, ["a"], [10], [5])
    delta.remember(["a"], [10], T, full)
    assert delta.build(["a"], [12], [6], T) is not None
    runner.input_batch.block_table[3].cpu[0, 0] += 1  # a group's first page changed
    assert delta.build(["a"], [12], [6], T) is None


def test_check_detects_difference():
    runner = _runner(["a"], 3)
    delta = kiro.DeltaPrepare(runner, verify=5)
    full = _full(runner, ["a"], [10], [5])
    delta.remember(["a"], [10], T, full)
    built = delta.build(["a"], [12], [6], T)
    wrong = _full(runner, ["a"], [13], [6])
    assert not delta.check(built, wrong) and delta.disabled


def test_mrope_positions():
    runner = _runner(["a", "b"], 4)
    runner.requests = {"a": SimpleNamespace(mrope_position_delta=-37), "b": SimpleNamespace(mrope_position_delta=5)}

    def full_mrope(starts, lasts):
        data = _full(runner, ["a", "b"], starts, lasts)
        shifts = torch.tensor([-37] * T + [5] * T)
        data.position_ids = (data.position_ids.reshape(-1) + shifts).to(torch.int32).reshape(1, -1).expand(3, -1).contiguous()
        return data

    delta = kiro.DeltaPrepare(runner, verify=100)
    delta.remember(["a", "b"], [200, 900], T, full_mrope([200, 900], [1, 2]))
    built = delta.build(["a", "b"], [203, 905], [3, 4], T)
    assert built is not None and built.position_ids.shape == (3, 2 * T)
    assert delta.check(built, full_mrope([203, 905], [3, 4]))
