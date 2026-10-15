"""Fused GDN core with the initial states gathered first (FLASHINFER_GAUDI_MTP_FUSED_GATHER).

The TPC kernel is replaced by a Python stand-in with the same contract: it reads row
``state_indices[b, clamp(accepted_b, 1, T) - 1]`` of the pool argument and returns the output
and the eight checkpoints. Both routes must leave the same pool and return the same output."""
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(os.environ.get("VLLM_GAUDI_SRC", Path(__file__).parent.parent / "vllm-gaudi" / "vllm_gaudi")).parent
sys.path.insert(0, str(ROOT))
# Other tests register stand-in vllm_gaudi modules; import the real package underneath them.
_stand_ins = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "vllm_gaudi" or k.startswith("vllm_gaudi.")}
try:
    gdn = pytest.importorskip("flashinfer_gaudi.gdn_decode")
finally:
    sys.modules.update(_stand_ins)

H, HV, D, T = 16, 48, 128, 8


def fake_fused(pool, packed, log_decay, beta, state_indices, accepted):
    """Stand-in for the kernel: the gated delta rule on the selected initial state."""
    batch = packed.shape[0]
    rows = state_indices[torch.arange(batch), accepted.long().clamp(1, T) - 1].long()
    state = pool[rows].clone()  # [B, HV, D, D] laid out as [v, k]
    qk = packed[..., :2 * H * D].float().reshape(batch, T, 2 * H, D)
    qk = qk / qk.norm(dim=-1, keepdim=True)
    q = (qk[:, :, :H] * D**-0.5).repeat_interleave(HV // H, dim=2)
    k = qk[:, :, H:].repeat_interleave(HV // H, dim=2)
    v = packed[..., 2 * H * D:].float().reshape(batch, T, HV, D)
    g, b = log_decay.exp(), beta.float()
    out, cps = [], []
    for t in range(T):
        state = state * g[:, t, :, None, None]
        delta = b[:, t, :, None] * (v[:, t] - torch.einsum("bhvk,bhk->bhv", state, k[:, t]))
        state = state + delta[..., None] * k[:, t, :, None, :]
        out.append(torch.einsum("bhvk,bhk->bhv", state, q[:, t]))
        cps.append(state.clone())
    return torch.stack(out, 1), torch.stack(cps, 1)


def _inputs(batch, seed):
    gen = torch.Generator().manual_seed(seed)
    pool = torch.randn(40, HV, D, D, generator=gen) * 0.05
    packed = torch.randn(batch, T, 2 * H * D + HV * D, generator=gen).to(torch.bfloat16)
    log_decay = -torch.rand(batch, T, HV, generator=gen) * 0.5
    beta = torch.rand(batch, T, HV, generator=gen).to(torch.bfloat16)
    slots = torch.randperm(39, generator=gen)[:batch * T].add(1).reshape(batch, T).to(torch.int32)
    accepted = torch.randint(0, T + 2, (batch,), generator=gen).to(torch.int32)  # includes 0 and > T
    return pool, packed, log_decay, beta, slots, accepted


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("direct", [True, False])
def test_gathered_and_in_kernel_selection_agree(monkeypatch, batch, seed, direct):
    monkeypatch.setattr(gdn, "public_mtp_fused_op", lambda: fake_fused)
    results = []
    for gather in ("1", "0"):
        monkeypatch.setenv("FLASHINFER_GAUDI_MTP_FUSED_GATHER", gather)
        pool, packed, log_decay, beta, slots, accepted = _inputs(batch, seed)
        start = 40 - batch * T if direct else None
        out, pool = gdn._gated_delta_rule_mtp_fused(packed, log_decay, beta, pool, slots, accepted,
                                                     assume_distinct_checkpoints=True, checkpoint_start=start)
        results.append((out.float(), pool))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


def test_gather_hands_the_kernel_only_the_selected_rows(monkeypatch):
    seen = {}

    def spy(pool, packed, log_decay, beta, state_indices, accepted):
        seen["pool"], seen["slots"] = pool.clone(), state_indices.clone()
        return fake_fused(pool, packed, log_decay, beta, state_indices, accepted)

    monkeypatch.setattr(gdn, "public_mtp_fused_op", lambda: spy)
    monkeypatch.setenv("FLASHINFER_GAUDI_MTP_FUSED_GATHER", "1")
    pool, packed, log_decay, beta, slots, accepted = _inputs(2, 5)
    expected = pool[slots[torch.arange(2), accepted.long().clamp(1, T) - 1].long()].clone()
    gdn._gated_delta_rule_mtp_fused(packed, log_decay, beta, pool, slots, accepted,
                                    assume_distinct_checkpoints=True, checkpoint_start=1)
    assert torch.equal(seen["pool"], expected)
    assert seen["slots"].tolist() == [[0] * T, [1] * T]
    assert seen["pool"].is_contiguous() and seen["slots"].is_contiguous()


def test_gather_is_on_by_default(monkeypatch):
    monkeypatch.delenv("FLASHINFER_GAUDI_MTP_FUSED_GATHER", raising=False)
    from flashinfer_gaudi._config import mtp_fused_gather_enabled
    assert mtp_fused_gather_enabled()
