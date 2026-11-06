# SPDX-License-Identifier: Apache-2.0
"""Grouped decode mix matches the per-expert loop, including duplicate slots.

The grouped path is the Strata launch structure (one GEMM per projection)
with the same NVFP4→BF16 weights the loop uses. It must not change the mix.

Run:
  VLLM_HW=cpu python tools/qwen4exp_flashnext/test_moe_grouped.py
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_HW", "cpu")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vllm-gaudi"))

import torch  # noqa: E402

from vllm_gaudi.models.qwen4exp.packed_moe import (  # noqa: E402
    apply_grouped_slots,
    apply_routed_experts,
    apply_routed_experts_grouped,
    compact_expert_ids,
    use_grouped_slots,
)


@contextlib.contextmanager
def _compile_only_simulated():
    """Simulate PT_COMPILE_ONLY_MODE on CPU (checked per call, not cached)."""
    previous = os.environ.get("PT_COMPILE_ONLY_MODE")
    os.environ["PT_COMPILE_ONLY_MODE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PT_COMPILE_ONLY_MODE", None)
        else:
            os.environ["PT_COMPILE_ONLY_MODE"] = previous


def _mix(hidden, ids, weights, gate, up, down, grouped: bool) -> torch.Tensor:
    if not grouped:
        # Compact ids the way the prefill path does, so duplicate slots share
        # one weight row and their mix weights add.
        unique = torch.unique(ids)
        local = torch.searchsorted(unique, ids.reshape(-1)).reshape(ids.shape).to(torch.int32)
        return apply_routed_experts(
            hidden, local, weights, gate[unique], up[unique], down[unique]
        )
    tokens, k = ids.shape
    return apply_grouped_slots(
        hidden,
        weights,
        gate[ids].view(tokens, k, gate.shape[1], gate.shape[2]),
        up[ids].view(tokens, k, up.shape[1], up.shape[2]),
        down[ids].view(tokens, k, down.shape[1], down.shape[2]),
    )


def main() -> int:
    gen = torch.Generator().manual_seed(3)
    hidden_size, intermediate, experts, k = 32, 16, 8, 4
    gate = torch.randn(experts, intermediate, hidden_size, generator=gen)
    up = torch.randn(experts, intermediate, hidden_size, generator=gen)
    down = torch.randn(experts, hidden_size, intermediate, generator=gen)
    for tokens, dtype, repeats in ((1, torch.float32, False), (3, torch.float32, True),
                                   (1, torch.bfloat16, True)):
        hidden = torch.randn(tokens, hidden_size, generator=gen).to(dtype)
        ids = torch.randint(0, experts, (tokens, k), generator=gen)
        if repeats:
            ids[:, 1] = ids[:, 0]
        weights = torch.softmax(torch.randn(tokens, k, generator=gen), dim=-1).to(dtype)
        g, u, d = gate.to(dtype), up.to(dtype), down.to(dtype)
        loop = _mix(hidden, ids, weights, g, u, d, grouped=False)
        grouped = _mix(hidden, ids, weights, g, u, d, grouped=True)
        gap = (loop.float() - grouped.float()).abs().max().item()
        scale = max(loop.float().abs().max().item(), 1e-6)
        rel = gap / scale
        cos = torch.nn.functional.cosine_similarity(
            loop.float().reshape(1, -1), grouped.float().reshape(1, -1)
        ).item()
        # Float32 is the same dots. BF16 stacked GEMM can differ in the last
        # bits from K separate dots; that is rounding, not a dropped scale.
        limit = 1e-4 if dtype == torch.float32 else 2e-2
        assert rel <= limit and cos > 0.999, f"T={tokens} {dtype} rel {rel} cos {cos}"
        print(f"  T={tokens} {dtype} duplicate={repeats} rel {rel:.3e} cos {cos:.8f}")
    # Regression: compile-only / graph-capture warmup must not read
    # int(present.sum()) (uninitialized storage there). The static fallback
    # selects every expert (present first, ascending), which must leave the
    # local index of every routed slot -- and the final MoE output -- exact.
    for ids in (
            torch.tensor([[1, 3], [0, 1]], dtype=torch.int32),
            torch.tensor([[4, 4], [4, 4]], dtype=torch.int32),
            torch.tensor([[7, 0], [3, 5], [6, 2], [1, 4]], dtype=torch.int32),
            torch.tensor([[2], [2], [7]], dtype=torch.int32),
    ):
        tokens = ids.shape[0]
        case_hidden = torch.randn(tokens, hidden_size, generator=gen)
        case_weights = torch.softmax(torch.randn(tokens, ids.shape[1], generator=gen), dim=-1)
        selected, local = compact_expert_ids(ids, experts)
        assert torch.equal(selected[local], ids), "exact local does not point at its expert"
        out_loop = apply_routed_experts(
            case_hidden, local, case_weights, gate[selected.long()], up[selected.long()],
            down[selected.long()])
        out_group = apply_routed_experts_grouped(
            case_hidden, local, case_weights, gate[selected.long()], up[selected.long()],
            down[selected.long()])
        out_direct = apply_routed_experts_grouped(case_hidden, ids, case_weights, gate, up, down)
        with _compile_only_simulated():
            selected_s, local_s = compact_expert_ids(ids, experts)
            out_loop_s = apply_routed_experts(
                case_hidden, local_s, case_weights, gate[selected_s.long()], up[selected_s.long()],
                down[selected_s.long()])
            out_group_s = apply_routed_experts_grouped(
                case_hidden, local_s, case_weights, gate[selected_s.long()], up[selected_s.long()],
                down[selected_s.long()])
            # Direct grouped call also exercises the static bucket cap
            # (cap = slots instead of int(counts.max())).
            out_direct_s = apply_routed_experts_grouped(
                case_hidden, ids, case_weights, gate, up, down)
        absent = [e for e in range(experts) if e not in selected.tolist()]
        assert selected_s.shape == (experts,), f"static selected {tuple(selected_s.shape)}"
        assert torch.equal(selected_s[:selected.numel()], selected), "present prefix changed"
        assert selected_s[selected.numel():].tolist() == absent, "absent tail not ascending"
        assert torch.equal(local_s, local), "static local changed a routed slot's expert"
        assert torch.equal(selected_s[local_s], ids), "static local does not point at its expert"
        for name, exact, static in (("loop", out_loop, out_loop_s), ("grouped", out_group, out_group_s),
                                    ("grouped-direct", out_direct, out_direct_s)):
            # Batched-GEMM tiling can change with U/cap, so the grouped
            # comparisons use the float32 relative tolerance of the loop
            # test above rather than bit equality.
            gap = (exact - static).abs().max().item()
            rel = gap / max(exact.abs().max().item(), 1e-6)
            assert rel <= 1e-4, f"static {name} output drifted: rel {rel}"
        print(f"  compact static==exact ids={ids.reshape(-1).tolist()} absent={absent}")
    assert use_grouped_slots(1, 10)
    assert use_grouped_slots(8, 10)  # 80 slots: the decode ceiling after H9(a)
    assert not use_grouped_slots(32, 10)  # T=32 prefill: 320 slots, not decode
    assert not use_grouped_slots(1, 81)
    print("grouped mix matches the per-expert loop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
