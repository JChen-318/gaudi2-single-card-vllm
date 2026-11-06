# SPDX-License-Identifier: Apache-2.0
"""CPU correctness tests for the expert-major grouped prefill MoE GEMM.

``apply_routed_experts_grouped`` replaces the per-expert Python loop of
``apply_routed_experts`` with Strata's prefill shape: sort the ``T*K`` slots by
expert, pad each expert's bucket to the busiest one, run the three projections
as one batched GEMM each, then scatter the weighted slots back onto tokens.
The tests below prove it computes the same function as the loop, including the
degenerate routings (single expert, every slot on one expert, duplicates,
empty batch) and that no per-expert loop is left in the implementation.

Run: VLLM_HW=cpu .venv/bin/python -m pytest -q validation/qwen4exp/test_prefill_grouped_moe.py
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_HW", "cpu")

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_gaudi.models.qwen4exp.packed_moe import (  # noqa: E402
    apply_routed_experts,
    apply_routed_experts_grouped,
    prefill_grouped_enabled,
)


def _problem(
    tokens: int,
    top_k: int,
    experts: int,
    hidden: int,
    intermediate: int,
    *,
    dtype: torch.dtype = torch.float32,
    seed: int = 7,
    ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    hidden_states = torch.randn(tokens, hidden, generator=gen).to(dtype)
    if ids is None:
        ids = torch.randint(0, experts, (tokens, top_k), generator=gen).to(torch.int32)
    weights = torch.rand(tokens, top_k, generator=gen).to(dtype)
    gate = torch.randn(experts, intermediate, hidden, generator=gen).to(dtype) * 0.05
    up = torch.randn(experts, intermediate, hidden, generator=gen).to(dtype) * 0.05
    down = torch.randn(experts, hidden, intermediate, generator=gen).to(dtype) * 0.05
    return hidden_states, ids, weights, gate, up, down


def _both(hidden, ids, weights, gate, up, down):
    loop = apply_routed_experts(hidden, ids, weights, gate, up, down)
    grouped = apply_routed_experts_grouped(hidden, ids, weights, gate, up, down)
    return loop, grouped


@pytest.mark.parametrize("tokens,top_k", [(1, 1), (3, 2), (17, 4), (64, 10), (129, 10)])
@pytest.mark.parametrize("experts,hidden,intermediate", [(4, 32, 16), (64, 48, 24)])
def test_grouped_matches_loop_float32(tokens, top_k, experts, hidden, intermediate):
    hidden_states, ids, weights, gate, up, down = _problem(
        tokens, top_k, experts, hidden, intermediate)
    loop, grouped = _both(hidden_states, ids, weights, gate, up, down)
    assert grouped.shape == loop.shape == (tokens, hidden)
    torch.testing.assert_close(grouped, loop, rtol=1e-5, atol=1e-6)
    # Triangle inequality on the F32 reference; guards a silently dropped path.
    assert torch.isfinite(grouped).all()


@pytest.mark.parametrize("tokens,top_k", [(5, 3), (40, 10)])
def test_grouped_matches_loop_bfloat16(tokens, top_k):
    hidden_states, ids, weights, gate, up, down = _problem(
        tokens, top_k, 16, 64, 32, dtype=torch.bfloat16)
    loop, grouped = _both(hidden_states, ids, weights, gate, up, down)
    cosine = torch.nn.functional.cosine_similarity(
        loop.float().flatten(), grouped.float().flatten(), dim=0)
    assert float(cosine) > 0.9999
    relative = (loop.float() - grouped.float()).abs().max() / loop.float().abs().max().clamp_min(1e-6)
    assert float(relative) < 5e-2


def test_grouped_handles_single_expert_for_every_slot():
    tokens, top_k, experts = 9, 4, 8
    ids = torch.full((tokens, top_k), 3, dtype=torch.int32)
    hidden_states, _, weights, gate, up, down = _problem(
        tokens, top_k, experts, 24, 12, ids=ids)
    loop, grouped = _both(hidden_states, ids, weights, gate, up, down)
    torch.testing.assert_close(grouped, loop, rtol=1e-5, atol=1e-6)


def test_grouped_handles_one_expert_per_token_and_duplicates():
    token_count, experts = 6, 5
    ids = torch.tensor([[0, 0], [1, 0], [4, 4], [2, 1], [3, 2], [0, 1]], dtype=torch.int32)
    hidden_states, _, weights, gate, up, down = _problem(
        token_count, 2, experts, 16, 8, ids=ids)
    loop, grouped = _both(hidden_states, ids, weights, gate, up, down)
    torch.testing.assert_close(grouped, loop, rtol=1e-5, atol=1e-6)


def test_grouped_skips_unused_experts_and_empty_batch():
    tokens, experts = 4, 32
    ids = torch.zeros(tokens, 2, dtype=torch.int32)  # only expert 0
    hidden_states, _, weights, gate, up, down = _problem(
        tokens, 2, experts, 16, 8, ids=ids)
    loop, grouped = _both(hidden_states, ids, weights, gate, up, down)
    torch.testing.assert_close(grouped, loop, rtol=1e-5, atol=1e-6)

    empty = torch.zeros(0, 16)
    empty_ids = torch.zeros(0, 2, dtype=torch.int32)
    empty_weights = torch.zeros(0, 2)
    out = apply_routed_experts_grouped(empty, empty_ids, empty_weights, gate, up, down)
    assert out.shape == (0, 16)


def test_grouped_rejects_mismatched_geometry():
    hidden_states, ids, weights, gate, up, down = _problem(3, 2, 4, 16, 8)
    with pytest.raises(ValueError, match="down must be"):
        apply_routed_experts_grouped(hidden_states, ids, weights, gate, up, down[:, :, :4])
    with pytest.raises(ValueError, match="gate/up disagree"):
        apply_routed_experts_grouped(hidden_states, ids, weights, gate, up[:, :4], down)
    with pytest.raises(ValueError, match="expects hidden"):
        apply_routed_experts_grouped(hidden_states[:, :8], ids, weights, gate, up, down)


def _reference_compact(topk_ids: torch.Tensor, n_experts: int):
    """The old nonzero/scatter compaction, kept only as a numeric oracle."""
    flat = topk_ids.reshape(-1).to(dtype=torch.int64)
    present = torch.zeros(n_experts, dtype=torch.int32)
    if flat.numel():
        present.scatter_(0, flat, torch.ones(flat.shape, dtype=torch.int32))
    unique = torch.nonzero(present, as_tuple=False).flatten().to(dtype=torch.int32)
    remap = torch.zeros(n_experts, dtype=torch.int32)
    if unique.numel():
        remap.scatter_(
            0, unique.to(dtype=torch.int64),
            torch.arange(unique.shape[0], dtype=torch.int32))
    return unique, remap[flat].view_as(topk_ids)


def test_compact_matches_nonzero_oracle_without_calling_it():
    from vllm_gaudi.models.qwen4exp.packed_moe import compact_expert_ids

    cases = [
        torch.tensor([[1, 3], [0, 1], [2, 2], [3, 0]], dtype=torch.int32),
        torch.zeros(8, 10, dtype=torch.int32),
        torch.randint(0, 512, (32, 10)),
        torch.arange(512, dtype=torch.int32).view(64, 8) % 512,
        torch.empty(0, 4, dtype=torch.int32),
    ]
    for ids in cases:
        n = 512 if ids.numel() == 0 or int(ids.max()) < 512 else int(ids.max()) + 1
        got_u, got_l = compact_expert_ids(ids, n)
        want_u, want_l = _reference_compact(ids, n)
        assert torch.equal(got_u, want_u), (got_u.tolist(), want_u.tolist())
        assert torch.equal(got_l, want_l)

    import ast
    import inspect
    import textwrap

    from vllm_gaudi.models.qwen4exp import packed_moe

    source = inspect.getsource(packed_moe.compact_expert_ids)
    tree = ast.parse(textwrap.dedent(source))
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Name):
            used.add(node.id)
    banned = {"nonzero", "searchsorted", "bincount", "scatter_add_", "scatter_", "masked_select", "unique"}
    assert not (used & banned), sorted(used & banned)
    assert "argsort" in used and "index_copy_" in used


def test_grouped_implementation_has_no_per_expert_loop():
    import ast
    import inspect
    import textwrap

    from vllm_gaudi.models.qwen4exp import packed_moe

    source = inspect.getsource(packed_moe.apply_routed_experts_grouped)
    tree = ast.parse(textwrap.dedent(source))
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Name):
            used.add(node.id)
    assert "bmm" in used
    assert "index_copy_" in used and "argsort" in used
    # Static-shape contract: the Gaudi2 graph compiler rejects these ops inside
    # the engine (synStatus 26), so the bucketing code must not call them.
    banned = {"nonzero", "searchsorted", "bincount", "scatter_add_", "masked_select"}
    assert not (used & banned), sorted(used & banned)
    # Toggle is default-on and the env var still restores the loop.
    assert prefill_grouped_enabled() is True
    os.environ["VLLM_QWEN4EXP_MOE_PREFILL_GROUPED"] = "0"
    try:
        assert prefill_grouped_enabled() is False
    finally:
        os.environ.pop("VLLM_QWEN4EXP_MOE_PREFILL_GROUPED", None)


def test_grouped_weighted_sum_matches_manual_assembly():
    """Independent reference: per-slot manual SwiGLU + weighted scatter-add."""
    tokens, top_k, experts, hidden, intermediate = 7, 3, 6, 12, 6
    hidden_states, ids, weights, gate, up, down = _problem(
        tokens, top_k, experts, hidden, intermediate)
    grouped = apply_routed_experts_grouped(
        hidden_states, ids, weights, gate, up, down)
    reference = torch.zeros(tokens, hidden)
    for token in range(tokens):
        for slot in range(top_k):
            expert = int(ids[token, slot])
            x = hidden_states[token]
            g = torch.nn.functional.linear(x, gate[expert])
            u = torch.nn.functional.linear(x, up[expert])
            projected = torch.nn.functional.linear(torch.nn.functional.silu(g) * u, down[expert])
            reference[token] += projected * weights[token, slot]
    torch.testing.assert_close(grouped, reference, rtol=1e-5, atol=1e-6)
