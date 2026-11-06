# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the QSA torch fallback.

1. ``select_qsa_indices`` matches an independent per-query loop (the
   transformers eager indexer) on sets and on attention output.
2. A context of length ``<= indexer_budget`` selects every causal token, so
   sparse attention equals dense causal attention. One extra complete block
   drops tokens.
3. RoPE cos/sin composition matches vLLM's MRoPE native path.
4. ``Qwen4ExpQSAAttention`` owns the 3 indexer parameters, carries a prefill
   into decode, and keeps two requests in separate slots.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("VLLM_HW", "cpu")

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vllm-gaudi"))


def _cos_sin_table(max_pos: int, rotary_dim: int, base: float = 10000.0) -> torch.Tensor:
    inv = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    freqs = torch.outer(torch.arange(max_pos).float(), inv)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _dummy_vllm_config(max_model_len: int = 64):
    """A throwaway ``VllmConfig`` for tests that build a vLLM ``CustomOp``.

    Instantiating ``MRotaryEmbedding`` (or any ``CustomOp``) reads the
    compilation config, which needs ``set_current_vllm_config``.
    """
    from vllm.config import ModelConfig, VllmConfig

    dummy = Path(tempfile.mkdtemp()) / "dummy-model"
    dummy.mkdir(parents=True, exist_ok=True)
    (dummy / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "vocab_size": 32,
        "torch_dtype": "bfloat16",
    }))
    model_config = ModelConfig(
        model=str(dummy), tokenizer=str(dummy), dtype="bfloat16", seed=0,
        max_model_len=max_model_len,
    )
    return VllmConfig(model_config=model_config)


def _naive_select(
    q: torch.Tensor,
    raw_keys: torch.Tensor,
    *,
    k_weight: torch.Tensor,
    eps: float,
    cos_sin: torch.Tensor,
    token_budget: int,
    compress_ratio: int,
) -> torch.Tensor:
    """Independent loop mirroring ``Qwen4ExpTextQSAIndexer``."""
    from vllm_gaudi.models.qwen4exp.qsa_math import apply_partial_rope, gemma_rmsnorm

    num_queries, n_heads, head_dim = q.shape
    ratio = compress_ratio
    width = token_budget + ratio - 1
    block_topk = token_budget // ratio
    out = torch.full((num_queries, width), -1, dtype=torch.int32)
    num_blocks = raw_keys.shape[0] // ratio
    if num_blocks:
        groups = raw_keys[: num_blocks * ratio].view(num_blocks, ratio, head_dim)
        pooled = gemma_rmsnorm(
            groups.float().mean(1).to(raw_keys.dtype), k_weight, eps
        )
        starts = torch.arange(num_blocks) * ratio
        pooled = apply_partial_rope(pooled, cos_sin[starts])
    else:
        pooled = raw_keys.new_empty(0, head_dim)
    for t in range(num_queries):
        visible = int(t + 1)  # queries sit at positions 0..T-1
        n_complete = visible // ratio
        selected: list[int] = []
        if n_complete:
            # [heads, blocks] -> sum over heads, matching the eager indexer.
            scores = torch.matmul(q[t].float(), pooled[:n_complete].float().T)
            scores = torch.relu(scores).sum(0) / math.sqrt(head_dim)
            picked = scores.topk(min(block_topk, n_complete), dim=0).indices
            for block in picked.tolist():
                selected.extend(range(block * ratio, (block + 1) * ratio))
        tail = range(n_complete * ratio, visible)
        selected.extend(tail)
        out[t, : len(selected)] = torch.tensor(selected, dtype=torch.int32)
    return out


def _sets(indices: torch.Tensor) -> list[set[int]]:
    return [{int(i) for i in row.tolist() if i >= 0} for row in indices]


def test_select_matches_loop() -> None:
    from vllm_gaudi.models.qwen4exp.qsa_math import (
        apply_partial_rope,
        gemma_rmsnorm,
        select_qsa_indices,
        sparse_attention,
    )

    torch.manual_seed(0)
    ratio, budget, heads, dim, rotary = 4, 8, 3, 8, 4
    seq = 13  # more than one extra block past the budget
    raw = torch.randn(seq, dim)
    q_raw = torch.randn(seq, heads, dim)
    weight_q = torch.randn(dim)
    weight_k = torch.randn(dim)
    table = _cos_sin_table(seq, rotary)
    q = apply_partial_rope(gemma_rmsnorm(q_raw, weight_q, 1e-6), table)
    positions = torch.arange(seq)
    starts = torch.arange(seq // ratio) * ratio
    got = select_qsa_indices(
        q,
        raw,
        positions,
        k_norm_weight=weight_k,
        eps=1e-6,
        block_cos_sin=table[starts],
        token_budget=budget,
        compress_ratio=ratio,
    )
    ref = _naive_select(
        q, raw, k_weight=weight_k, eps=1e-6, cos_sin=table,
        token_budget=budget, compress_ratio=ratio,
    )
    assert got.shape == (seq, budget + ratio - 1), got.shape
    assert _sets(got) == _sets(ref), (got, ref)
    # Last query must have dropped a complete block.
    assert len(_sets(got)[-1]) < seq
    k = torch.randn(seq, 2, dim)
    v = torch.randn(seq, 2, dim)
    q_main = torch.randn(seq, 4, dim)
    scale = dim**-0.5
    attn_got = sparse_attention(q_main, k, v, got, scale)
    attn_ref = sparse_attention(q_main, k, v, ref, scale)
    assert torch.allclose(attn_got, attn_ref, rtol=1e-5, atol=1e-5)
    print(f"  select matches loop: seq={seq} width={got.shape[1]} last={len(_sets(got)[-1])}")


def test_short_context_is_dense() -> None:
    from vllm_gaudi.models.qwen4exp.qsa_math import select_qsa_indices, sparse_attention

    torch.manual_seed(1)
    ratio, budget, dim = 4, 8, 8
    seq = budget  # exact budget: every causal token is kept
    raw = torch.randn(seq, dim)
    q = torch.randn(seq, 2, dim)
    table = _cos_sin_table(seq, 4)
    starts = torch.arange(seq // ratio) * ratio
    idx = select_qsa_indices(
        q, raw, torch.arange(seq),
        k_norm_weight=torch.zeros(dim), eps=1e-6,
        block_cos_sin=table[starts],
        token_budget=budget, compress_ratio=ratio,
    )
    for t, row in enumerate(_sets(idx)):
        assert row == set(range(t + 1)), (t, row)
    k = torch.randn(seq, 1, dim)
    v = torch.randn(seq, 1, dim)
    q_main = torch.randn(seq, 2, dim)
    scale = dim**-0.5
    sparse = sparse_attention(q_main, k, v, idx, scale)
    dense_idx = torch.arange(seq).view(1, seq).expand(seq, seq).clone()
    causal = dense_idx > torch.arange(seq).view(seq, 1)
    dense_idx = dense_idx.masked_fill(causal, -1)
    dense = sparse_attention(q_main, k, v, dense_idx, scale)
    assert torch.allclose(sparse, dense, rtol=1e-5, atol=1e-5)
    print(f"  context <= budget ({seq}) is dense causal: OK")


def test_tail_only_and_empty_query() -> None:
    from vllm_gaudi.models.qwen4exp.qsa_math import select_qsa_indices

    ratio, budget, dim = 4, 8, 8
    raw = torch.randn(3, dim)  # no complete block
    q = torch.randn(3, 2, dim)
    idx = select_qsa_indices(
        q, raw, torch.arange(3),
        k_norm_weight=torch.zeros(dim), eps=1e-6,
        block_cos_sin=None,
        token_budget=budget, compress_ratio=ratio,
    )
    assert _sets(idx) == [{0}, {0, 1}, {0, 1, 2}]
    empty = select_qsa_indices(
        q[:0], raw, torch.arange(0),
        k_norm_weight=torch.zeros(dim), eps=1e-6,
        block_cos_sin=None,
        token_budget=budget, compress_ratio=ratio,
    )
    assert empty.shape == (0, budget + ratio - 1)
    print("  tail-only and empty query: OK")


def test_dense_static_bucket_matches_prefix() -> None:
    """Lazy static-capacity attention == active-prefix attention (CPU)."""
    from vllm_gaudi.models.qwen4exp.qsa_math import (
        dense_prefix_attention,
        qsa_static_bucket_enabled,
    )

    torch.manual_seed(3)
    q = torch.randn(2, 4, 16)
    k = torch.randn(8, 2, 16)
    v = torch.randn(8, 2, 16)
    scale = 16**-0.5
    cap_k = torch.zeros_like(k)
    cap_v = torch.zeros_like(v)
    cap_k[:5] = k[:5]
    cap_v[:5] = v[:5]
    # Decode form (no positions) and causal prefill form.
    ref = dense_prefix_attention(q, k[:5], v[:5], scale)
    got = dense_prefix_attention(q, cap_k, cap_v, scale, valid_num_keys=5)
    assert torch.allclose(ref, got, atol=1e-6), (ref - got).abs().max()
    pos = torch.tensor([2, 4])
    ref = dense_prefix_attention(q, k[:5], v[:5], scale, query_positions=pos)
    got = dense_prefix_attention(
        q, cap_k, cap_v, scale, query_positions=pos, valid_num_keys=5)
    assert torch.allclose(ref, got, atol=1e-6), (ref - got).abs().max()
    # Full-validity is the plain path; out-of-range validity is rejected.
    assert torch.allclose(
        dense_prefix_attention(q, k, v, scale, valid_num_keys=8),
        dense_prefix_attention(q, k, v, scale))
    try:
        dense_prefix_attention(q, k, v, scale, valid_num_keys=9)
    except ValueError:
        pass
    else:
        raise AssertionError("valid_num_keys > num_keys must raise")
    # Auto enables only for lazy HPU; explicit 0/1 wins.
    old = {n: os.environ.get(n) for n in ("VLLM_QWEN4EXP_QSA_STATIC_BUCKET", "PT_HPU_LAZY_MODE")}
    try:
        os.environ["VLLM_QWEN4EXP_QSA_STATIC_BUCKET"] = "0"
        assert not qsa_static_bucket_enabled("hpu")
        os.environ["VLLM_QWEN4EXP_QSA_STATIC_BUCKET"] = "1"
        assert qsa_static_bucket_enabled("cpu")
        del os.environ["VLLM_QWEN4EXP_QSA_STATIC_BUCKET"]
        os.environ["PT_HPU_LAZY_MODE"] = "1"
        assert qsa_static_bucket_enabled("hpu")
        assert not qsa_static_bucket_enabled("cpu")
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    print("  dense static-bucket == active prefix: OK")


def test_slot_truncate_and_static_backing() -> None:
    """Backing capacity is zero-tailed; rollback + re-append works."""
    from vllm_gaudi.models.qwen4exp.qsa import _QSASlot

    slot = _QSASlot(
        torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4),
        torch.arange(3), torch.empty(0, 4))
    assert slot.token_capacity >= 3
    assert bool((slot.key_backing[3:] == 0).all())
    assert bool((slot.value_backing[3:] == 0).all())
    slot.append_tokens(
        torch.randn(1, 4), torch.randn(1, 4), torch.randn(1, 4), torch.tensor([3]))
    assert slot.key.shape[0] == 4
    slot.truncate_tokens(3)
    assert slot.key.shape[0] == slot.raw_keys.shape[0] == 3
    assert slot.rope_positions.shape[0] == 3
    slot.append_tokens(
        torch.randn(1, 4), torch.randn(1, 4), torch.randn(1, 4), torch.tensor([3]))
    assert slot.key.shape[0] == 4, "re-append after rollback failed"
    print("  slot truncate / re-append + zero tail: OK")


def test_gc_kernel_path_helper() -> None:
    """ensure_gc_kernel_path prepends once and preserves other entries."""
    from vllm_gaudi.models.qwen4exp._gc_kernel_path import (
        ensure_gc_kernel_path,
        qwen4exp_kernel_library,
    )

    if not qwen4exp_kernel_library().is_file():
        print("  gc kernel path helper: skipped (kernel not built)")
        return
    first = ensure_gc_kernel_path()
    entries = first.split(os.pathsep)
    assert entries[0] == str(qwen4exp_kernel_library()), entries
    assert ensure_gc_kernel_path() == first, "not idempotent"
    print("  gc kernel path helper prepends once: OK")


def test_mrope_matches_vllm() -> None:
    from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope

    from vllm_gaudi.models.qwen4exp.qsa_math import mrope_cos_sin

    torch.manual_seed(2)
    rotary = 8  # cos||sin width; section sum must be rotary/2 = 4
    section = [2, 1, 1]
    cache = torch.randn(16, rotary)
    positions = torch.randint(0, 16, (3, 5))
    got = mrope_cos_sin(
        cache, positions, mrope_section=section, mrope_interleaved=True
    )
    gathered = cache[positions]
    cos, sin = gathered.chunk(2, dim=-1)
    cos = apply_interleaved_rope(cos, section)
    sin = apply_interleaved_rope(sin, section)
    ref = torch.cat((cos, sin), dim=-1)
    assert torch.equal(got, ref), (got - ref).abs().max()
    one_d = torch.tensor([1, 4, 7])
    assert torch.equal(mrope_cos_sin(cache, one_d), cache[one_d])
    print("  mrope cos/sin matches vLLM native composition: OK")


def test_mrope_equal_rows_and_shared_qk_rope() -> None:
    """H28-2: the text-only M-RoPE shortcut and the shared q/k rotation.

    ``[3, N]`` positions with three identical rows are what the runner builds
    for text (``Qwen4ExpModel.get_mrope_input_positions``). The H28 fast path
    indexes ``positions[0]`` once instead of gathering 3N rows and running
    four ``torch.where`` ops; it must be bit-identical to the general path.
    The shared-``cos/sin`` rotation must be bit-identical to
    ``MRotaryEmbedding.forward`` on the same module and inputs, which is the
    property the QSA layer relies on to keep the twelve layers' q/k RoPE on one
    composition per step.
    """
    import torch.nn as nn

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    from vllm_gaudi.models.qwen4exp.qsa import _apply_qk_rope, _cos_sin_rows
    from vllm_gaudi.models.qwen4exp.qsa_math import mrope_cos_sin

    torch.manual_seed(5)
    head_dim, rotary_dim, heads, kv_heads, tokens = 32, 8, 4, 2, 6
    with set_current_vllm_config(_dummy_vllm_config()):
        rotary = MRotaryEmbedding(
            head_size=head_dim, rotary_dim=rotary_dim, max_position_embeddings=64,
            base=10000.0, is_neox_style=True, dtype=torch.bfloat16,
            mrope_section=[2, 1, 1], mrope_interleaved=True,
        )
    cache = rotary.cos_sin_cache
    rows = torch.arange(tokens).unsqueeze(0).expand(3, -1).contiguous()
    general = mrope_cos_sin(cache, rows, mrope_section=[2, 1, 1],
                           mrope_interleaved=True, equal_rows=False)
    shortcut = mrope_cos_sin(cache, rows, mrope_section=[2, 1, 1],
                             mrope_interleaved=True, equal_rows=True)
    assert torch.equal(general, shortcut)
    assert torch.equal(shortcut, cache.index_select(0, rows[0]))
    # A multimodal step (rows differ) must NOT take the shortcut's answer.
    mixed = rows.clone()
    mixed[1, -1] = 7
    assert not torch.equal(
        mrope_cos_sin(cache, mixed, mrope_section=[2, 1, 1], mrope_interleaved=True),
        cache.index_select(0, mixed[0]))

    q = (torch.randn(tokens, heads * head_dim) * 0.5).bfloat16()
    k = (torch.randn(tokens, kv_heads * head_dim) * 0.5).bfloat16()
    vendor_q, vendor_k = rotary(rows, q.clone(), k.clone())
    cos_sin = _cos_sin_rows(rotary, rows, q)
    shared_q, shared_k = _apply_qk_rope(rotary, cos_sin, q.clone(), k.clone())
    assert torch.equal(vendor_q, shared_q), "q RoPE drifted from the vendor path"
    assert torch.equal(vendor_k, shared_k), "k RoPE drifted from the vendor path"
    # ``nn.Module`` keeps MRotaryEmbedding registration honest above.
    assert isinstance(rotary, nn.Module)
    print("  H28 mrope equal-rows + shared q/k RoPE: bit-identical")


def test_module_prefill_decode() -> None:
    """Tiny QSA layer: indexer weights exist, decode reuses the prefill cache."""
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    from vllm_gaudi.models.qwen4exp.config import Qwen4ExpTextConfig
    from vllm_gaudi.models.qwen4exp.qsa import Qwen4ExpQSAAttention

    cfg = Qwen4ExpTextConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        indexer_budget=4,
        indexer_compress_ratio=2,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        extras={
            "rope_parameters": {
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "rope_type": "default",
            },
            "attn_output_gate": True,
        },
    )
    dummy = Path(tempfile.mkdtemp()) / "dummy-model"
    dummy.mkdir(parents=True, exist_ok=True)
    (dummy / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "vocab_size": 32,
        "torch_dtype": "bfloat16",
    }))
    model_config = ModelConfig(
        model=str(dummy), tokenizer=str(dummy), dtype="bfloat16", seed=0, max_model_len=64
    )
    model_config.hf_config = cfg
    model_config.hf_text_config = cfg
    model_config.model_arch_config = model_config.get_model_arch_config()
    vllm_config = VllmConfig(model_config=model_config)

    tmp = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1, rank=0, distributed_init_method=f"file://{tmp}",
        local_rank=0, backend="gloo",
    )
    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(1, 1)
        import vllm.platforms as vp

        vp.current_platform.current_device = lambda: torch.device("cpu")
        layer = Qwen4ExpQSAAttention(
            vllm_config=vllm_config,
            config=cfg,
            layer_id=3,
            prefix="model.layers.3.self_attn",
        )
        # state_dict keys are relative to this module; the ForCausalLM path
        # prefixes them with model.layers.N.self_attn.
        names = {
            "indexer.index_qk_proj.weight",
            "indexer.q_layernorm.weight",
            "indexer.k_layernorm.weight",
        }
        have = set(layer.state_dict())
        assert names <= have, names - have
        assert layer.indexer.index_qk_proj.weight.shape == (24, 32)
        with torch.no_grad():
            layer.qkv_proj.weight.normal_(0, 0.02)
            layer.o_proj.weight.normal_(0, 0.02)
            layer.indexer.index_qk_proj.weight.normal_(0, 0.02)

        hidden = torch.randn(6, 32)
        positions = torch.arange(6)
        out = layer(positions, hidden, query_start_loc=torch.tensor([0, 6]))
        assert out.shape == (6, 32), out.shape
        assert torch.isfinite(out).all()
        assert layer._bringup[0].raw_keys.shape[0] == 6

        # Decode must see the prefill keys: corrupting them changes the output.
        decode_h = torch.randn(1, 32)
        decode_pos = torch.tensor([6])
        first = layer(decode_pos, decode_h, query_start_loc=torch.tensor([0, 1]))
        assert layer._bringup[0].raw_keys.shape[0] == 7
        saved = layer._bringup[0].raw_keys[:-1].clone()
        layer._bringup[0].raw_keys[:-1].zero_()
        layer._bringup[0].key[:-1].zero_()
        layer._bringup[0].value[:-1].zero_()
        # Roll the cache back so the next call appends at position 6 again.
        slot = layer._bringup[0]
        slot.raw_keys = slot.raw_keys[:-1]
        slot.key = slot.key[:-1]
        slot.value = slot.value[:-1]
        slot.rope_positions = slot.rope_positions[:-1]
        second = layer(decode_pos, decode_h, query_start_loc=torch.tensor([0, 1]))
        assert not torch.allclose(first, second), "decode ignored cached history"
        # Restore and confirm the cold-start contract: a chunk that does not
        # continue the cached history no longer raises — it rebuilds the slot
        # (history discarded) and warns once (qsa.py _commit).
        slot.raw_keys = torch.cat((saved, slot.raw_keys[-1:]), dim=0)
        layer._gap_warned = False
        import logging as _logging

        class _Grab(_logging.Handler):
            def __init__(self):
                super().__init__(level=_logging.WARNING)
                self.records = []

            def emit(self, record):
                self.records.append(record.getMessage())

        grab = _Grab()
        qsa_logger = _logging.getLogger("vllm_gaudi.models.qwen4exp.qsa")
        qsa_logger.addHandler(grab)
        try:
            third = layer(torch.tensor([9]), decode_h)
            layer(torch.tensor([12]), decode_h)
        finally:
            qsa_logger.removeHandler(grab)
        assert third.shape == decode_h.shape
        assert torch.isfinite(third).all()
        assert getattr(layer, "_gap_warned", False), "gap did not arm _gap_warned"
        cold = [m for m in grab.records if "cold start" in m]
        assert len(cold) == 1, f"expected exactly one cold-start warning, got {cold}"
        layer.reset_bringup_cache()
        hidden = torch.randn(5, 32)
        # req0: positions 0,1,2  req1: positions 0,1 — separate slots.
        positions = torch.tensor([0, 1, 2, 0, 1])
        qsl = torch.tensor([0, 3, 5])
        batched = layer(positions, hidden, query_start_loc=qsl)
        assert batched.shape == (5, 32)
        assert layer._bringup[0].raw_keys.shape[0] == 3
        assert layer._bringup[1].raw_keys.shape[0] == 2
        assert torch.isfinite(batched).all()
        print(f"  QSA layer prefill {tuple(out.shape)} + decode cache carry: OK")
        print("  two-request slots stay separate: OK")


def main() -> int:
    test_select_matches_loop()
    test_short_context_is_dense()
    test_tail_only_and_empty_query()
    test_dense_static_bucket_matches_prefix()
    test_slot_truncate_and_static_backing()
    test_gc_kernel_path_helper()
    test_mrope_matches_vllm()
    test_mrope_equal_rows_and_shared_qk_rope()
    test_module_prefill_decode()
    print("test_qsa_torch: all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
