# SPDX-License-Identifier: Apache-2.0
"""Graph-size compromise tests for the QSA indexer (tile replay).

The Qwen3.8-27B recipe keeps compiled programs shape-stable by replaying a
fixed 8192-token context tile a host-chosen number of times (11 prefill
bundles, 7 decode bundles). Qwen4Exp applies the same compromise one level
deeper: the QSA attention gather is already a fixed ``budget + ratio - 1``
width, so the only context-length-dependent tensor shape is the indexer's
block scoring. ``select_qsa_indices_from_pooled`` scores a bundled window of
the pooled-block cache instead of the raw history.

These tests prove the windowed selection is bit-identical to the full
``select_qsa_indices`` reference — including at tile boundaries, for early
queries with fewer visible blocks than the budget, and end-to-end through
``Qwen4ExpQSAAttention`` with the incremental pooled cache.

Run: python tools/qwen4exp_flashnext/test_qsa_tile_replay.py
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


def _blocks_per_tile(ratio: int) -> int:
    from vllm_gaudi.extension import prefill_tile_replay

    return prefill_tile_replay.context_tile() // ratio


def _pooled_window(
    raw_keys: torch.Tensor,
    *,
    k_norm_weight: torch.Tensor,
    eps: float,
    cos_sin: torch.Tensor,
    ratio: int,
    bundle_blocks: int,
) -> tuple[torch.Tensor, int]:
    """Pooled cache padded to the compile-stable bundle width."""
    from vllm_gaudi.models.qwen4exp.qsa_math import pool_indexer_blocks

    head_dim = raw_keys.shape[1]
    n_blocks = raw_keys.shape[0] // ratio
    starts = torch.arange(n_blocks) * ratio
    pooled = raw_keys.new_empty(0, head_dim)
    if n_blocks:
        groups = raw_keys[: n_blocks * ratio].view(n_blocks, ratio, head_dim)
        pooled = pool_indexer_blocks(
            groups, k_norm_weight=k_norm_weight, eps=eps, block_cos_sin=cos_sin[starts]
        )
    if pooled.shape[0] < bundle_blocks:
        pooled = torch.cat((pooled, pooled.new_zeros(bundle_blocks - pooled.shape[0], head_dim)))
    return pooled[:bundle_blocks], n_blocks


def test_windowed_matches_full() -> None:
    from vllm_gaudi.extension.prefill_tile_replay import PREFILL_BUNDLE_COUNTS, bundle_count
    from vllm_gaudi.models.qwen4exp.qsa_math import (
        select_qsa_indices,
        select_qsa_indices_from_pooled,
    )

    ratio, budget, heads, dim = 4, 64, 4, 32
    block_topk = budget // ratio
    per_tile = _blocks_per_tile(ratio)
    assert per_tile == 2048, per_tile
    # Contexts around every early boundary plus a 49k-token prompt.
    lengths = [5, 8192 - 4, 8192, 8192 + 4, 2 * 8192, 30000, 49283]
    for length in lengths:
        gen = torch.Generator().manual_seed(length)
        raw_keys = torch.randn(length, dim, generator=gen) * 0.5
        k_weight = torch.randn(dim, generator=gen)
        cos_sin = _cos_sin_table(length, dim)
        n_blocks = length // ratio
        n_tiles = (n_blocks + per_tile - 1) // per_tile
        bundle_blocks = bundle_count(max(n_tiles, 1)) * per_tile
        window, _ = _pooled_window(
            raw_keys,
            k_norm_weight=k_weight,
            eps=1e-6,
            cos_sin=cos_sin,
            ratio=ratio,
            bundle_blocks=bundle_blocks,
        )
        # Two query layouts: a chunk at the end (prefill) and the whole
        # context (early queries see fewer blocks than the budget).
        for tag, q_pos in (
            ("tail-chunk", torch.arange(length - 64, length)),
            ("whole", torch.arange(length)),
        ):
            q = torch.randn(q_pos.shape[0], heads, dim, generator=gen)
            full = select_qsa_indices(
                q,
                raw_keys,
                q_pos,
                k_norm_weight=k_weight,
                eps=1e-6,
                block_cos_sin=cos_sin[::ratio][:n_blocks],
                token_budget=budget,
                compress_ratio=ratio,
            )
            windowed = select_qsa_indices_from_pooled(
                q,
                window,
                q_pos,
                total_blocks=n_blocks,
                token_budget=budget,
                compress_ratio=ratio,
            )
            assert full.shape == windowed.shape == (q_pos.shape[0], budget + ratio - 1)
            assert torch.equal(full, windowed), (
                f"length={length} {tag}: {(full != windowed).any(dim=-1).sum()} rows differ"
            )
        print(f"  length={length:6d} blocks={n_blocks:6d} bundle={bundle_blocks // per_tile:2d}: "
              f"windowed == full (tail-chunk + whole) OK")
    assert bundle_count(1) == 1 and bundle_count(7) == 8 and bundle_count(32) == 32
    assert PREFILL_BUNDLE_COUNTS[-1] * 8192 == 262144


def test_attention_output_identical() -> None:
    from vllm_gaudi.models.qwen4exp.qsa_math import (
        select_qsa_indices,
        select_qsa_indices_from_pooled,
        sparse_attention,
    )

    ratio, budget, heads, kv_heads, dim = 4, 64, 8, 2, 16
    gen = torch.Generator().manual_seed(7)
    length = 20000
    raw_keys = torch.randn(length, dim, generator=gen) * 0.5
    k_weight = torch.randn(dim, generator=gen)
    cos_sin = _cos_sin_table(length, dim)
    n_blocks = length // ratio
    per_tile = _blocks_per_tile(ratio)
    from vllm_gaudi.extension.prefill_tile_replay import bundle_count

    bundle_blocks = bundle_count((n_blocks + per_tile - 1) // per_tile) * per_tile
    window, _ = _pooled_window(
        raw_keys, k_norm_weight=k_weight, eps=1e-6, cos_sin=cos_sin,
        ratio=ratio, bundle_blocks=bundle_blocks,
    )
    q_pos = torch.arange(length - 128, length)
    q = torch.randn(128, heads, dim, generator=gen)
    k = torch.randn(length, kv_heads, dim, generator=gen)
    v = torch.randn(length, kv_heads, dim, generator=gen)
    full = select_qsa_indices(
        q, raw_keys, q_pos, k_norm_weight=k_weight, eps=1e-6,
        block_cos_sin=cos_sin[::ratio][:n_blocks], token_budget=budget, compress_ratio=ratio,
    )
    windowed = select_qsa_indices_from_pooled(
        q, window, q_pos, total_blocks=n_blocks, token_budget=budget, compress_ratio=ratio,
    )
    scale = 1.0 / math.sqrt(dim)
    out_full = sparse_attention(q, k, v, full, scale)
    out_windowed = sparse_attention(q, k, v, windowed, scale)
    assert torch.equal(out_full, out_windowed)
    print(f"  sparse attention over windowed indices: bit-identical (length={length})")


def test_module_chunked_prefill_matches() -> None:
    """Layer level: flag off vs on across two prefill chunks + decode."""
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

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
    # Keep the context referenced: a bare __enter__ result gets GC'd and the
    # global config resets between the two layer builds below.
    keep_ctx = set_current_vllm_config(vllm_config)
    keep_ctx.__enter__()
    initialize_model_parallel(1, 1)
    import vllm.platforms as vp

    vp.current_platform.current_device = lambda: torch.device("cpu")

    def build_and_run(flag: bool) -> list[torch.Tensor]:
        os.environ["VLLM_HPU_PREFILL_TILE_REPLAY"] = "1" if flag else "0"
        torch.manual_seed(11)
        layer = Qwen4ExpQSAAttention(
            vllm_config=vllm_config,
            config=cfg,
            layer_id=3,
            # Distinct prefix per build: vLLM rejects duplicate layer names.
            prefix="model.layers.3.self_attn" + (".tiled" if flag else ""),
        )
        with torch.no_grad():
            layer.qkv_proj.weight.normal_(0, 0.02)
            layer.o_proj.weight.normal_(0, 0.02)
            layer.indexer.index_qk_proj.weight.normal_(0, 0.02)
        outs = []
        gen = torch.Generator().manual_seed(3)
        for start, size in ((0, 24), (24, 20)):
            hidden = torch.randn(size, 32, generator=gen)
            positions = torch.arange(start, start + size)
            outs.append(layer(positions, hidden, query_start_loc=torch.tensor([0, size])))
        decode_h = torch.randn(1, 32, generator=gen)
        outs.append(layer(torch.tensor([44]), decode_h, query_start_loc=torch.tensor([0, 1])))
        slot = layer._bringup[0]
        assert slot.raw_keys.shape[0] == 45
        assert slot.pooled.shape[0] == 45 // 2
        return outs

    plain, tiled = build_and_run(False), build_and_run(True)
    os.environ.pop("VLLM_HPU_PREFILL_TILE_REPLAY", None)
    for i, (a, b) in enumerate(zip(plain, tiled)):
        assert torch.equal(a, b), f"step {i}: tile-replay output differs (max {float((a - b).abs().max())})"
    print("  Qwen4ExpQSAAttention chunked prefill + decode: tile replay bit-identical")


def main() -> int:
    test_windowed_matches_full()
    test_attention_output_identical()
    test_module_chunked_prefill_matches()
    print("all QSA tile replay tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
