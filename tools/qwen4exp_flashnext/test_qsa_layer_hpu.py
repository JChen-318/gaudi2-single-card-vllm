# SPDX-License-Identifier: Apache-2.0
"""Bring-up check: real-weight QSA layer forward on the Gaudi2 HPU.

Loads ``model.layers.3.self_attn`` weights from the converted checkpoint
(MXFP8 dequantized to BF16), runs chunked prefill through
``Qwen4ExpQSAAttention`` on the HPU, and verifies:

1. HPU output matches a CPU bf16 run of the same layer within bf16 tolerance
   on a sparse context (3072 tokens > the 2048 budget). An fp32-weight oracle
   is not used: score rounding flips borderline indexer blocks.
2. The tile-replay (bundled window) selection on the HPU is bit-identical to
   the full-history selection on the HPU — the graph-size compromise holds on
   device.
3. Outputs are finite and the pooled incremental cache matches a full
   recomputation.

Run:
  PT_HPU_LAZY_MODE=0 python tools/qwen4exp_flashnext/test_qsa_layer_hpu.py
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vllm-gaudi"))
sys.path.insert(0, str(Path(__file__).parent))

import vllm_gaudi  # noqa: F401, E402  (registers the HPU platform first)

CHECKPOINT = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()
LAYER = "model.language_model.layers.3.self_attn"
_DIST_READY = False


def _shards():
    return sorted((CHECKPOINT / "tensors").glob("*.safetensors"))


def _headers():
    for shard in _shards():
        with open(shard, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            yield shard, json.loads(f.read(size))


def _load(name: str) -> torch.Tensor:
    from safetensors import safe_open

    for shard, header in _headers():
        if name in header:
            with safe_open(shard, framework="pt", device="cpu") as f:
                return f.get_tensor(name)
    raise FileNotFoundError(name)


def _dequant_mxfp8(base: str) -> torch.Tensor:
    from vllm_gaudi.models.qwen4exp.weights import dequant_mxfp8

    weight = _load(f"{base}.weight")
    scale = _load(f"{base}.weight_scale")
    out = dequant_mxfp8(weight, scale)
    assert out.shape == weight.shape, (base, out.shape, weight.shape)
    return out


def _layer_weights() -> dict[str, torch.Tensor]:
    """Checkpoint tensors -> layer state_dict entries (BF16)."""
    q = _dequant_mxfp8(f"{LAYER}.q_proj")
    k = _dequant_mxfp8(f"{LAYER}.k_proj")
    v = _dequant_mxfp8(f"{LAYER}.v_proj")
    return {
        "qkv_proj.weight": torch.cat((q, k, v), dim=0).contiguous(),
        "q_norm.weight": _load(f"{LAYER}.q_norm.weight"),
        "k_norm.weight": _load(f"{LAYER}.k_norm.weight"),
        "o_proj.weight": _dequant_mxfp8(f"{LAYER}.o_proj"),
        "indexer.index_qk_proj.weight": _dequant_mxfp8(f"{LAYER}.indexer.index_qk_proj"),
        "indexer.q_layernorm.weight": _load(f"{LAYER}.indexer.q_layernorm.weight"),
        "indexer.k_layernorm.weight": _load(f"{LAYER}.indexer.k_layernorm.weight"),
    }


def _build(dtype: str):
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_gaudi.models.qwen4exp.config import Qwen4ExpTextConfig
    from vllm_gaudi.models.qwen4exp.qsa import Qwen4ExpQSAAttention

    text = json.load(open(CHECKPOINT / "config.json"))["text_config"]
    cfg = Qwen4ExpTextConfig(
        hidden_size=int(text["hidden_size"]),
        num_attention_heads=int(text["num_attention_heads"]),
        num_key_value_heads=int(text["num_key_value_heads"]),
        head_dim=int(text["head_dim"]),
        max_position_embeddings=int(text["max_position_embeddings"]),
        rms_norm_eps=float(text["rms_norm_eps"]),
        indexer_budget=int(text["indexer_budget"]),
        indexer_compress_ratio=int(text["indexer_compress_ratio"]),
        indexer_n_heads=int(text["indexer_n_heads"]),
        indexer_kv_heads=int(text["indexer_kv_heads"]),
        indexer_head_dim=int(text["indexer_head_dim"]),
        extras={"rope_parameters": text["rope_parameters"]},
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
        "max_position_embeddings": 262144,
        "torch_dtype": "bfloat16",
    }))
    model_config = ModelConfig(
        model=str(dummy), tokenizer=str(dummy), dtype=dtype, seed=0, max_model_len=8192
    )
    model_config.hf_config = cfg
    model_config.hf_text_config = cfg
    model_config.model_arch_config = model_config.get_model_arch_config()
    vllm_config = VllmConfig(model_config=model_config)
    global _DIST_READY
    if not _DIST_READY:
        tmp = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=1, rank=0, distributed_init_method=f"file://{tmp}",
            local_rank=0, backend="gloo",
        )
        # Keep the context referenced: a GC'd __enter__ result resets the
        # global config between builds.
        _KEEP_CTX = set_current_vllm_config(vllm_config)
        _KEEP_CTX.__enter__()
        initialize_model_parallel(1, 1)
        _DIST_READY = True

    def build(prefix: str) -> torch.nn.Module:
        torch.manual_seed(0)
        layer = Qwen4ExpQSAAttention(
            vllm_config=vllm_config, config=cfg, layer_id=3, prefix=prefix,
        )
        state = _layer_weights()
        if dtype == "float32":
            state = {name: w.float() for name, w in state.items()}
        missing, unexpected = layer.load_state_dict(state, strict=False)
        assert not [m for m in missing if "rotary" not in m and "attn." not in m], missing
        assert not unexpected, unexpected
        layer.to(torch.float32 if dtype == "float32" else torch.bfloat16)
        return layer

    return build


def _run(layer, device: str, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
    outs = []
    start = 0
    for hidden in chunks:
        n = hidden.shape[0]
        positions = torch.arange(start, start + n, device=device)
        outs.append(layer(positions, hidden.to(device), query_start_loc=torch.tensor([0, n], device=device)))
        start += n
    return outs


def main() -> int:
    from vllm_gaudi.models.qwen4exp.qsa_math import (
        pool_indexer_blocks,
        select_qsa_indices,
        select_qsa_indices_from_pooled,
    )

    build = _build(dtype="bfloat16")
    device = "hpu"

    gen = torch.Generator().manual_seed(2026)
    # 3072 total tokens: chunk 2 sits past the 2048 budget, so its selection
    # is genuinely sparse (512 of 768 blocks). Chunk sizes keep the gather
    # einsum's transient (~4.3 GB x 2 for 2048 queries) inside HBM.
    chunks = [torch.randn(2048, 2560, generator=gen).bfloat16() * 0.5,
              torch.randn(1024, 2560, generator=gen).bfloat16() * 0.5]

    # --- HPU, tile replay off ---
    os.environ["VLLM_HPU_PREFILL_TILE_REPLAY"] = "0"
    hpu_layer = build("hpu.plain").to("hpu")
    hpu_out = _run(hpu_layer, device, chunks)
    for out in hpu_out:
        assert out.device.type == "hpu" and torch.isfinite(out.float()).all()

    # --- H28-2: [3, N] text-only M-RoPE positions -------------------------
    # The runner builds M-RoPE positions as three identical rows for text
    # (``Qwen4ExpModel.get_mrope_input_positions``). H28 composes one row for
    # that case and shares the composition across the step, which must not
    # change a single bit: the shortcut only removes the 3N-row gather and the
    # four interleave ``torch.where`` ops. Run the same chunks with 2-D
    # positions, once with no host control (the general 2-D composition) and
    # once bound (the equal-row shortcut plus the per-step cache).
    from vllm_gaudi.models.qwen4exp.control import bind_host_control, reset_host_control

    def _run_mrope_2d(layer, bound: bool) -> list[torch.Tensor]:
        layer.reset_bringup_cache()
        outs = []
        start = 0
        for hidden in chunks:
            n = hidden.shape[0]
            pos = torch.arange(start, start + n, device=device).unsqueeze(0)
            pos = pos.expand(3, -1).contiguous()
            qsl = torch.tensor([0, n], device=device)
            token = bind_host_control(pos, None, qsl) if bound else None
            try:
                outs.append(layer(pos, hidden.to(device), query_start_loc=qsl))
            finally:
                if token is not None:
                    reset_host_control(token)
            start += n
        return outs

    for bound in (False, True):
        got = _run_mrope_2d(hpu_layer, bound)
        for i, (a, b) in enumerate(zip(hpu_out, got)):
            assert torch.equal(a, b), (
                f"chunk {i}: [3,N] M-RoPE positions (bound={bound}) changed "
                "the layer output")
        print(f"  H28 [3,N] M-RoPE positions (host control {bound}): bit-identical")
    # Restore the 1-D 3072-token cache the rest of this test reads.
    hpu_layer.reset_bringup_cache()
    for i, (a, b) in enumerate(zip(hpu_out, _run(hpu_layer, device, chunks))):
        assert torch.equal(a, b), f"chunk {i}: cache rebuild differs"

    # --- HPU, tile replay on: must be bit-identical on the same device ---
    q_idx = hpu_layer.indexer.project_qk(
        chunks[1].to(device), torch.arange(2048, 3072, device=device), hpu_layer.rotary_emb)[0]
    torch.hpu.synchronize()
    os.environ["VLLM_HPU_PREFILL_TILE_REPLAY"] = "1"
    tiled_layer = build("hpu.tiled").to("hpu")
    tiled_out = _run(tiled_layer, device, chunks)
    os.environ["VLLM_HPU_PREFILL_TILE_REPLAY"] = "0"
    for i, (a, b) in enumerate(zip(hpu_out, tiled_out)):
        assert torch.equal(a, b), f"chunk {i}: tile replay differs on HPU"
    print("  HPU tile-replay vs full-history: bit-identical (chunks 2048 + 1024)")

    # Pooled incremental cache vs full recomputation, on device.
    slot = hpu_layer._bringup[0]
    raw = slot.raw_keys
    ratio = hpu_layer.indexer.compress_ratio
    n_blocks = raw.shape[0] // ratio
    groups = raw[: n_blocks * ratio].view(n_blocks, ratio, hpu_layer.indexer.index_head_dim)
    starts = torch.arange(n_blocks, device=raw.device) * ratio
    rope = slot.rope_positions
    block_pos = rope.index_select(0, starts) if rope.ndim == 1 else rope[:, starts]
    from vllm_gaudi.models.qwen4exp.qsa import _cos_sin_rows

    full_pooled = pool_indexer_blocks(
        groups,
        k_norm_weight=hpu_layer.indexer.k_layernorm.weight,
        eps=hpu_layer.indexer.k_layernorm.variance_epsilon,
        block_cos_sin=_cos_sin_rows(hpu_layer.rotary_emb, block_pos, raw),
    )
    diff = (full_pooled.float() - slot.pooled.float()).abs().max()
    assert float(diff) == 0.0, f"incremental pooled cache drift {float(diff)}"
    print(f"  HPU pooled cache: incremental == recompute over {n_blocks} blocks")

    # Selection sets on device: full vs windowed (q_idx from the plain layer).
    logical = torch.arange(2048, 3072, device=device)
    full_idx = select_qsa_indices(
        q_idx, raw, logical,
        k_norm_weight=hpu_layer.indexer.k_layernorm.weight,
        eps=hpu_layer.indexer.k_layernorm.variance_epsilon,
        block_cos_sin=hpu_layer._block_cos_sin(slot, q_idx),
        token_budget=hpu_layer.indexer.token_topk, compress_ratio=ratio,
    )
    width = 2048  # one bundled tile of 8192 tokens / ratio 4
    pooled = slot.pooled
    window = torch.cat((pooled, pooled.new_zeros(width - pooled.shape[0], pooled.shape[1])))
    win_idx = select_qsa_indices_from_pooled(
        q_idx, window, logical, total_blocks=pooled.shape[0],
        token_budget=hpu_layer.indexer.token_topk, compress_ratio=ratio,
        total_tokens=raw.shape[0],
    )
    assert torch.equal(full_idx, win_idx), "HPU windowed selection != full selection"
    print(f"  HPU selection indices: windowed == full ({tuple(full_idx.shape)})")
    del tiled_layer
    torch.hpu.synchronize()

    # Same-dtype reference. An fp32-weight oracle flips borderline indexer
    # blocks (relu zeroes and score rounding), so the sparse chunk's output
    # is not a stable comparison across dtypes. CPU bf16 runs the same path.
    cpu_layer = build("cpu.bf16")
    cpu_out = _run(cpu_layer, "cpu", chunks)
    for i, (h, c) in enumerate(zip(hpu_out, cpu_out)):
        abs_max = (h.float().cpu() - c.float()).abs().max().item()
        assert abs_max < 2e-2, f"chunk {i}: HPU vs CPU bf16 absmax {abs_max}"
        print(f"  chunk {i}: HPU vs CPU bf16 absmax {abs_max:.3e}")

    # --- H23: the decode regimes the chunked prefill does not reach -------
    # Both caches now hold 3072 tokens, so a one-token decode step runs the
    # sparse selection (S_kv > indexer_budget 2048) at a position past the
    # budget -- the 2107-token serve shape. The gates below are what H23-1
    # (BF16 MME dense attention) and H23-4 (fused RMSNorm) must hold: the
    # production default path (all H23 items on) against the CPU bf16
    # reference, on the same weights and the same cache history.
    gen_d = torch.Generator().manual_seed(4242)

    def _decode_step(layer, device, position: int, hidden: torch.Tensor):
        pos = torch.tensor([position], device=device)
        qsl = torch.tensor([0, 1], device=device)
        return layer(pos, hidden.to(device), query_start_loc=qsl)

    # One input for both sides: two draws from the same generator would
    # compare two different tokens.
    dec_hidden = (torch.randn(1, 2560, generator=gen_d) * 0.1).bfloat16()
    hpu_dec = _decode_step(hpu_layer, device, 3072, dec_hidden)
    cpu_dec = _decode_step(cpu_layer, "cpu", 3072, dec_hidden)
    abs_max = (hpu_dec.float().cpu() - cpu_dec.float()).abs().max().item()
    assert abs_max < 2e-2, f"long-context decode (S_kv=3073): absmax {abs_max}"
    print(f"  decode T=1 @ S_kv=3073 (> budget): HPU vs CPU bf16 absmax {abs_max:.3e}")

    # Short-context decode: S_kv = 257, the dense prefix path (H23-1's BF16
    # MME dense attention). Fresh caches primed with a 256-token chunk.
    hpu_layer.reset_bringup_cache()
    cpu_layer.reset_bringup_cache()
    prime = (torch.randn(256, 2560, generator=gen_d) * 0.1).bfloat16()
    prime_pos = torch.arange(256)
    prime_qsl = torch.tensor([0, 256], dtype=torch.int32)
    hpu_p = hpu_layer(prime_pos.to(device), prime.to(device),
                      query_start_loc=prime_qsl.to(device))
    cpu_p = cpu_layer(prime_pos, prime, query_start_loc=prime_qsl)
    prim_abs = (hpu_p.float().cpu() - cpu_p.float()).abs().max().item()
    assert prim_abs < 2e-2, f"256-token prefill: absmax {prim_abs}"
    hpu_dec = _decode_step(hpu_layer, device, 256, dec_hidden)
    cpu_dec = _decode_step(cpu_layer, "cpu", 256, dec_hidden)
    abs_max = (hpu_dec.float().cpu() - cpu_dec.float()).abs().max().item()
    assert abs_max < 2e-2, f"decode T=1 at S_kv=257: absmax {abs_max}"
    print(f"  decode T=1 @ S_kv=257 (dense): HPU vs CPU bf16 absmax {abs_max:.3e}")

    print("all QSA layer HPU tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
