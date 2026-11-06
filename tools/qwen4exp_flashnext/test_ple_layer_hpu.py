# SPDX-License-Identifier: Apache-2.0
"""Bring-up check: real-weight PLE layer forward on the Gaudi2 HPU.

Loads ``model.layers.1.ple`` projections (BF16 key/value, F32 norms, BF16
conv) and the n-gram shards that the fixture actually addresses. Tables stay
pinned on the host; lookup gathers packed rows and dequantizes on the HPU.
Checks:

1. Pinned host gather + HPU dequant matches the CPU dequant of the same rows.
2. Prefill then a short continuation writes conv state, and the continuation
   differs from a fresh-state rerun.
3. HPU BF16 matches a CPU BF16 run of the same layer.

The full 128-shard attach (~27 GiB) is not repeated here: routing is flat, so
only the shards the fixture hits are loaded. Pinning is still exercised.

Run:
  PT_HPU_LAZY_MODE=0 python tools/qwen4exp_flashnext/test_ple_layer_hpu.py
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

import vllm_gaudi  # noqa: F401, E402

CHECKPOINT = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()
PREFIX = "model.language_model.layers.1.ple"
_DIST_READY = False
_VLLM_CONFIG = None


def _headers():
    for shard in sorted((CHECKPOINT / "tensors").glob("*.safetensors")):
        with open(shard, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            yield shard, json.loads(f.read(size))


def _load(name: str) -> torch.Tensor:
    from safetensors import safe_open

    for shard, header in _headers():
        if name in header:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                return handle.get_tensor(name)
    raise FileNotFoundError(name)


def _build():
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    from vllm_gaudi.models.qwen4exp.config import Qwen4ExpTextConfig
    from vllm_gaudi.models.qwen4exp.ple_layer import Qwen4ExpPLELayer

    text = json.load(open(CHECKPOINT / "config.json"))["text_config"]
    cfg = Qwen4ExpTextConfig.from_dict(text)
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
        model=str(dummy), tokenizer=str(dummy), dtype="bfloat16", seed=0, max_model_len=8192,
    )
    model_config.hf_config = cfg
    model_config.hf_text_config = cfg
    model_config.model_arch_config = model_config.get_model_arch_config()
    vllm_config = VllmConfig(model_config=model_config)
    global _DIST_READY, _VLLM_CONFIG
    _VLLM_CONFIG = vllm_config
    if not _DIST_READY:
        tmp = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=1, rank=0, distributed_init_method=f"file://{tmp}",
            local_rank=0, backend="gloo",
        )
        kept = set_current_vllm_config(vllm_config)
        kept.__enter__()
        initialize_model_parallel(1, 1)
        _DIST_READY = True

    layer = Qwen4ExpPLELayer(cfg, vllm_config, layer_idx=1, prefix="model.layers.1.ple")
    key = _load(f"{PREFIX}.key_proj.weight")
    value = _load(f"{PREFIX}.value_proj.weight")
    state = {
        "kv_proj.weight": torch.cat((key, value), dim=0).contiguous(),
        "norm_key.weight": _load(f"{PREFIX}.norm_key.weight"),
        "norm_query.weight": _load(f"{PREFIX}.norm_query.weight"),
        "norm_conv.weight": _load(f"{PREFIX}.norm_conv.weight"),
        "conv1d.weight": _load(f"{PREFIX}.conv1d.weight").contiguous(),
    }
    params = dict(layer.named_parameters())
    cast = {name: weight.to(dtype=params[name].dtype) for name, weight in state.items()}
    missing, unexpected = layer.load_state_dict(cast, strict=False)
    assert not missing, missing
    assert not unexpected, unexpected
    layer.to(dtype=torch.bfloat16)
    return layer, cfg


def _attach(
    layer,
    chunks: list[tuple[torch.Tensor, torch.Tensor]],
    context: torch.Tensor,
    extra: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
):
    """Pin only the shards this fixture's n-gram ids address.

    ``extra`` is ``[(token ids, query_start_loc, ngram context)]`` for steps
    outside the main fixture (the H29 batched-decode probe); its n-gram rows
    have to be resolved here so their shards get pinned too.
    """
    from vllm_gaudi.models.qwen4exp.ple_runtime import (
        PLENGramEmbedding,
        compose_ngram_ids,
        load_ple_shard,
        ple_ckpt_prefix,
    )

    weight_map = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"]
    ckpt_prefix = ple_ckpt_prefix(1)
    multipliers = _load(f"{ckpt_prefix}.layer_multipliers")
    offsets = _load(f"{ckpt_prefix}.ngram_heads_offsets")
    sizes = _load(f"{ckpt_prefix}.ngram_heads_vocab_sizes")
    scale_name = f"{ckpt_prefix}.ngram_embedding.weight_scale_2"
    global_scale = float(_load(scale_name).reshape(-1)[0]) if scale_name in weight_map else 1.0
    # Each forward is its own request (qsl [0, n]), matching the HPU run below.
    row_ids = []
    for ids, _hidden in chunks:
        qsl = torch.tensor([0, ids.shape[0]], dtype=torch.int32)
        row_ids.append(compose_ngram_ids(
            ids, qsl, context, multipliers, sizes, offsets,
            layer.eos_token_id, layer.heads_per_ngram,
        ))
    for ids, qsl, ctx in extra or ():
        row_ids.append(compose_ngram_ids(
            ids, qsl, ctx, multipliers, sizes, offsets,
            layer.eos_token_id, layer.heads_per_ngram,
        ))
    row_ids = torch.cat(row_ids, dim=0)
    # Shard width is the same for every table; read it from shard 0's header
    # via a load, then keep that table.
    first = load_ple_shard(
        CHECKPOINT / "tensors", weight_map, ckpt_prefix, 0,
        global_scale=global_scale, pin_memory=True,
    )
    assert first.packed.is_pinned() and first.scales.is_pinned(), "shard 0 did not pin"
    shard_rows = first.rows
    needed = sorted({int(row) // shard_rows for row in row_ids.reshape(-1).tolist()})
    tables: list = [None] * 128
    tables[0] = first
    for shard_id in needed:
        if tables[shard_id] is None:
            tables[shard_id] = load_ple_shard(
                CHECKPOINT / "tensors", weight_map, ckpt_prefix, shard_id,
                global_scale=global_scale, pin_memory=True,
            )
            assert tables[shard_id].packed.is_pinned()
    layer.attach_ngram(PLENGramEmbedding(
        tables, multipliers, sizes, offsets,
        heads_per_ngram=layer.heads_per_ngram,
        ngram_size=layer.short_conv_dilation,
        shard_rows=shard_rows,
    ))
    return needed, first


def _assert_close(name: str, hpu: torch.Tensor, cpu: torch.Tensor, abs_limit: float) -> None:
    diff = (hpu.float().cpu() - cpu.float()).abs()
    abs_max = diff.max().item()
    cosine = torch.nn.functional.cosine_similarity(
        hpu.float().cpu().reshape(1, -1), cpu.float().reshape(1, -1),
    ).item()
    print(f"  {name}: absmax {abs_max:.3e} cosine {cosine:.6f}")
    assert torch.isfinite(hpu.float()).all(), name
    assert abs_max < abs_limit, f"{name}: HPU vs CPU bf16 absmax {abs_max}"
    assert cosine > 0.999, f"{name}: cosine {cosine}"


def _h29_run(layer, hidden, ids, qsl, context, positions, *, on: bool):
    """Run one step with the H29 umbrella set, from a fixed conv-state seed."""
    import os

    saved = {
        k: os.environ.get(k)
        for k in ("VLLM_QWEN4EXP_PLE_H29", "VLLM_QWEN4EXP_PLE_H29_GATE",
                  "VLLM_QWEN4EXP_PLE_H29_CONV", "VLLM_QWEN4EXP_PLE_H29_BATCH")
    }
    try:
        for key in saved:
            os.environ.pop(key, None)
        os.environ["VLLM_QWEN4EXP_PLE_H29"] = "1" if on else "0"
        with torch.no_grad():
            out = layer(hidden, ids, qsl, context, positions=positions)
        torch.hpu.synchronize()
        state = layer._bringup_conv_state.detach().clone()
        return out, state
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_batched_decode(layer, cfg, probe_ids, prime_ids) -> None:
    """H29-3: a 4-request decode step, batched vs the per-span loop.

    One prefill primes slot 0's window, then four concurrent single-token
    requests run: the H29 batched path (one compose / lookup / ``kv_proj`` /
    gate kernel / conv kernel for the whole step) must agree with the per-span
    loop, including the rolled conv states of all four slots.
    """
    hidden_size = cfg.hc_count * cfg.hidden_size
    gen = torch.Generator().manual_seed(4243)
    context = torch.full((4, 2), layer.eos_token_id, dtype=torch.int32)
    # ``run("cpu")`` above left the module on the host; put it back and drop
    # the CPU-resident conv state.
    layer.to("hpu")
    layer._bringup_conv_state = None
    # Prime slot 0 with a 12-token chunk (longer than the 9-row window).
    prime_hidden = (
        torch.randn(12, hidden_size, generator=gen).bfloat16() * 0.02
    ).to("hpu")
    with torch.no_grad():
        layer(
            prime_hidden,
            prime_ids.to("hpu"),
            torch.tensor([0, 12], dtype=torch.int32),
            context[:1],
            positions=torch.arange(12, dtype=torch.int64),
        )
    torch.hpu.synchronize()
    primed = layer._bringup_conv_state.detach().clone()

    hidden = (torch.randn(4, hidden_size, generator=gen).bfloat16() * 0.02).to("hpu")
    qsl = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    positions = torch.tensor([12, 12, 12, 12], dtype=torch.int64)

    outs, states = {}, {}
    for on in (False, True):
        layer._bringup_conv_state = primed.clone()
        outs[on], states[on] = _h29_run(
            layer, hidden, probe_ids.to("hpu"), qsl, context, positions, on=on)
    assert outs[True].shape == (4, hidden_size)
    out_gap = (outs[True].float() - outs[False].float()).abs().max().item()
    state_gap = (states[True].float() - states[False].float()).abs().max().item()
    print(f"  batched decode (4 reqs x 1 token) vs per-span loop: out absmax "
          f"{out_gap:.3e}, state absmax {state_gap:.3e}")
    assert torch.isfinite(outs[True].float()).all()
    assert out_gap < 2e-2, f"H29 batched decode drifted: {out_gap}"
    assert state_gap < 2e-2, f"H29 batched conv state drifted: {state_gap}"
    assert states[True].shape[0] == 4 and states[True].abs().sum() > 0


def main() -> int:
    layer, cfg = _build()
    assert layer.eos_token_id == 248044, layer.eos_token_id
    gen = torch.Generator().manual_seed(2026)
    # 16 tokens covers the dilated window (state_len = 9); 4 more continue it.
    prefill_ids = torch.randint(0, 1000, (16,), generator=gen, dtype=torch.int32)
    prefill_ids[3] = layer.eos_token_id
    decode_ids = torch.randint(0, 1000, (4,), generator=gen, dtype=torch.int32)
    # Context width is ngram_size - 1. EOS-filled: no prior tokens.
    context = torch.full((1, 2), layer.eos_token_id, dtype=torch.int32)
    hidden = [
        torch.randn(16, cfg.hc_count * cfg.hidden_size, generator=gen).bfloat16() * 0.02,
        torch.randn(4, cfg.hc_count * cfg.hidden_size, generator=gen).bfloat16() * 0.02,
    ]
    ids = [prefill_ids, decode_ids]
    # Gaudi2 int64 multiply is 32-bit. The hash must still match the host.
    from vllm_gaudi.models.qwen4exp.ple_runtime import compose_ngram_ids

    qsl = torch.tensor([0, 16], dtype=torch.int32)
    host_ids = compose_ngram_ids(
        prefill_ids, qsl, context, torch.tensor([23703573157769, 20109073645365, 8052911324071]),
        torch.tensor([20000003]), torch.tensor([0]), layer.eos_token_id, 8,
    )
    device_ids = compose_ngram_ids(
        prefill_ids.to("hpu"), qsl.to("hpu"), context.to("hpu"),
        torch.tensor([23703573157769, 20109073645365, 8052911324071]),
        torch.tensor([20000003]), torch.tensor([0]), layer.eos_token_id, 8,
    )
    torch.hpu.synchronize()
    assert torch.equal(host_ids, device_ids.cpu()), "HPU n-gram hash diverged from host"
    print("  n-gram hash: HPU inputs == host int64 arithmetic")
    # The H29 batched-decode probe runs its own steps; resolve their n-gram
    # rows too so the shards they address get pinned as well.
    probe_ctx = torch.full((4, 2), layer.eos_token_id, dtype=torch.int32)
    probe_gen = torch.Generator().manual_seed(4242)
    probe_ids = torch.randint(0, 1000, (4,), generator=probe_gen, dtype=torch.int32)
    prime_ids = torch.randint(0, 1000, (12,), generator=probe_gen, dtype=torch.int32)
    extra = [
        (probe_ids, torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32), probe_ctx),
        (prime_ids, torch.tensor([0, 12], dtype=torch.int32), probe_ctx[:1]),
    ]
    needed, shard0 = _attach(layer, list(zip(ids, hidden)), context, extra)
    print(f"  pinned shards: {len(needed)} of 128 ({needed[:8]}...)")

    probe = torch.tensor([0, shard0.rows // 2, shard0.rows - 1])
    hpu_rows = shard0.lookup(probe, dtype=torch.bfloat16, device="hpu")
    cpu_rows = shard0.lookup(probe, dtype=torch.bfloat16, device="cpu")
    torch.hpu.synchronize()
    row_gap = (hpu_rows.float().cpu() - cpu_rows.float()).abs().max().item()
    print(f"  pinned gather, host dequant, HPU copy vs CPU: absmax {row_gap:.3e}")
    assert hpu_rows.device.type == "hpu"
    assert row_gap == 0.0 or row_gap < 1e-6, row_gap

    def run(device: str):
        layer._bringup_conv_state = None
        host = layer.to(device)
        outs = []
        for hidden_i, ids_i in zip(hidden, ids):
            outs.append(host(
                hidden_i.to(device),
                ids_i.to(device),
                torch.tensor([0, ids_i.shape[0]], dtype=torch.int32, device=device),
                context.to(device),
            ))
            if device == "hpu":
                torch.hpu.synchronize()
        state = host._bringup_conv_state.detach().clone()
        return outs, state

    hpu_out, hpu_state = run("hpu")
    assert hpu_out[0].shape == (16, cfg.hc_count * cfg.hidden_size)
    assert hpu_state[0].abs().sum() > 0, "conv state was not written"
    fresh_state = layer._bringup_conv_state
    layer._bringup_conv_state = None
    with torch.no_grad():
        fresh = layer(
            hidden[1].to("hpu"),
            ids[1].to("hpu"),
            torch.tensor([0, 4], dtype=torch.int32, device="hpu"),
            context.to("hpu"),
        )
    torch.hpu.synchronize()
    carried = (hpu_out[1].float() - fresh.float()).abs().max().item()
    assert carried > 1e-4, f"carried conv state did not affect the continuation ({carried})"
    print(f"  HPU conv carry: continuation vs fresh absmax {carried:.3e}")
    layer._bringup_conv_state = fresh_state

    cpu_out, cpu_state = run("cpu")
    for i, (h, c) in enumerate(zip(hpu_out, cpu_out)):
        _assert_close(f"chunk {i}", h, c, abs_limit=2e-2)
    _assert_close("conv state", hpu_state[0], cpu_state[0], abs_limit=2e-2)
    test_batched_decode(layer, cfg, probe_ids, prime_ids)
    print("all PLE layer HPU tests passed")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PT_HPU_LAZY_MODE", "0")
    raise SystemExit(main())
