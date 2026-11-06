# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the PLE torch runtime (ple_runtime.py).

1. ``compose_ngram_ids`` matches a naive per-token loop (multi-request,
   chunk boundaries, EOS terminator).
2. ``ple_gate`` matches the documented formulas with the kernels' per-boundary
   dtype materialization.
3. ``ple_short_conv`` prefill output equals ``F.conv1d`` with dilation/left
   padding; state carry-over makes prefill(prefix)+decode(rest) equal to a
   single prefill of the whole sequence.
4. Real layer-1 PLE tables: 128 shards load, head/shard routing indexes the
   right rows, and lookup output matches direct row dequantization.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_HW", "cpu")

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vllm-gaudi"))

CKPT_DIR = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()


def test_ngram_ids() -> None:
    from vllm_gaudi.models.qwen4exp.ple_runtime import compose_ngram_ids

    torch.manual_seed(0)
    ctx_len, heads_per_ngram = 2, 4
    num_heads = ctx_len * heads_per_ngram
    vocab = 32
    input_ids = torch.randint(0, vocab, (9,))
    query_start_loc = torch.tensor([0, 4, 9], dtype=torch.int32)
    ngram_context = torch.tensor([[7, 8], [9, 10]], dtype=torch.int32)
    multipliers = torch.tensor([1, 101, 103], dtype=torch.int64)  # per shift
    sizes = torch.full((num_heads,), 11, dtype=torch.int64)
    offsets = torch.arange(num_heads, dtype=torch.int64) * 13
    eos = 5

    got = compose_ngram_ids(
        input_ids, query_start_loc, ngram_context, multipliers, sizes, offsets,
        eos, heads_per_ngram,
    )

    # Naive reference: per token, per head.
    want = torch.zeros(9, num_heads, dtype=torch.int64)
    for r in range(2):
        lo, hi = int(query_start_loc[r]), int(query_start_loc[r + 1])
        for t in range(lo, hi):
            j = t - lo
            for h in range(num_heads):
                mixed = int(input_ids[t]) * int(multipliers[0])
                crossed = False
                for shift in range(1, ctx_len + 1):
                    if j >= shift:
                        cand = int(input_ids[t - shift])
                    else:
                        cand = int(ngram_context[r, ctx_len - shift + j])
                    if crossed:
                        cand = eos
                    crossed = crossed or cand == eos
                    order = h // heads_per_ngram + 2
                    if order > shift:
                        mixed ^= cand * int(multipliers[shift])
                want[t, h] = mixed % int(sizes[h]) + int(offsets[h])
    assert torch.equal(got, want), f"ngram ids mismatch:\n{got}\n{want}"
    print("  ngram id composition == naive per-token reference: OK")


def test_ngram_vs_transformers_reference() -> None:
    """compose_ngram_ids vs a faithful transcription of the transformers
    ``Qwen4ExpTextNGramEmbedding.forward`` (segment-restart shifts, per-order
    xor mixing, int64 wraparound). Ids are int64, so equality must be exact."""
    from vllm_gaudi.models.qwen4exp.ple_runtime import compose_ngram_ids

    def reference(input_ids_b, previous_context, eos, multipliers, sizes, offsets,
                  ngram_size, heads_per_ngram):
        batch, seq = input_ids_b.shape
        history = torch.cat([previous_context, input_ids_b], dim=1)
        length = history.shape[1]
        pos = torch.arange(length)
        eos_pos = torch.where(
            history == eos, pos.unsqueeze(0).expand(batch, -1),
            torch.full((batch, length), -1),
        )
        prev_eos = torch.cat(
            [torch.full((batch, 1), -1), torch.cummax(eos_pos, dim=1).values[:, :-1]],
            dim=1,
        )
        pos_in_seg = pos.unsqueeze(0) - (prev_eos + 1)
        shifted = [history]
        for s in range(1, ngram_size):
            src = (pos - s).clamp_min(0).unsqueeze(0).expand(batch, -1)
            vals = history.gather(1, src)
            valid = (pos_in_seg >= s) & ((pos - s).unsqueeze(0) >= 0)
            shifted.append(torch.where(valid, vals, torch.full_like(vals, eos)))
        blocks = []
        for n in range(2, ngram_size + 1):
            mixed = shifted[0] * multipliers[0]
            for p in range(1, n):
                mixed = torch.bitwise_xor(mixed, shifted[p] * multipliers[p])
            h0 = (n - 2) * heads_per_ngram
            h1 = h0 + heads_per_ngram
            ids = torch.remainder(
                mixed.unsqueeze(-1), sizes[h0:h1].view(1, 1, -1)
            )
            blocks.append(ids + offsets[h0:h1].view(1, 1, -1))
        return torch.cat(blocks, dim=-1)[:, -seq:]

    eos, vocab, ngram_size, heads_per = 5, 20, 3, 3
    ctx = ngram_size - 1
    num_heads = ctx * heads_per
    torch.manual_seed(11)
    checked = 0
    for trial, eos_prob in enumerate((0.0, 0.15, 0.5, 0.9)):
        sizes = torch.tensor([97, 101, 103, 107, 109, 113], dtype=torch.int64)
        offsets = torch.tensor([0, 97, 198, 301, 408, 517], dtype=torch.int64)
        # Force int64 wraparound: products of ~2^62 multipliers overflow.
        multipliers = torch.randint(
            2**62, 2**63 - 1, (ngram_size,), dtype=torch.int64
        ) * torch.where(torch.rand(ngram_size) < 0.5, -1, 1)
        lengths = [7, 1, 12, 3]
        chunks, contexts = [], []
        for b, length in enumerate(lengths):
            ids = torch.randint(0, vocab, (length,))
            sprinkle = torch.rand(length) < eos_prob
            ids[sprinkle] = eos
            chunks.append(ids)
            if b % 2:  # carry real history; the rest start fresh (EOS padding)
                contexts.append(torch.randint(0, vocab, (ctx,)))
            else:
                contexts.append(torch.full((ctx,), eos, dtype=torch.long))
        flat = torch.cat(chunks)
        qsl = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32)
        ours = compose_ngram_ids(
            flat,
            qsl,
            torch.stack(contexts),
            multipliers,
            sizes,
            offsets,
            eos,
            heads_per,
        ).reshape(-1, num_heads)
        ref = reference(
            torch.stack([torch.nn.functional.pad(c, (0, max(lengths) - len(c)), value=eos)
                         for c in chunks]),
            torch.stack(contexts),
            eos, multipliers, sizes, offsets, ngram_size, heads_per,
        )
        start = 0
        for b, length in enumerate(lengths):
            got = ours[start:start + length]
            want = ref[b, :length]
            assert torch.equal(got, want), (trial, b)
            checked += length
            start += length
    print(f"  ngram ids == transformers reference (4 eos densities, wraparound): "
          f"{checked} tokens exact: OK")


def test_gate() -> None:
    from vllm_gaudi.models.qwen4exp.ple_runtime import ple_gate

    torch.manual_seed(1)
    hc, h, tokens = 4, 64, 5
    key = torch.randn(tokens, hc * h, dtype=torch.bfloat16)
    value = torch.randn(tokens, h, dtype=torch.bfloat16)
    hidden = torch.randn(tokens, hc * h, dtype=torch.bfloat16)
    nk = torch.randn(hc * h, dtype=torch.bfloat16)
    nq = torch.randn(hc * h, dtype=torch.bfloat16)
    ncw = torch.randn(hc * h, dtype=torch.bfloat16)
    eps = 1e-6

    gated, conv_in = ple_gate(key, value, hidden, nk, nq, ncw, eps)

    def rmsnorm(x, w):
        xf = x.float().unflatten(-1, (hc, h))
        var = xf.square().mean(-1, keepdim=True)
        n = xf * torch.rsqrt(var + eps)
        return (n * (1.0 + w.float().unflatten(0, (hc, h)).unsqueeze(0))).flatten(-2)

    kn = rmsnorm(key, nk).to(torch.bfloat16)
    qn = rmsnorm(hidden, nq).to(torch.bfloat16)
    products = (kn.float() * qn.float()).to(torch.bfloat16)
    dot = products.float().unflatten(-1, (hc, h)).sum(-1)
    d = (dot / h**0.5).to(torch.bfloat16).float()
    sign = torch.sign(d)
    mag = torch.sqrt(torch.clamp(d.abs(), min=1e-6)).to(torch.bfloat16)
    g = torch.sigmoid(sign * mag.float()).to(torch.bfloat16)
    want_gated = (g[:, :, None] * value.float()[:, None, :]).flatten(-2).to(torch.bfloat16)
    want_conv = rmsnorm(want_gated, ncw).to(torch.bfloat16)
    assert torch.equal(gated, want_gated), "gated mismatch"
    assert torch.equal(conv_in, want_conv), "conv input mismatch"
    print("  ple_gate == documented formula w/ kernel rounding: OK")


def test_short_conv() -> None:
    from vllm_gaudi.models.qwen4exp.ple_runtime import ple_short_conv

    torch.manual_seed(2)
    c, kernel, dilation = 8, 4, 3
    state_len = (kernel - 1) * dilation
    seq = torch.randn(11, c, dtype=torch.bfloat16)
    weights = torch.randn(c, 1, kernel, dtype=torch.bfloat16)
    qsl = torch.tensor([0, 11], dtype=torch.int64)
    conv_state = torch.zeros(1, state_len, c, dtype=torch.bfloat16)
    slots = torch.tensor([0])

    got = ple_short_conv(
        seq, qsl, conv_state, weights, slots, dilation=dilation, has_state=None
    )
    # Left-pad only: causal dilated conv, groups=C so channels stay independent.
    padded = F.pad(seq.t().unsqueeze(0).float(), (state_len, 0))
    conv = F.conv1d(
        padded, weights.float(), dilation=dilation, groups=c
    ).squeeze(0).t()
    conv = conv.to(torch.bfloat16).float()
    want = (conv * torch.sigmoid(conv)).to(torch.bfloat16)
    assert got.shape == want.shape
    assert torch.allclose(got.float(), want.float(), atol=1e-2), (
        got.float() - want.float()
    ).abs().max()

    # Split: prefill 6 tokens, then decode the remaining 5 one by one.
    conv_state2 = torch.zeros(1, state_len, c, dtype=torch.bfloat16)
    pre = ple_short_conv(
        seq[:6], torch.tensor([0, 6]), conv_state2, weights, slots, dilation=dilation
    )
    assert torch.equal(pre, got[:6]), "prefill prefix mismatch"
    outs = []
    for t in range(6, 11):
        outs.append(
            ple_short_conv(
                seq[t : t + 1], torch.tensor([0, 1]), conv_state2, weights, slots,
                dilation=dilation, has_state=torch.tensor([True]),
            )
        )
    decoded = torch.cat(outs, dim=0)
    assert torch.equal(decoded, got[6:]), "decode continuation mismatch"
    print("  short conv == F.conv1d(dilation) and prefill->decode carry: OK")


def test_real_tables() -> None:
    from vllm_gaudi.models.qwen4exp.ple_embedding import PLEShardTable
    from vllm_gaudi.models.qwen4exp.ple_runtime import PLENGramEmbedding

    weight_map = json.loads((CKPT_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    from safetensors import safe_open

    def get(name: str) -> torch.Tensor:
        with safe_open(CKPT_DIR / "tensors" / weight_map[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    prefix = "model.language_model.layers.1.ple.ple_embedding"
    multipliers = get(prefix + ".layer_multipliers")
    offsets = get(prefix + ".ngram_heads_offsets")
    sizes = get(prefix + ".ngram_heads_vocab_sizes")
    shard_names = sorted(
        (n for n in weight_map if prefix + ".ngram_embedding.shard_" in n
         and n.endswith(".weight") and "scale" not in n),
        key=lambda n: int(n.split("shard_")[1].split(".")[0]),
    )
    assert len(shard_names) == 128, len(shard_names)
    # The first 16 shards cover flat rows 0..40,000,191, which fully contains
    # the valid ranges of heads 0 and 1 (head 1 ends at 40,000,026).
    tables = []
    for name in shard_names[:16]:
        base = name[: -len(".weight")]
        tables.append(PLEShardTable(get(name), get(base + ".weight_scale"), pin_memory=False))
    shard_rows = tables[0].rows
    emb = PLENGramEmbedding(
        tables, multipliers, sizes, offsets,
        heads_per_ngram=8, ngram_size=3, shard_rows=shard_rows,
    )
    assert emb.num_heads == 16, emb.num_heads
    assert emb.head_dim == 160, emb.head_dim

    def flat_row(global_id: int) -> tuple[int, int]:
        return divmod(global_id, shard_rows)

    def direct(global_id: int) -> torch.Tensor:
        s, row = flat_row(global_id)
        return emb.tables[s].lookup(torch.tensor([row]), dtype=torch.bfloat16)[0]

    # Sample starts, middles, and last valid ids of heads 0 and 1, plus the
    # head-1 boundary id itself. Columns for heads 2..15 repeat these ids;
    # routing is per-element flat math and does not depend on the column.
    sampled = [
        int(offsets[0]), int(offsets[0]) + 12345, int(offsets[0] + sizes[0]) - 1,
        int(offsets[1]), int(offsets[1]) + 999, int(offsets[1] + sizes[1]) - 1,
    ]
    # Each row repeats one sampled id across heads; routing is per-element
    # flat math and does not depend on the column.
    ids = torch.tensor(sampled, dtype=torch.int64).unsqueeze(1).expand(
        -1, emb.num_heads
    )
    out = emb.lookup(ids, dtype=torch.bfloat16)
    assert out.shape == (len(sampled), emb.num_heads * 160), out.shape
    assert torch.isfinite(out.float()).all()
    for t, gid in enumerate(sampled):
        assert torch.equal(out[t, :160], direct(gid)), f"flat routing mismatch for id {gid}"

    # Regression for the old per-head routing: the head-1 boundary id used to
    # be re-relativized (id - offset -> shard 0 of "head 1", i.e. tables[8]
    # row 0). Flat routing must hit shard 7, and the two disagree on the
    # real table contents.
    boundary = int(offsets[1])
    s, row = flat_row(boundary)
    assert (s, row) == (7, boundary - 7 * shard_rows)
    assert torch.equal(direct(boundary), emb.tables[7].lookup(
        torch.tensor([row]), dtype=torch.bfloat16)[0])
    wrong = emb.tables[8].lookup(torch.tensor([0]), dtype=torch.bfloat16)[0]
    assert not torch.equal(direct(boundary), wrong), "boundary id must not hit the next shard"
    # Metadata invariants of the flat layout: contiguous head ranges that fit
    # inside the physical table with only tail padding left over.
    total_rows = 128 * shard_rows
    last_valid = int(offsets[-1] + sizes[-1])
    assert all(int(offsets[i] + sizes[i]) == int(offsets[i + 1]) for i in range(15))
    assert 0 <= last_valid <= total_rows
    assert 0 < total_rows - last_valid < shard_rows  # 90 rows of exporter padding
    print(f"  real PLE tables: flat routing OK over head boundary "
          f"(shard {s}, row {row:,}); padding rows {total_rows - last_valid}")


def test_ngram_lookup_bit_exact() -> None:
    """Synthetic mini-shards with the EXACT row layout (80 B packed + 10 B
    E4M3 scales): the per-step gather + dequant pipeline equals the csf.py
    dequant_nvfp4 golden bit-exactly, on the fp32 path and after the bf16
    cast; the CPU twin (``device="cpu"``) is the same code path and never
    builds a pinned staging buffer."""
    from vllm_gaudi.models.qwen4exp.ple_embedding import ROW_BYTES, PLEShardTable
    from vllm_gaudi.models.qwen4exp.ple_runtime import PLENGramEmbedding

    sys.path.insert(0, str(Path(__file__).parent))
    from csf import dequant_nvfp4 as np_golden  # noqa: E402

    global_scale = 6.64847e-05
    shard_rows, num_shards = 8, 16
    gen = torch.Generator().manual_seed(9)
    tables = []
    for _ in range(num_shards):
        packed = torch.randint(0, 256, (shard_rows, 80), generator=gen).to(torch.uint8)
        scales = torch.randint(0x20, 0x78, (shard_rows, 10), generator=gen).to(torch.uint8)
        tables.append(PLEShardTable(packed, scales, global_scale=global_scale,
                                    pin_memory=False))
    # One row = 90 bytes: 80 packed E2M1 (160 values) + 10 E4M3 block scales.
    assert (tables[0].packed.shape[1], tables[0].scales.shape[1]) == (80, 10)
    assert ROW_BYTES == 90 and PLENGramEmbedding.ROW_BYTES == 90
    assert tables[0].embed_dim == 160

    emb = PLENGramEmbedding(
        tables,
        torch.tensor([1, 3, 5], dtype=torch.int64),
        torch.full((16,), shard_rows, dtype=torch.int64),
        torch.arange(16, dtype=torch.int64) * shard_rows,
        heads_per_ngram=8,
        ngram_size=3,
        shard_rows=shard_rows,
    )
    assert emb.global_scale == global_scale

    # Token rows spread over every shard, with boundary ids and duplicates.
    ids = torch.randint(0, num_shards * shard_rows, (7, 16), generator=gen)
    ids[0, 0] = 0
    ids[3, 5] = num_shards * shard_rows - 1
    ids[6, 6] = ids[2, 2]

    got = emb.lookup(ids, dtype=torch.float32)
    assert got.shape == (7, 16 * 160)
    for t in range(7):
        for h in range(16):
            gid = int(ids[t, h])
            shard, row = divmod(gid, shard_rows)
            gold = np_golden(
                tables[shard].packed[row:row + 1].numpy(),
                tables[shard].scales[row:row + 1].numpy(),
                global_scale,
            )[0]
            assert torch.equal(got[t, h * 160:(h + 1) * 160], torch.from_numpy(gold)), (t, h, gid)
    print(f"  gather+dequant == csf.py dequant_nvfp4 golden: {7 * 16} rows bit-exact: OK")

    # The gather returns the raw packed rows (still NVFP4) in flat order.
    packed_rows, scale_rows = emb.gather(ids)
    assert packed_rows.shape == (7 * 16, 80) and scale_rows.shape == (7 * 16, 10)
    flat = ids.reshape(-1)
    for i in (0, 5, 7, 111):
        shard, row = divmod(int(flat[i]), shard_rows)
        assert torch.equal(packed_rows[i], tables[shard].packed[row])
        assert torch.equal(scale_rows[i], tables[shard].scales[row])

    # CPU twin: an explicit cpu device runs the identical dequant and never
    # stages into pinned memory (the staging path is device-only).
    twin = emb.lookup(ids, dtype=torch.float32, device="cpu")
    assert torch.equal(twin, got)
    assert emb._stage is None, "the CPU path must not build a pinned staging buffer"

    # bf16 output is the fp32 result rounded once.
    got_bf16 = emb.lookup(ids, dtype=torch.bfloat16)
    assert got_bf16.dtype == torch.bfloat16
    assert torch.equal(got_bf16, got.to(torch.bfloat16))
    print("  CPU twin (device='cpu') identical, no pinned staging; bf16 cast exact: OK")


def test_ple_table_geometry() -> None:
    """Real-table geometry, computed from shard headers -- never materialized:
    128 shards x 2,500,012 rows x 90 B = 320,001,536 rows = 26.8222 GiB."""
    import struct

    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    weight_map = json.loads((CKPT_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    shard_ids = sorted(
        int(n.split("shard_")[1].split(".")[0]) for n in weight_map
        if n.startswith(prefix + ".shard_") and n.endswith(".weight") and "scale" not in n
    )
    assert shard_ids == list(range(128)), len(shard_ids)

    # Stream headers only: every one of the 128 shards declares the same
    # U8 [rows, 80] packed plane + F8_E4M3 [rows, 10] scale plane.
    headers: dict[str, dict] = {}
    rows = width = scale_cols = None
    for shard_id in shard_ids:
        entry = weight_map[f"{prefix}.shard_{shard_id}.weight"]
        if entry not in headers:
            with open(CKPT_DIR / "tensors" / entry, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                headers[entry] = json.loads(handle.read(length))
        weight = headers[entry][f"{prefix}.shard_{shard_id}.weight"]
        scales = headers[entry][f"{prefix}.shard_{shard_id}.weight_scale"]
        assert weight["dtype"] == "U8" and weight["shape"] == [2_500_012, 80]
        assert scales["dtype"] == "F8_E4M3" and scales["shape"] == [2_500_012, 10]
        rows, width, scale_cols = weight["shape"][0], weight["shape"][1], scales["shape"][1]

    # Arithmetic check of the 51B host-residency budget (no tensors read).
    assert (rows, width, scale_cols) == (2_500_012, 80, 10)
    total_rows = 128 * rows
    total_bytes = 128 * rows * (width + scale_cols)
    assert total_rows == 320_001_536
    assert total_bytes == 128 * 2_500_012 * 90 == 28_800_138_240
    gib = total_bytes / 2**30
    assert abs(gib - 26.8222) < 0.0005, gib
    # One row decodes to 160 values: the per-token gather is 16 such rows.
    assert width * 2 == 160 and scale_cols * 16 == 160
    print(f"  table geometry: 128 shards x {rows:,} rows x 90 B = {total_bytes:,} B "
          f"({gib:.4f} GiB), headers only: OK")


def test_apply_ple() -> None:
    from vllm_gaudi.models.qwen4exp.ple_runtime import apply_ple, ple_gate, ple_short_conv

    torch.manual_seed(4)
    tokens, hc, h = 3, 2, 8
    hidden = torch.randn(tokens, hc * h, dtype=torch.bfloat16)
    key = torch.randn(tokens, hc * h, dtype=torch.bfloat16)
    value = torch.randn(tokens, h, dtype=torch.bfloat16)
    nk = torch.randn(hc * h, dtype=torch.bfloat16)
    nq = torch.randn(hc * h, dtype=torch.bfloat16)
    ncw = torch.randn(hc * h, dtype=torch.bfloat16)
    qsl = torch.tensor([0, tokens])
    state = torch.zeros(1, 6, hc * h, dtype=torch.bfloat16)
    weight = torch.randn(hc * h, 1, 4, dtype=torch.bfloat16)
    slots = torch.tensor([0])
    out = apply_ple(
        hidden, key, value, qsl, state, weight, slots, nk, nq, ncw,
        eps=1e-6, dilation=2,
    )
    gated, conv_in = ple_gate(key, value, hidden, nk, nq, ncw, 1e-6)
    # State was updated by apply_ple; recompute conv against a fresh buffer.
    fresh = torch.zeros_like(state)
    activated = ple_short_conv(
        conv_in, qsl, fresh, weight, slots, dilation=2
    )
    mixed = (gated.float() + activated.float()).to(torch.bfloat16)
    want = (hidden.float() + mixed.float()).to(torch.bfloat16)
    assert out.shape == hidden.shape
    assert torch.equal(out, want), "apply_ple residual fusion mismatch"
    assert not torch.equal(state, torch.zeros_like(state)), "conv state was not updated"
    print("  apply_ple residual fusion + state update: OK")


def test_layer_forward() -> None:
    """PLE layer forward on CPU with tiny stand-in shard tables."""
    import tempfile

    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    from vllm_gaudi.models.qwen4exp.config import Qwen4ExpConfig
    from vllm_gaudi.models.qwen4exp.ple_embedding import PLEShardTable
    from vllm_gaudi.models.qwen4exp.ple_layer import Qwen4ExpPLELayer
    from vllm_gaudi.models.qwen4exp.ple_runtime import PLENGramEmbedding

    snap = sorted(
        (
            Path.home()
            / ".cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots"
        ).iterdir()
    )[0]
    cfg = Qwen4ExpConfig.from_json(snap / "config.json").text
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
        layer = Qwen4ExpPLELayer(cfg, vllm_config, layer_idx=1, prefix="model.layers.1.ple")
        shard_rows = 8
        tables = [
            PLEShardTable(
                torch.randint(0, 256, (shard_rows, 80), dtype=torch.uint8),
                torch.randint(1, 120, (shard_rows, 10), dtype=torch.uint8),
                pin_memory=False,
            )
            for _ in range(16)
        ]
        ngram = PLENGramEmbedding(
            tables,
            torch.tensor([1, 3, 5], dtype=torch.int64),
            torch.full((16,), shard_rows, dtype=torch.int64),
            torch.arange(16, dtype=torch.int64) * shard_rows,
            heads_per_ngram=8,
            ngram_size=3,
            shard_rows=shard_rows,
        )
        layer.attach_ngram(ngram)
        # Vendor init leaves these at zero; the wiring test needs a real signal.
        with torch.no_grad():
            layer.kv_proj.weight.normal_(0, 0.02)
            layer.conv1d.weight.normal_(0, 0.02)
        hidden = torch.randn(4, cfg.hc_count * cfg.hidden_size, dtype=torch.bfloat16)
        ids = torch.randint(0, 50, (4,))
        qsl = torch.tensor([0, 4], dtype=torch.int32)
        ctx = torch.full((1, 2), layer.eos_token_id, dtype=torch.int32)
        out = layer(hidden, ids, qsl, ctx)
        assert out.shape == hidden.shape, out.shape
        assert torch.isfinite(out.float()).all()
        decode = layer(
            hidden[:1],
            ids[:1],
            torch.tensor([0, 1], dtype=torch.int32),
            ctx,
        )
        assert decode.shape == (1, hidden.shape[-1])
        assert layer._bringup_conv_state is not None
        assert layer._bringup_conv_state.abs().sum() > 0
        print(f"  PLE layer forward: {tuple(out.shape)} finite, conv state carried: OK")


def test_load_shard() -> None:
    from vllm_gaudi.models.qwen4exp.ple_runtime import load_ple_shard, ple_ckpt_prefix

    weight_map = json.loads((CKPT_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = ple_ckpt_prefix(1)
    shard_names = [
        n for n in weight_map
        if n.startswith(prefix + ".ngram_embedding.shard_") and n.endswith(".weight")
        and "scale" not in n
    ]
    assert len(shard_names) == 128, len(shard_names)
    table = load_ple_shard(
        CKPT_DIR / "tensors", weight_map, prefix, 0, pin_memory=False
    )
    assert table.embed_dim == 160, table.embed_dim
    assert table.rows > 1_000_000, table.rows
    row = table.lookup(torch.tensor([0]), dtype=torch.bfloat16)
    assert row.shape == (1, 160)
    assert torch.isfinite(row.float()).all()
    assert row.float().abs().sum() > 0
    print(f"  load_ple_shard: 128 shards, shard0 rows={table.rows} dim={table.embed_dim}: OK")


def test_prepare_ple_inputs() -> None:
    from vllm_gaudi.models.qwen4exp.ple_runtime import infer_request_spans, prepare_ple_inputs

    eos = 7
    assert infer_request_spans(torch.tensor([0, 1, 2, 3])) == [(0, 4)]
    assert infer_request_spans(torch.tensor([0, 1, 0, 1])) == [(0, 2), (2, 4)]
    mrope = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    assert infer_request_spans(mrope) == [(0, 3)]

    history: list[torch.Tensor] = []
    ids = torch.tensor([10, 11, 12, 13])
    qsl, ctx = prepare_ple_inputs(
        ids, torch.arange(4), query_start_loc=None, ngram_context=None,
        ngram_size=3, eos_token_id=eos, history=history,
    )
    assert qsl.tolist() == [0, 4]
    assert ctx.tolist() == [[eos, eos]]
    assert history[0].tolist() == [10, 11, 12, 13]

    qsl, ctx = prepare_ple_inputs(
        torch.tensor([14]), torch.tensor([4]), query_start_loc=None, ngram_context=None,
        ngram_size=3, eos_token_id=eos, history=history,
    )
    assert qsl.tolist() == [0, 1]
    assert ctx.tolist() == [[12, 13]]
    assert history[0].tolist() == [10, 11, 12, 13, 14]

    history.clear()
    ids = torch.tensor([1, 2, 3, 4])
    qsl, ctx = prepare_ple_inputs(
        ids, torch.tensor([0, 1, 0, 1]), query_start_loc=None, ngram_context=None,
        ngram_size=3, eos_token_id=eos, history=history,
    )
    assert qsl.tolist() == [0, 2, 4]
    assert ctx.tolist() == [[eos, eos], [eos, eos]]
    assert [h.tolist() for h in history] == [[1, 2], [3, 4]]

    # Unrelated decode positions need an explicit split.
    try:
        prepare_ple_inputs(
            torch.tensor([8, 9]), torch.tensor([5, 9]),
            query_start_loc=None, ngram_context=None,
            ngram_size=3, eos_token_id=eos, history=[],
        )
        raise AssertionError("ambiguous positions should raise")
    except RuntimeError as exc:
        assert "query_start_loc" in str(exc)

    qsl, ctx = prepare_ple_inputs(
        torch.tensor([8, 9]), torch.tensor([5, 9]),
        query_start_loc=torch.tensor([0, 1, 2]), ngram_context=None,
        ngram_size=3, eos_token_id=eos, history=[],
    )
    assert qsl.tolist() == [0, 1, 2]
    assert ctx.tolist() == [[eos, eos], [eos, eos]]
    print("  prepare_ple_inputs: prefill, decode carry, two requests, explicit split: OK")


def main() -> int:
    test_ngram_ids()
    test_ngram_vs_transformers_reference()
    test_prepare_ple_inputs()
    test_gate()
    test_short_conv()
    test_apply_ple()
    test_ngram_lookup_bit_exact()
    test_ple_table_geometry()
    test_real_tables()
    test_load_shard()
    test_layer_forward()
    print("test_ple_runtime: all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
