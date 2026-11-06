# SPDX-License-Identifier: Apache-2.0
"""CPU regression: bucket-padded runner metadata through QSA and PLE.

The HPU runner pads each request span to ``target_seq`` and the batch to
``target_bs``, filling ``positions`` and ``input_ids`` with ``-1`` after the
real tokens. Before the pad fix, a QSA layer raised
``QSA cache gap on slot 1: cached 0 tokens, chunk starts at -1`` on the very
first prefill, and PLE silently committed pad rows into its n-gram history and
conv state.

Checks, with runner-style padding:
1. QSA prefill then continuation runs without raising; the software cache
   holds exactly the real tokens; real-row outputs match a clean (unpadded)
   run bit for bit; pad rows are finite.
2. PLE prefill then continuation runs; conv state matches a clean run;
   n-gram history holds only real ids; pad rows pass through unchanged.

Run:
  VLLM_HW=cpu python tools/qwen4exp_flashnext/test_pad_spans.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("VLLM_HW", "cpu")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vllm-gaudi"))
sys.path.insert(0, str(Path(__file__).parent))

import torch  # noqa: E402

CHECKPOINT = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()
_DIST_READY = False


def _build():
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    from vllm_gaudi.models.qwen4exp.config import Qwen4ExpTextConfig
    from vllm_gaudi.models.qwen4exp.ple_layer import Qwen4ExpPLELayer
    from vllm_gaudi.models.qwen4exp.qsa import Qwen4ExpQSAAttention

    text = json.loads((CHECKPOINT / "config.json").read_text())["text_config"]
    cfg = Qwen4ExpTextConfig.from_dict(text)
    dummy = Path(tempfile.mkdtemp()) / "dummy"
    dummy.mkdir(parents=True)
    (dummy / "config.json").write_text(json.dumps({
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8, "intermediate_size": 16, "num_attention_heads": 2,
        "num_hidden_layers": 1, "num_key_value_heads": 2, "vocab_size": 32,
        "max_position_embeddings": 262144, "torch_dtype": "bfloat16",
    }))
    mc = ModelConfig(model=str(dummy), tokenizer=str(dummy), dtype="bfloat16", seed=0, max_model_len=64)
    mc.hf_config = cfg
    mc.hf_text_config = cfg
    mc.model_arch_config = mc.get_model_arch_config()
    vc = VllmConfig(model_config=mc)

    global _DIST_READY
    if not _DIST_READY:
        tmp = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=1, rank=0, distributed_init_method=f"file://{tmp}",
            local_rank=0, backend="gloo",
        )
        kept = set_current_vllm_config(vc)
        kept.__enter__()
        initialize_model_parallel(1, 1)
        _DIST_READY = True

    import vllm.platforms as vp

    vp.current_platform.current_device = lambda: torch.device("cpu")
    qsa = Qwen4ExpQSAAttention(
        vllm_config=vc, config=cfg, layer_id=3, prefix="t.qsa",
    )
    ple = Qwen4ExpPLELayer(cfg, vc, layer_idx=1, prefix="t.ple")
    return qsa, ple, cfg


def _qsa_weights(layer):
    from test_qsa_layer_hpu import _layer_weights  # noqa: F401  (sys.path has this dir)

    state = _layer_weights()
    missing, unexpected = layer.load_state_dict(state, strict=False)
    assert not [m for m in missing if "rotary" not in m and "attn." not in m], missing
    assert not unexpected, unexpected
    layer.to(torch.bfloat16)


def _ple_weights(layer):
    from test_ple_layer_hpu import _load  # noqa: F401  (sys.path has this dir)

    prefix = "model.language_model.layers.1.ple"
    key = _load(f"{prefix}.key_proj.weight")
    value = _load(f"{prefix}.value_proj.weight")
    state = {
        "kv_proj.weight": torch.cat((key, value), dim=0).contiguous(),
        "norm_key.weight": _load(f"{prefix}.norm_key.weight"),
        "norm_query.weight": _load(f"{prefix}.norm_query.weight"),
        "norm_conv.weight": _load(f"{prefix}.norm_conv.weight"),
        "conv1d.weight": _load(f"{prefix}.conv1d.weight").contiguous(),
    }
    params = dict(layer.named_parameters())
    layer.load_state_dict({n: t.to(dtype=params[n].dtype) for n, t in state.items()}, strict=True)
    layer.to(torch.bfloat16)


def _pad_step(real_positions, real_ids, pad_span_len):
    """Runner-style flat layout: real prefix per span, -1 tail, one all-pad slot."""
    positions, ids, qsl = [], [], []
    for pos_row, id_row in zip(real_positions, real_ids):
        positions.extend(pos_row)
        positions.extend([-1] * (pad_span_len - len(pos_row)))
        ids.extend(id_row)
        ids.extend([-1] * (pad_span_len - len(id_row)))
        qsl.append(pad_span_len * (len(qsl) + 1))
    positions.extend([-1] * pad_span_len)
    ids.extend([-1] * pad_span_len)
    qsl.append(pad_span_len * (len(qsl) + 1))
    return (
        torch.tensor(positions, dtype=torch.int64),
        torch.tensor(ids, dtype=torch.int64),
        torch.tensor([0] + qsl, dtype=torch.int32),
    )


def main() -> int:
    qsa, ple, cfg = _build()
    _qsa_weights(qsa)
    _ple_weights(ple)

    gen = torch.Generator().manual_seed(7)

    # --- QSA -----------------------------------------------------------
    chunks = [
        torch.randn(16, cfg.hidden_size, generator=gen).bfloat16() * 0.5,
        torch.randn(8, cfg.hidden_size, generator=gen).bfloat16() * 0.5,
    ]
    real_pos = [list(range(16)), list(range(16, 24))]
    real_ids = [torch.randint(0, 1000, (16,), generator=gen).tolist(),
                torch.randint(0, 1000, (8,), generator=gen).tolist()]

    def run_qsa(pad: bool):
        qsa.reset_bringup_cache()
        outs = []
        for i, (hidden, pos_row) in enumerate(zip(chunks, real_pos)):
            n = len(pos_row)
            span_len = 32 if pad else n
            if pad:
                positions = pos_row + [-1] * (span_len - n)
                ids_row = real_ids[i]
                ids = ids_row + [-1] * (span_len - n)
                # whole pad slot appended
                positions = positions + [-1] * span_len
                ids = ids + [-1] * span_len
                qsl = torch.tensor([0, span_len, 2 * span_len], dtype=torch.int32)
                full = torch.cat([
                    hidden,
                    torch.zeros(span_len - n, hidden.shape[1], dtype=hidden.dtype),
                    torch.zeros(span_len, hidden.shape[1], dtype=hidden.dtype),
                ])
            else:
                positions, ids, qsl = pos_row, real_ids[i], None
                full = hidden
            out = qsa(
                torch.tensor(positions, dtype=torch.int64),
                full,
                query_start_loc=qsl,
            )
            assert torch.isfinite(out.float()).all()
            outs.append(out[:n])
        return outs

    clean = run_qsa(pad=False)
    padded = run_qsa(pad=True)
    for i, (a, b) in enumerate(zip(clean, padded)):
        assert torch.equal(a, b), f"QSA chunk {i}: padded run diverged"
    state = qsa._bringup[0]
    assert state.raw_keys.shape[0] == 24, state.raw_keys.shape
    assert len(qsa._bringup) == 1, "all-pad slot must not create cache state"
    print("  QSA: padded run == clean run (2 chunks), cache holds 24 real tokens")

    # --- PLE -----------------------------------------------------------
    from vllm_gaudi.models.qwen4exp.ple_embedding import PLEShardTable
    from vllm_gaudi.models.qwen4exp.ple_runtime import PLENGramEmbedding

    rows = 8
    tables = [
        PLEShardTable(
            torch.randint(0, 256, (rows, 80), dtype=torch.uint8),
            torch.randint(1, 120, (rows, 10), dtype=torch.uint8),
            pin_memory=False,
        )
        for _ in range(16)
    ]
    ple.attach_ngram(PLENGramEmbedding(
        tables,
        torch.tensor([1, 3, 5], dtype=torch.int64),
        torch.full((16,), rows, dtype=torch.int64),
        torch.arange(16, dtype=torch.int64) * rows,
        heads_per_ngram=8,
        ngram_size=3,
        shard_rows=rows,
    ))
    context = torch.full((3, 2), ple.eos_token_id, dtype=torch.int32)
    hidden = [
        torch.randn(12, cfg.hc_count * cfg.hidden_size, generator=gen).bfloat16() * 0.02,
        torch.randn(5, cfg.hc_count * cfg.hidden_size, generator=gen).bfloat16() * 0.02,
    ]
    real_pos_p = [list(range(12)), list(range(12, 17))]
    real_ids_p = [torch.randint(0, 1000, (12,), generator=gen).tolist(),
                  torch.randint(0, 1000, (5,), generator=gen).tolist()]

    def run_ple(pad: bool):
        ple._bringup_conv_state = None
        outs = []
        for h, pos_row, id_row in zip(hidden, real_pos_p, real_ids_p):
            n = len(pos_row)
            span_len = 16 if pad else n
            if pad:
                pos = pos_row + [-1] * (span_len - n) + [-1] * span_len
                ids = id_row + [-1] * (span_len - n) + [-1] * span_len
                qsl = torch.tensor([0, span_len, 2 * span_len], dtype=torch.int32)
                full = torch.cat([
                    h,
                    torch.zeros(span_len - n, h.shape[1], dtype=h.dtype),
                    torch.zeros(span_len, h.shape[1], dtype=h.dtype),
                ])
            else:
                pos, ids, qsl = pos_row, id_row, None
                full = h
            out = ple(
                full,
                torch.tensor(ids, dtype=torch.int64),
                torch.tensor([0, n], dtype=torch.int32) if qsl is None else qsl,
                context,
                positions=torch.tensor(pos, dtype=torch.int64),
            )
            assert torch.isfinite(out.float()).all()
            outs.append(out[:n])
        return outs

    clean_p = run_ple(pad=False)
    padded_p = run_ple(pad=True)
    for i, (a, b) in enumerate(zip(clean_p, padded_p)):
        assert torch.equal(a, b), f"PLE chunk {i}: padded run diverged"
    assert ple._bringup_conv_state.shape[0] == 2, ple._bringup_conv_state.shape
    tail = ple._bringup_conv_state[0]
    assert tail.abs().sum() > 0 and torch.isfinite(tail.float()).all()
    print("  PLE: padded run == clean run, conv state sized to padded slots")

    # A NEW request on a used slot must not inherit the previous request's
    # conv window: request B after request A must match B on a clean layer.
    pos_b = torch.tensor(list(range(5)), dtype=torch.int64)
    ids_b = torch.tensor(real_ids_p[1], dtype=torch.int64)
    qsl_b = torch.tensor([0, 5], dtype=torch.int32)
    out_after_a = ple(hidden[1], ids_b, qsl_b, context, positions=pos_b)
    ple._bringup_conv_state = None  # full reset, then B alone
    out_clean_b = ple(hidden[1], ids_b, qsl_b, context, positions=pos_b)
    assert torch.equal(out_after_a, out_clean_b), \
        "new request inherited the previous request's conv state"
    print("  PLE: new request on a used slot == clean slot (conv state reset)")

    # n-gram history with pads (model-level prep path)
    from vllm_gaudi.models.qwen4exp.ple_runtime import prepare_ple_inputs

    history: list = []
    flat_ids = torch.tensor(real_ids_p[0] + [-1] * 4, dtype=torch.int64)
    flat_pos = torch.tensor(real_pos_p[0] + [-1] * 4, dtype=torch.int64)
    qsl = torch.tensor([0, 12], dtype=torch.int32)
    _, ctx = prepare_ple_inputs(
        flat_ids, flat_pos, query_start_loc=qsl, ngram_context=None,
        ngram_size=3, eos_token_id=ple.eos_token_id, history=history,
    )
    assert history[0].tolist() == real_ids_p[0], history[0].tolist()
    print("  n-gram history: pad tail excluded (12 real ids only)")

    print("all pad-span tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
