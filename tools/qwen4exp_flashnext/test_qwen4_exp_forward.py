# SPDX-License-Identifier: Apache-2.0
"""Tier-3 CPU forward test for the Qwen4Exp HPU port.

Builds the registry-canonical ``Qwen4ExpForCausalLM`` (``qwen4_exp.py``) from
a 2-layer micro config -- layer 0 GDN + PLE, layer 1 QSA full attention --
under ``VLLM_HW=cpu`` and drives the whole decoder composition end to end:

* GDN layer on the real Qwen3.8 geometry (16 k-heads / 48 v-heads /
  head_dim 128 / conv 4, ``in_proj`` width 10240) through the
  ``HPUGatedDeltaNetAttention`` monkeypatch, prefill with real conv/ssm
  state writes and a one-token decode continuation;
* QSA layer with the torch indexer fallback (budget/ratio micro-scaled,
  4 indexer heads x 128), M-RoPE sections [11, 11, 10] interleaved;
* PLE layer inside the stack (synthetic 16-shard host table) so the
  ``layer_idx + 1 in ple_layer_ids`` hook is exercised;
* packed-NVFP4 MoE (64 experts, top-2, inter 640, hidden 1024 -- the real
  plane divisibility: K % 128 == 0, K/2 % 64 == 0, E % 64 == 0, N % 4 == 0);
* hyperconnections hc_count=4 hc_lowrank=320 and the final mix-only mixer;
* MTP draft (``mtp.py``) constructible + forwardable standalone.

Golden compositions, tight-tolerance:

* QSA layer output vs a dense causal GQA recomputation (contexts below
  ``indexer_budget`` must attend every causal token) -- bf16-tight;
* QSA index selection vs a naive per-query block loop -- exact;
* expert dequant vs ``weights.dequant_nvfp4`` with scale2 folded after the
  BF16 cast -- bit-exact;
* grouped expert-major buckets vs the per-expert loop -- same math,
  different reduction order, relative tolerance;
* ``GatedResidual.mix`` vs the manual grouped-norm/down/silu/up/sigmoid
  composition -- exact; ``combine`` vs ``residual + 2*sigmoid(inj/hc)*out``
  -- bit-exact;
* real-checkpoint smoke: one expert tensor pair from
  ``~/models/qwen4exp-flashnext-plain`` is decoded and the
  low-nibble=even-k packing law is verified on real bytes (skipped when the
  checkpoint is absent).

CPU-only: ``VLLM_HW=cpu`` is forced before any vllm_gaudi import; no
``torch.hpu`` op runs, no tensor ever moves to "hpu", no TPC kernel is
rebuilt (so no recipe-cache clearing is needed).

Run::

    .venv/bin/python tools/qwen4exp_flashnext/test_qwen4_exp_forward.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

# CPU-only contract: set before any vllm_gaudi import.
os.environ.setdefault("VLLM_HW", "cpu")
os.environ.setdefault("VLLM_WEIGHT_LOAD_FORCE_SYNC", "false")

import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vllm-gaudi"))

import vllm_gaudi  # noqa: E402,F401  (platform registration)

# The GDN monkeypatch (HPUGatedDeltaNetAttention) is applied by the model
# registry import; the forward path depends on it.
vllm_gaudi.register_models()

CHECKPOINT = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()

# ---------------------------------------------------------------------------
# Micro configuration: 2 layers, everything real except the widths that only
# affect cost.  Divisibility rules of the NVFP4 plane are kept exactly:
# hidden 1024 (K % 128 == 0, K/2 = 512 % 64 == 0), experts 64 (E % 64 == 0,
# the Track C GEMV range floor), intermediate 640 (N % 4 == 0; down_proj K =
# 640 = 5 x 128), PLE head_dim 64 (per row: 32 packed bytes + 4 E4M3 bytes,
# the 1-byte-per-16-values law).
MICRO_TEXT_CONFIG: dict = {
    "hidden_size": 1024,
    "num_hidden_layers": 2,
    "layer_types": ["linear_attention", "full_attention"],
    # GDN: the REAL Qwen3.8 geometry (matches qwen38_native_qk's TP1 target).
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    # QSA full attention + indexer (real head-count/head-dim, micro budget).
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "indexer_budget": 64,
    "indexer_compress_ratio": 4,
    "indexer_n_heads": 4,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 128,
    # MoE: real intermediate, E and top-k micro (plane rules kept).
    "num_experts": 64,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "output_gate_type": "sigmoid",
    "norm_topk_prob": True,
    # Hyperconnections: real hc geometry.
    "hc_count": 4,
    "hc_lowrank": 320,
    # PLE: ple_layer_ids=[1] puts the PLE on layer 0 (layer_idx+1 == 1),
    # the 2-layer scaled analogue of the real [2] -> layer 1.
    "ple_layer_ids": [1],
    "ple_embed_dim": 1024,
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "ngram_vocab_size_base": 512,
    "split_ngram_parts": 16,
    "make_ngram_vocab_size_divisible_by": 128,
    # MTP
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    # Misc fields the vendor modules read.
    "vocab_size": 512,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 4096,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "attn_output_gate": True,
    "attention_bias": False,
    "eos_token_id": 7,
    "bos_token_id": 1,
    "initializer_range": 0.02,
    "mamba_ssm_dtype": "float32",
    "rope_parameters": {
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.5,
        "rope_type": "default",
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
    },
}

_DIST_READY = False
_KEEP_CONTEXT = None


def _setup() -> tuple[object, object, object]:
    """Dist + config context + CPU platform pin (test_model_skeleton pattern)."""
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    from vllm_gaudi.models.qwen4exp.config import Qwen4ExpTextConfig

    global _DIST_READY
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
        model=str(dummy), tokenizer=str(dummy), dtype="bfloat16", seed=0, max_model_len=256,
    )
    cfg = Qwen4ExpTextConfig.from_dict(MICRO_TEXT_CONFIG)
    model_config.hf_config = cfg
    model_config.hf_text_config = cfg
    model_config.model_arch_config = model_config.get_model_arch_config()
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.enable_prefix_caching = False
    if not _DIST_READY:
        tmp = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=1, rank=0, distributed_init_method=f"file://{tmp}",
            local_rank=0, backend="gloo",
        )
        kept = set_current_vllm_config(vllm_config)
        kept.__enter__()  # hold for the whole test: model modules read it lazily
        initialize_model_parallel(1, 1)
        import vllm.platforms as vp

        vp.current_platform.current_device = lambda: torch.device("cpu")  # type: ignore[method-assign]
        global _KEEP_CONTEXT
        _KEEP_CONTEXT = kept  # GC of the handle would reset the config
        _DIST_READY = True
    return vllm_config, cfg, _KEEP_CONTEXT


def _init_params(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for p in module.parameters():
            if p.dtype.is_floating_point:
                p.normal_(0, 0.02)
                p.requires_grad_(False)  # inference-only test graph


def _fill_experts(owner: object, cfg) -> None:
    """Synthetic NVFP4 planes for every PackedRoutedExperts store.

    Scale codes 56..63 decode to E4M3 values in [1.0, 1.75) so activations
    stay in a realistic magnitude range (real checkpoints' block scales sit
    near 1.0; codes near 120 would give values of ~64 and bf16 GEMM rounding
    would swamp the golden comparisons).
    """
    hidden = int(cfg.hidden_size)
    inter = int(cfg.moe_intermediate_size)
    layers = list(owner.model.layers)  # type: ignore[attr-defined]
    for layer in layers:
        store = layer.mlp.experts
        for expert in range(int(cfg.num_experts)):
            for kind, n, k in (
                ("gate_proj", inter, hidden),
                ("up_proj", inter, hidden),
                ("down_proj", hidden, inter),
            ):
                store.load_tensor(
                    f"experts.{expert}.{kind}.weight",
                    torch.randint(0, 256, (n, k // 2), dtype=torch.uint8))
                store.load_tensor(
                    f"experts.{expert}.{kind}.weight_scale",
                    torch.randint(56, 64, (n, k // 16), dtype=torch.uint8))
                store.load_tensor(
                    f"experts.{expert}.{kind}.weight_scale_2",
                    torch.tensor([0.05], dtype=torch.float32))


def _attach_ple(model: torch.nn.Module, cfg) -> int:
    """Attach a synthetic 16-shard host table to the model's PLE layer."""
    from vllm_gaudi.models.qwen4exp.ple_embedding import PLEShardTable
    from vllm_gaudi.models.qwen4exp.ple_runtime import PLENGramEmbedding

    num_heads = (int(cfg.ngram_size) - 1) * int(cfg.heads_per_ngram)  # 16
    head_dim = int(cfg.ple_embed_dim) // num_heads  # 64
    shard_rows = 16
    tables = [
        PLEShardTable(
            torch.randint(0, 256, (shard_rows, head_dim // 2), dtype=torch.uint8),
            torch.randint(56, 64, (shard_rows, head_dim // 16), dtype=torch.uint8),
            pin_memory=False,
        )
        for _ in range(16)
    ]
    ngram = PLENGramEmbedding(
        tables,
        torch.tensor([1, 3, 5, 7], dtype=torch.int64),
        torch.full((num_heads,), shard_rows, dtype=torch.int64),
        torch.arange(num_heads, dtype=torch.int64) * shard_rows,
        heads_per_ngram=int(cfg.heads_per_ngram),
        ngram_size=int(cfg.ngram_size),
        shard_rows=shard_rows,
    )
    attached = 0
    for layer in model.model.layers:
        if getattr(layer, "ple", None) is not None:
            layer.ple.attach_ngram(ngram)
            attached += 1
    return attached


def _gdn_metadata(num_tokens: int, qsl: torch.Tensor, *, is_prompt: bool,
                  has_initial: list[bool]) -> SimpleNamespace:
    """The fields ``HPUGatedDeltaNetAttention._extract_metadata`` reads."""
    n = qsl.numel() - 1
    return SimpleNamespace(
        is_prompt=is_prompt,
        load_indices_tensor=torch.arange(n, dtype=torch.int32),
        store_indices_tensor=torch.arange(n, dtype=torch.int32),
        query_start_loc_p=qsl,
        has_initial_states_p=torch.tensor(has_initial),
        padding_mask_flat=None,
        direct_gdn_state=False,
        num_accepted_tokens=None,
        gdn_initial_state_indices=None,
        dflash_full_query=False,
    )


def _bind_gdn_cache(layer: torch.nn.Module, slots: int = 2) -> None:
    conv_shape, ssm_shape = layer.get_state_shape()
    conv_dtype, ssm_dtype = layer.get_state_dtype()
    layer.kv_cache = (
        torch.zeros((slots, *conv_shape), dtype=conv_dtype),
        torch.zeros((slots, *ssm_shape), dtype=ssm_dtype),
    )


class _ModelFixture:
    """One built, weight-initialized, expert- and PLE-attached micro model.

    A process-wide singleton: the PLE layer registers its prefix in the
    compilation config's static forward context, so a second build with the
    same prefixes would trip the duplicate-layer check. Every test re-runs
    its own prefill, so sharing the model (and its GDN cache rebinding) is
    deterministic.
    """

    _instance: "_ModelFixture | None" = None

    def __init__(self) -> None:
        from vllm_gaudi.models.qwen4_exp import Qwen4ExpForCausalLM

        self.vllm_config, self.cfg, _ = _setup()
        torch.manual_seed(0)
        torch.set_default_dtype(torch.bfloat16)
        self.model = Qwen4ExpForCausalLM(vllm_config=self.vllm_config, prefix="")
        _init_params(self.model)
        _fill_experts(self.model, self.cfg)
        assert _attach_ple(self.model, self.cfg) == 1
        self.gdn = self.model.model.layers[0].linear_attn
        _bind_gdn_cache(self.gdn)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                qsl: torch.Tensor, *, is_prompt: bool, has_initial: list[bool]):
        from vllm.forward_context import set_forward_context

        meta = _gdn_metadata(input_ids.shape[0], qsl, is_prompt=is_prompt,
                             has_initial=has_initial)
        with set_forward_context(meta, self.vllm_config,
                                 num_tokens=input_ids.shape[0]):
            return self.model(input_ids=input_ids, positions=positions,
                              query_start_loc=qsl)


def fixture() -> _ModelFixture:
    if _ModelFixture._instance is None:
        _ModelFixture._instance = _ModelFixture()
    return _ModelFixture._instance


# ---------------------------------------------------------------------------
# 1. end-to-end decoder forward
# ---------------------------------------------------------------------------
def test_two_layer_forward_logits() -> None:
    fx = fixture()
    cfg = fx.cfg
    for batch, seq in ((2, 16), (1, 64)):
        torch.manual_seed(10 + batch * seq)
        ids = torch.randint(0, int(cfg.vocab_size), (batch * seq,))
        positions = torch.arange(seq).repeat(batch)
        qsl = torch.arange(0, batch + 1, dtype=torch.int32) * seq
        out = fx.forward(ids, positions, qsl, is_prompt=True,
                         has_initial=[False] * batch)
        assert out.shape == (batch * seq, int(cfg.hidden_size)), out.shape
        assert torch.isfinite(out.float()).all()
        logits = fx.model.compute_logits(out)
        assert logits is not None
        assert logits.shape == (batch * seq, int(cfg.vocab_size)), logits.shape
        assert torch.isfinite(logits.float()).all(), "logits must be finite"
        print(f"  forward B={batch} S={seq}: hidden {tuple(out.shape)}, "
              f"logits {tuple(logits.shape)} finite")


def test_gdn_state_written_and_decode_continues() -> None:
    fx = fixture()
    cfg = fx.cfg
    seq = 16
    ids = torch.randint(0, int(cfg.vocab_size), (seq,))
    positions = torch.arange(seq)
    qsl = torch.tensor([0, seq], dtype=torch.int32)
    fx.forward(ids, positions, qsl, is_prompt=True, has_initial=[False])
    conv, ssm = fx.gdn.kv_cache
    assert float(conv.detach().abs().sum()) > 0, "GDN conv state was not written"
    assert float(ssm.detach().abs().sum()) > 0, "GDN ssm state was not written"
    # one-token decode continuation: cache carry through the same layer set
    decode = fx.forward(ids[-1:], torch.tensor([seq]),
                        torch.tensor([0, 1], dtype=torch.int32),
                        is_prompt=False, has_initial=[True])
    assert decode.shape == (1, int(cfg.hidden_size))
    assert torch.isfinite(decode.float()).all()
    print("  GDN prefill wrote conv+ssm state; 1-token decode continuation finite")


# ---------------------------------------------------------------------------
# 2. MTP draft standalone
# ---------------------------------------------------------------------------
def test_mtp_standalone_forward() -> None:
    from vllm_gaudi.models.qwen4exp.mtp import Qwen4ExpMTP

    vllm_config, cfg, _ = _setup()
    torch.manual_seed(1)
    torch.set_default_dtype(torch.bfloat16)
    draft = Qwen4ExpMTP(vllm_config=vllm_config, prefix="")
    _init_params(draft)
    assert not any(l.ple is not None for l in draft.model.layers), \
        "draft layers must never own a PLE table (ple_layer_ids=[1] trunk)"
    _fill_experts(draft, cfg)
    tokens = 8
    multi = torch.randn(tokens, int(cfg.hc_count) * int(cfg.hidden_size))
    ids = torch.randint(0, int(cfg.vocab_size), (tokens,))
    sample, multi_out = draft(
        input_ids=ids, positions=torch.arange(tokens), hidden_states=multi)
    assert sample.shape == (tokens, int(cfg.hidden_size))
    assert multi_out.shape == (tokens, int(cfg.hc_count) * int(cfg.hidden_size))
    assert torch.isfinite(sample.float()).all()
    logits = draft.compute_logits(sample)
    assert logits is not None and logits.shape == (tokens, int(cfg.vocab_size))
    assert torch.isfinite(logits.float()).all()
    print(f"  MTP draft: sample {tuple(sample.shape)}, multi {tuple(multi_out.shape)}, "
          f"logits finite")


# ---------------------------------------------------------------------------
# 3. QSA goldens
# ---------------------------------------------------------------------------
def test_qsa_matches_dense_causal_attention() -> None:
    """Below ``indexer_budget`` the sparse selection must equal dense causal."""
    fx = fixture()
    cfg = fx.cfg
    seq = 12
    ids = torch.randint(0, int(cfg.vocab_size), (seq,))
    positions = torch.arange(seq)
    qsl = torch.tensor([0, seq], dtype=torch.int32)
    hidden = fx.forward(ids, positions, qsl, is_prompt=True, has_initial=[False])
    qsa = fx.model.model.layers[1].self_attn
    out = qsa(positions=positions, hidden_states=hidden, query_start_loc=qsl)

    qkv = qsa.qkv_proj(hidden)[0]
    q, k, v, gate = qsa._project_qkv_gate(qkv, positions)
    num_tokens = hidden.shape[0]
    qf = q.view(num_tokens, qsa.num_heads, qsa.head_dim).float()
    kf = k.view(num_tokens, qsa.num_kv_heads, qsa.head_dim).float()
    vf = v.view(num_tokens, qsa.num_kv_heads, qsa.head_dim).float()
    n_rep = qsa.num_heads // qsa.num_kv_heads
    kexp = kf.repeat_interleave(n_rep, dim=1).permute(1, 2, 0)  # [H, D, S]
    scores = torch.matmul(qf.transpose(0, 1), kexp) * qsa.scaling  # [H, T, S]
    causal = torch.tril(torch.ones(num_tokens, num_tokens, dtype=torch.bool))
    scores = scores.masked_fill(~causal, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    dense = torch.matmul(attn, vf.repeat_interleave(n_rep, dim=1).permute(1, 0, 2))
    dense = dense.transpose(0, 1)  # [T, H, D]
    if gate is not None:
        g = torch.sigmoid(gate.float()).view(num_tokens, qsa.num_heads, qsa.head_dim)
        dense = dense * g
    golden = qsa.o_proj(dense.reshape(num_tokens, -1).to(hidden.dtype))[0]
    delta = (out.float() - golden.float()).abs()
    rel = float(delta.max() / golden.float().abs().max().clamp(min=1e-6))
    print(f"  QSA vs dense causal: absmax {float(delta.max()):.3e} rel {rel:.2e}")
    assert rel < 5e-3, "sparse attention below budget must equal dense causal"


def test_qsa_selection_matches_naive_loop() -> None:
    fx = fixture()
    cfg = fx.cfg
    import math

    from vllm_gaudi.models.qwen4exp import qsa_math

    seq = 12
    ids = torch.randint(0, int(cfg.vocab_size), (seq,))
    positions = torch.arange(seq)
    qsl = torch.tensor([0, seq], dtype=torch.int32)
    hidden = fx.forward(ids, positions, qsl, is_prompt=True, has_initial=[False])
    qsa = fx.model.model.layers[1].self_attn
    q_idx, _raw_k = qsa.indexer.project_qk(hidden, positions, qsa.rotary_emb)
    state = qsa._bringup[0]
    assert state is not None
    block_cos_sin = qsa._block_cos_sin(state, q_idx)
    indices = qsa_math.select_qsa_indices(
        q_idx, state.raw_keys, torch.arange(seq),
        k_norm_weight=qsa.indexer.k_layernorm.weight,
        eps=qsa.indexer.k_layernorm.variance_epsilon,
        block_cos_sin=block_cos_sin,
        token_budget=qsa.indexer.token_topk,
        compress_ratio=qsa.indexer.compress_ratio,
    )
    ratio = qsa.indexer.compress_ratio
    head_dim = qsa.indexer.index_head_dim
    naive = torch.full((seq, indices.shape[-1]), -1, dtype=torch.int32)
    for t in range(seq):
        visible = t + 1
        n_complete = visible // ratio
        tail_len = visible - n_complete * ratio
        picks: list[int] = []
        if n_complete:
            scores = []
            for b in range(n_complete):
                pooled = state.raw_keys[b * ratio:(b + 1) * ratio].float().mean(0)
                pooled = qsa_math.gemma_rmsnorm(
                    pooled, qsa.indexer.k_layernorm.weight,
                    qsa.indexer.k_layernorm.variance_epsilon)
                pooled = qsa_math.apply_partial_rope(
                    pooled.unsqueeze(0), block_cos_sin[b].unsqueeze(0))[0]
                dot = torch.relu((q_idx[t].float() * pooled.float()).sum(-1)).sum()
                scores.append(float(dot) / math.sqrt(head_dim))
            kblocks = min(qsa.indexer.token_topk // ratio, n_complete)
            order = sorted(range(n_complete), key=lambda b: (-scores[b], b))[:kblocks]
            for b in order:
                picks.extend(b * ratio + j for j in range(ratio))
        picks.extend(n_complete * ratio + j for j in range(tail_len))
        picks = picks[: naive.shape[1]]
        naive[t, : len(picks)] = torch.tensor(picks, dtype=torch.int32)
    assert torch.equal(indices, naive), "packed selection != naive block loop"
    print("  QSA index selection == naive per-query block loop (exact)")


# ---------------------------------------------------------------------------
# 4. MoE goldens
# ---------------------------------------------------------------------------
def _micro_store(E: int = 4, N: int = 64, K: int = 128):
    from vllm_gaudi.models.qwen4exp import packed_moe

    planes = {}
    for kind, n, k in (("gate_proj", N, K), ("up_proj", N, K), ("down_proj", K, N)):
        planes[kind] = (
            torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8),
            torch.randint(56, 64, (E, n, k // 16), dtype=torch.uint8),
        )
    scale2 = torch.rand(E) * 0.1 + 0.05
    store = packed_moe.PackedRoutedExperts(E)
    for e in range(E):
        for kind in ("gate_proj", "up_proj", "down_proj"):
            p, s = planes[kind]
            store.load_tensor(f"experts.{e}.{kind}.weight", p[e].clone())
            store.load_tensor(f"experts.{e}.{kind}.weight_scale", s[e].clone())
            store.load_tensor(f"experts.{e}.{kind}.weight_scale_2",
                              scale2[e:e + 1].clone())
    return store, planes, scale2


def test_expert_dequant_bit_exact() -> None:
    from vllm_gaudi.models.qwen4exp.weights import dequant_nvfp4

    torch.manual_seed(2)
    E, N, K = 4, 64, 128
    store, planes, scale2 = _micro_store(E, N, K)
    ids = torch.tensor([1, 3])
    got = store.dequant("gate_proj", ids)
    pk, sc = planes["gate_proj"]
    golden = torch.stack([
        (dequant_nvfp4(pk[i], sc[i], 1.0)
         * scale2[i].to(torch.bfloat16)).reshape(N, K)  # scale2 after the BF16 cast
        for i in ids.tolist()])
    assert got.dtype == torch.bfloat16 and got.shape == (2, N, K)
    assert torch.equal(got, golden), "store dequant != weights.dequant_nvfp4 golden"
    print("  expert dequant bit-exact vs dequant_nvfp4 (scale2 folded after cast)")


def test_grouped_buckets_match_expert_loop() -> None:
    from vllm_gaudi.models.qwen4exp import packed_moe

    torch.manual_seed(3)
    E, N, K = 4, 64, 128
    store, _planes, _scale2 = _micro_store(E, N, K)
    hidden = torch.randn(6, K, dtype=torch.bfloat16)
    topk_ids = torch.tensor([[1, 3], [0, 1], [2, 2], [3, 0], [1, 1], [0, 2]],
                            dtype=torch.int32)
    weights = torch.softmax(torch.randn(6, 2).float(), -1).to(torch.bfloat16)
    unique, local = packed_moe.compact_expert_ids(topk_ids, E)
    gate = store.dequant("gate_proj", unique)
    up = store.dequant("up_proj", unique)
    down = store.dequant("down_proj", unique)
    out_loop = packed_moe.apply_routed_experts(hidden, local, weights, gate, up, down)
    out_group = packed_moe.apply_routed_experts_grouped(
        hidden, local, weights, gate, up, down)
    delta = (out_loop.float() - out_group.float()).abs()
    rel = float(delta.max() / out_loop.float().abs().max().clamp(min=1e-6))
    print(f"  grouped buckets vs per-expert loop: rel {rel:.2e}")
    assert rel < 5e-2, "grouped GEMM path drifted from the per-expert loop"
    slots = packed_moe.grouped_from_ids(hidden[:2], topk_ids[:2], weights[:2], store)
    assert slots.shape == (2, K) and torch.isfinite(slots.float()).all()
    print("  decode slot path (grouped_from_ids) finite")


def test_real_checkpoint_expert_smoke() -> None:
    """One real expert tensor pair from the converted shard decodes and the
    low-nibble=even-k packing law holds on real bytes (skipped w/o ckpt)."""
    if not (CHECKPOINT / "model.safetensors.index.json").is_file():
        print("  real checkpoint not present; smoke skipped")
        return
    from safetensors import safe_open

    from vllm_gaudi.models.qwen4exp.csf_math import e2m1_value_lut, e4m3_value_lut
    from vllm_gaudi.models.qwen4exp.weights import dequant_nvfp4

    index = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    base = "model.language_model.layers.0.mlp.experts.7.gate_proj"

    def get(name: str) -> torch.Tensor:
        shard = CHECKPOINT / "tensors" / weight_map[name]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    w = get(base + ".weight")
    s = get(base + ".weight_scale").view(torch.uint8)
    s2 = float(get(base + ".weight_scale_2").reshape(-1)[0])
    inter, hidden_half = w.shape
    assert s.shape == (inter, hidden_half // 8), "one E4M3 byte per 16 inputs"
    dec = dequant_nvfp4(w, s, s2)
    assert dec.shape == (inter, hidden_half * 2)
    assert torch.isfinite(dec).all()
    lut, scale_lut = e2m1_value_lut(), e4m3_value_lut()
    byte = int(w[0, 0])
    code = int(s[0, 0])
    k0 = (lut[byte & 0xF] * scale_lut[code] * s2).to(torch.bfloat16)
    k1 = (lut[byte >> 4] * scale_lut[code] * s2).to(torch.bfloat16)
    assert dec[0, 0] == k0 and dec[0, 1] == k1, "nibble order broke on real bytes"
    print(f"  real expert {base}: planes {tuple(w.shape)}+{tuple(s.shape)} decode, "
          f"low nibble = even k verified")


# ---------------------------------------------------------------------------
# 5. hyperconnection goldens
# ---------------------------------------------------------------------------
def test_hyperconnection_mix_and_combine_golden() -> None:
    from vllm_gaudi.models.qwen4exp.hyperconnection_amd import (
        GatedResidual,
        HyperConnectionConfig,
        hc_grouped_gemma_rmsnorm,
    )

    _setup()
    torch.manual_seed(4)
    hc = 4
    hidden = 64
    lowrank = 32
    hc_mod = GatedResidual(
        HyperConnectionConfig(hc_count=hc, hidden_size=hidden,
                              params_dtype=torch.bfloat16,
                              hc_lowrank=lowrank, rms_norm_eps=1e-6,
                              hc_per_branch_norm=True),
        prefix="golden",
    )
    with torch.no_grad():
        for param in hc_mod.parameters():
            param.normal_(0, 0.1)
            param.requires_grad_(False)  # inference-only test graph
    x = torch.randn(5, hc * hidden, dtype=torch.bfloat16)
    hs, block_in, injection = hc_mod.mix(x)

    # manual mix: grouped Gemma norm -> merged down+inject -> silu(x/hc) ->
    # up -> sigmoid gate -> mean over streams
    xn = hc_grouped_gemma_rmsnorm(x, hc_mod.hc_norm.weight, 1e-6, hc)
    merged = torch.nn.functional.linear(
        xn, hc_mod.input_mix_weight_down_block_inject.weight)
    low, inj, pad = merged.split([lowrank, hc, hc_mod.pad_size], dim=-1)
    assert int(pad.shape[-1]) == hc_mod.pad_size and float(pad.abs().sum()) >= 0
    lo = torch.nn.functional.silu(low.float() / hc).to(x.dtype)
    gate = torch.nn.functional.linear(lo, hc_mod.input_mix_weight_up.weight)
    gate4 = torch.sigmoid(gate.float().view(5, hc, hidden))
    mixed = (gate4 * xn.float().view(5, hc, hidden)).mean(1).to(x.dtype)
    assert torch.equal(block_in, mixed), "GatedResidual.mix != manual composition"
    assert torch.equal(injection, inj), "injection column must be the merged tail"

    # manual combine: residual + block_output * (2 * sigmoid(inj / hc))
    block_out = torch.randn(5, hidden, dtype=torch.bfloat16)
    combined = hc_mod.combine(hs, block_out, injection)
    injf = 2.0 * torch.sigmoid(injection.float() / hc)
    manual = (hs.float().view(5, hc, hidden)
              + block_out.float().unsqueeze(1) * injf.float().unsqueeze(-1))
    assert torch.equal(combined, manual.flatten(-2).to(torch.bfloat16)), \
        "GatedResidual.combine != residual + 2*sigmoid(inj/hc)*out"
    print("  hyperconnection mix (exact) and combine (bit-exact) match manual math")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def main() -> int:
    test_two_layer_forward_logits()
    test_gdn_state_written_and_decode_continues()
    test_mtp_standalone_forward()
    test_qsa_matches_dense_causal_attention()
    test_qsa_selection_matches_naive_loop()
    test_expert_dequant_bit_exact()
    test_grouped_buckets_match_expert_loop()
    test_real_checkpoint_expert_smoke()
    test_hyperconnection_mix_and_combine_golden()
    print("test_qwen4_exp_forward: all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
