"""CPU tests for attention KV groups sharing one decode mask/mapping (VLLM_HPU_SHARE_GROUP_MASKS).

Code under test is AST-extracted from the model runner in this repository.
Run from the repository root: python -m pytest -q validation
"""
import dataclasses, os, random, sys, types

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_shared_prefix as T  # noqa: E402

sp, agr = T.sp, T.agr
NAMES = T.RUNNER_NAMES
PROC = ["process_metadata", "_set_block_mapping", "_set_attn_bias"]


def make_runner(rng, n, width, shared_prefix, share_masks, padded=None):
    """Runner stand-in plus the main (first group) metadata exactly as _create_decode_input_data builds it."""
    env = types.SimpleNamespace(VLLM_HPU_SHARED_PREFIX_DECODE=shared_prefix, VLLM_HPU_FAST_DECODE_PREP="",
                                VLLM_HPU_FAST_DECODE_PREP_VERIFY=0, VLLM_HPU_SHARE_GROUP_MASKS=share_masks)
    funcs, logs, ns = T.class_methods(T.RUN_SRC, NAMES, {"gaudi_envs": env, "AttentionSpec": T.AttentionSpec,
                                                          "is_fake_hpu": lambda: False})
    r = types.SimpleNamespace()
    for k, f in funcs.items():
        setattr(r, k, types.MethodType(f, r))
    r.attn_block_size, r._PAD_BLOCK_ID = 128, 5000
    r.use_contiguous_pa = r.interleaved_sliding_window = r.model_has_chunked_attention = False
    r.use_framework_kv_cache_layout = r.warmup_mode = False
    r.bucketing_manager = T.Buckets()
    r.model_config = types.SimpleNamespace(dtype=torch.bfloat16)
    r.get_dp_padding = lambda size: 0
    r.device = "cpu"
    r.num_mamba_like_layers = 48
    r._legacy_attention_group_ids = (10, 11, 12, 13)
    r.kv_cache_config = types.SimpleNamespace(kv_cache_groups=[
        types.SimpleNamespace(kv_cache_spec=T.AttentionSpec() if g in (10, 11, 12, 13) else object())
        for g in range(15)])
    padded = padded or n
    ctx = [rng.choice([0, 125, 128, rng.randint(0, 3000)]) for _ in range(n)]
    qlen = [rng.choice([width, width, width, rng.randint(1, width)]) for _ in range(n)]
    pages = [(c + q + 127) // 128 for c, q in zip(ctx, qlen)]
    tables = []
    for g in range(15):
        t = torch.zeros((4, max(pages) + 4), dtype=torch.int32)
        pool = rng.sample(range(5000), sum(pages)); k = 0
        for i, p in enumerate(pages):
            t[i, :p] = torch.tensor(pool[k:k + p], dtype=torch.int32); k += p
        tables.append(T.FakeTable(t))
    req_ids = [f"q{i}" for i in range(n)]
    r.input_batch = types.SimpleNamespace(block_table=tables, req_ids=req_ids,
                                          req_id_to_index={q: i for i, q in enumerate(req_ids)})
    # main group buffers
    main_table = tables[10].get_cpu_tensor().numpy()[:n, :max(pages)]
    if shared_prefix and width > 1:
        blocks, owners, usage = r._shared_prefix_buffers(main_table, ctx, qlen, width, padded)
    else:
        rows = [main_table[i, :pages[i]].tolist() for i in range(n)]
        # main-group slots as _create_decode_input_data computes them
        slots = torch.empty((padded, width), dtype=torch.int64)
        dummy = iter(range(5000 * 128, 5000 * 128 + 10**6))
        prefixes = []
        for row in range(padded):
            for tok in range(width):
                if row < n and tok < qlen[row]:
                    pos = ctx[row] + tok
                    slots[row, tok] = rows[row][pos // 128] * 128 + pos % 128
                    prefixes.append(rows[row][:pos // 128 + 1])
                elif row < n:
                    slots[row, tok] = 5000 * 128 + next(dummy) % 128
                    prefixes.append([5000])
                else:
                    slots[row, tok] = 5000 * 128
        blocks, owners, usage = r.get_habana_paged_attn_buffers(prefixes, slots.view(-1, 1).tolist(), padded * width,
                                                                block_size=128)
    main = T.Meta(slot_mapping=torch.zeros(1), block_list=blocks.clone(), block_groups=owners.clone(),
                  block_usage=usage.clone())
    return r, ctx, qlen, pages, padded, main, (owners, usage)


def route(r, main, ctx, qlen, width, padded, buffers):
    n = len(ctx)
    return r._route_legacy_decode_groups(main, n, qlen + [0] * (padded - n), np.asarray(ctx), width, padded,
                                         main_buffers=buffers)


@pytest.mark.parametrize("seed", range(150))
def test_groups_reuse_the_first_groups_tensors(seed):
    rng = random.Random(seed)
    n, width = rng.randint(1, 2), rng.choice([8, 8, 1])
    shared_prefix = rng.random() < 0.6
    state = rng.getstate()
    r, ctx, qlen, pages, padded, main, buffers = make_runner(rng, n, width, shared_prefix, True)
    rng.setstate(state)
    r0, _, _, _, _, main0, buffers0 = make_runner(rng, n, width, shared_prefix, False)
    out, ref = route(r, main, ctx, qlen, width, padded, buffers), route(r0, main0, ctx, qlen, width, padded, buffers0)
    full_rows = all(q == width for q in qlen)
    for gid in (11, 12, 13):
        g, g0 = out.attention_groups[gid], ref.attention_groups[gid]
        for field in ("slot_mapping", "block_list", "block_groups", "block_usage"):
            a, b = getattr(g, field), getattr(g0, field)
            assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), (gid, field)
        assert g0.block_groups is not main0.block_groups           # switch off: every group has its own tensors
        shared = g.block_groups is main.block_groups and g.block_usage is main.block_usage
        if shared_prefix and width > 1 or full_rows:
            assert shared                                           # same page layout -> same tensors
        assert shared == (torch.equal(g0.block_groups, main0.block_groups)
                          and torch.equal(g0.block_usage, main0.block_usage))


def test_without_main_buffers_nothing_is_shared():
    rng = random.Random(1)
    r, ctx, qlen, pages, padded, main, buffers = make_runner(rng, 1, 8, True, True)
    out = route(r, main, ctx, qlen, 8, padded, None)
    assert all(out.attention_groups[g].block_groups is not main.block_groups for g in (11, 12, 13))


# ----------------------------------------------------------------------------- device-side processing
def processor(fake_hpu=True):
    funcs, logs, ns = T.class_methods(T.RUN_SRC, PROC, {"is_fake_hpu": lambda: fake_hpu})
    p = types.SimpleNamespace(block_size=128, interleaved_sliding_window=False, prefill_use_fusedsdpa=True, calls=0)
    for k, f in funcs.items():
        setattr(p, k, types.MethodType(f, p))
    inner = p._set_block_mapping

    def counting(*a, **k):
        p.calls += 1
        return inner(*a, **k)

    p._set_block_mapping = counting
    return p


@dataclasses.dataclass
class Trimmed(T.Meta):
    is_prompt: bool = False
    window_block_list: object = None


def trimmed(meta):
    fields = {f.name: getattr(meta, f.name) for f in dataclasses.fields(T.Meta)}
    groups = fields.pop("attention_groups")
    return Trimmed(**fields, attention_groups=None if groups is None else tuple(
        None if g is None else trimmed(g) for g in groups))


@pytest.mark.parametrize("seed", range(120))
def test_processing_reuses_mask_and_mapping(seed):
    rng = random.Random(100 + seed)
    n, width = rng.randint(1, 2), rng.choice([8, 8, 1])
    shared_prefix = rng.random() < 0.6
    state = rng.getstate()
    r, ctx, qlen, pages, padded, main, buffers = make_runner(rng, n, width, shared_prefix, True)
    rng.setstate(state)
    r0, _, _, _, _, main0, buffers0 = make_runner(rng, n, width, shared_prefix, False)
    out, ref = route(r, main, ctx, qlen, width, padded, buffers), route(r0, main0, ctx, qlen, width, padded, buffers0)
    p, p0 = processor(), processor()
    got = p.process_metadata(trimmed(out), padded * width, 1, torch.device("cpu"), torch.bfloat16)
    want = p0.process_metadata(trimmed(ref), padded * width, 1, torch.device("cpu"), torch.bfloat16)
    shared = sum(out.attention_groups[g].block_usage is main.block_usage for g in (11, 12, 13))
    assert p0.calls == 4 and p.calls == 4 - shared
    for gid in (10, 11, 12, 13):
        a, b = got.attention_groups[gid], want.attention_groups[gid]
        for field in ("block_mapping", "attn_bias", "block_groups", "block_list", "slot_mapping", "block_usage"):
            x, y = getattr(a, field), getattr(b, field)
            assert x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y), (gid, field)
    if shared:
        assert got.attention_groups[11].attn_bias is got.attn_bias or shared < 3
        expected_rows = padded if (shared_prefix and width > 1) else padded * width
        assert got.block_mapping.shape[1] == expected_rows


def test_attention_output_is_unchanged_by_sharing():
    """Each group's attention over its own pages gives the same result whether or not the mask is shared."""
    rng = random.Random(5)
    for _ in range(12):
        n, width = rng.randint(1, 2), 8
        state = rng.getstate()
        r, ctx, qlen, pages, padded, main, buffers = make_runner(rng, n, width, True, True)
        rng.setstate(state)
        r0, _, _, _, _, main0, buffers0 = make_runner(rng, n, width, True, False)
        out = processor().process_metadata(trimmed(route(r, main, ctx, qlen, width, padded, buffers)), padded * width,
                                           1, torch.device("cpu"), torch.float64)
        ref = processor().process_metadata(trimmed(route(r0, main0, ctx, qlen, width, padded, buffers0)),
                                           padded * width, 1, torch.device("cpu"), torch.float64)
        gen = torch.Generator().manual_seed(7)
        kc = torch.randn(5001 * 128, 2, 8, generator=gen, dtype=torch.float64)
        vc = torch.randn(5001 * 128, 2, 8, generator=gen, dtype=torch.float64)
        q = torch.randn(padded, width, 4 * 8, generator=gen, dtype=torch.float64)
        for gid in (11, 12, 13):
            res = []
            for meta in (out.attention_groups[gid], ref.attention_groups[gid]):
                res.append(T.OPS.flat_pa_query_block(query=q, block_list=meta.block_list,
                                                     block_mapping=meta.block_mapping, block_bias=meta.attn_bias,
                                                     block_groups=meta.block_groups,
                                                     **T.attention_args(kc, vc, 128, 8 ** -0.5)))
            assert torch.equal(res[0], res[1])


def test_unbacked_marking_switch_is_wired():
    src = T.RUN_SRC
    assert src.count("if self.flatten_input and not gaudi_envs.VLLM_HPU_SKIP_UNBACKED_BLOCK_LIST:") == 1
    assert src.count("_mark_unbacked_dim0(attn_md.block_list)") == 1
