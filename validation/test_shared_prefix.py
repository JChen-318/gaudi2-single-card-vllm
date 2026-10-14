"""CPU tests for shared-prefix decode attention (VLLM_HPU_SHARED_PREFIX_DECODE).

The code under test is AST-extracted from the sources in this repository, so what runs here is the shipped text:
`flat_pa` / `flat_pa_query_block` from vllm_gaudi/extension/ops.py and `_set_block_mapping`,
`get_habana_paged_attn_buffers`, `_route_legacy_decode_groups` from the model runner. The flattened layout (one
causal page prefix per verification token) is the reference for the shared layout. No Gaudi device or Habana
package is needed; torch, numpy and pytest are enough.

Run from the repository root: python -m pytest -q validation
"""
import ast, dataclasses, importlib.util, itertools, math, os, random, sys, textwrap, time, types

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.environ.get("VLLM_GAUDI_SRC", os.path.join(HERE, "..", "vllm-gaudi", "vllm_gaudi"))
WORKER = os.path.join(PKG, "v1", "worker")


def load_module(name, path):
    """Import one source file on its own, without importing the vllm_gaudi package (which needs the HPU stack)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sp = load_module("decode_shared_prefix", os.path.join(WORKER, "decode_shared_prefix.py"))
agr = load_module("attention_group_routing", os.path.join(WORKER, "attention_group_routing.py"))

OPS_SRC = open(os.path.join(PKG, "extension", "ops.py"), encoding="utf-8").read()
RUN_SRC = open(os.path.join(WORKER, "hpu_model_runner.py"), encoding="utf-8").read()


def flatten(in_list):
    return list(itertools.chain(*in_list))


def pad_list(input, target_len, val_generator):
    padding = target_len - len(input)
    if padding > 0:
        input.extend(itertools.islice(val_generator, padding))
    return input


def custom_tuple_replace(obj, _name, **kwargs):
    return dataclasses.replace(obj, **kwargs)


def round_up(value, k):
    return (value + k - 1) // k * k


# ----------------------------------------------------------------------------- extracted device code
CONFIG = types.SimpleNamespace(fp32_softmax=False, fused_block_softmax=False, fused_block_softmax_adjustment=False,
                               per_token_kv_scaling_support=False)
OPS_ENV = types.SimpleNamespace(VLLM_HPU_TP2_GQA_COMPACT_KV=False, VLLM_HPU_GQA_COMPACT_KV=False,
                                VLLM_HPU_GQA_NATIVE_MATMUL=False, VLLM_HPU_GQA_COMPACT_KV_MAX_BATCH=1,
                                VLLM_HPU_PAGED_KV_DUAL_GATHER=False)


def get_kv_fetch_extra_args(**kwargs):
    if not CONFIG.per_token_kv_scaling_support:
        kwargs.pop('scales', None)
    return kwargs


def module_functions(src, names):
    ns = {"torch": torch, "math": math, "gaudi_envs": OPS_ENV, "get_config": lambda: CONFIG, "is_hpu_gaudi2": False,
          "get_kv_fetch_extra_args": get_kv_fetch_extra_args, "Optional": object}
    found = set()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(ast.get_source_segment(src, node), ns)
            found.add(node.name)
    assert found == set(names), set(names) - found
    return types.SimpleNamespace(**{n: ns[n] for n in names})


OPS = module_functions(OPS_SRC, ["grouped_max", "b2b_impl", "batch2block", "block2batch", "matmul_shape",
                                  "pipelined_pa", "flat_pa", "flat_pa_query_block"])


def class_methods(src, names, extra=None):
    logs = []
    ns = {"torch": torch, "np": np, "math": math, "itertools": itertools, "flatten": flatten, "pad_list": pad_list,
          "Any": object, "Optional": object, "Union": object, "HPUAttentionMetadataV1": object,
          "custom_tuple_replace": custom_tuple_replace, "is_fake_hpu": lambda: True, "round_up": round_up,
          "decode_shared_prefix": sp, "decode_group_addresses": agr.decode_group_addresses,
          "replace": dataclasses.replace, "async_h2d_copy": lambda t, device=None, dtype=None: t.clone(),
          "logger": types.SimpleNamespace(warning=lambda *a, **k: logs.append(("warning", a)),
                                          info=lambda *a, **k: logs.append(("info", a)),
                                          error=lambda *a, **k: logs.append(("error", a)))}
    ns.update(extra or {})
    out = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in names:
                    exec(textwrap.dedent(ast.get_source_segment(src, item)), ns)
                    out[item.name] = ns[item.name]
    assert set(out) == set(names), set(names) - set(out)
    return out, logs, ns


SET_MAPPING = class_methods(RUN_SRC, ["_set_block_mapping"])[0]["_set_block_mapping"]
REF_BUFFERS = class_methods(RUN_SRC, ["get_habana_paged_attn_buffers"])[0]["get_habana_paged_attn_buffers"]


@dataclasses.dataclass
class Meta:
    block_list: object = None
    block_groups: object = None
    block_usage: object = None
    block_mapping: object = None
    attn_bias: object = None
    slot_mapping: object = None
    attention_groups: object = None
    block_size: int = 128
    window_block_usage: object = None
    window_block_groups: object = None
    chunked_block_usage: object = None
    chunked_block_groups: object = None


class Buckets:
    """Seed decode buckets like the deployment (1,2,4,..,64 then multiples of `step`), up to `limit` pages."""

    def __init__(self, step=128, limit=None):
        self.step, self.limit = step, limit

    def find_decode_bucket(self, batch_size, num_blocks, seed_buckets=False):
        capped = num_blocks if self.limit is None else min(num_blocks, self.limit)
        ramp = [b for b in (1, 2, 4, 8, 16, 32, 64) if b < self.step]
        size = next((b for b in ramp if b >= capped), None) or -(-capped // self.step) * self.step
        return (batch_size, 1, size)


def processor(block_size):
    return types.SimpleNamespace(block_size=block_size)


def device_metadata(block_list, block_groups, block_usage, block_size, token_rows, dtype):
    """What HpuModelAdapter.forward does on the device before the model runs (fake-HPU branch on CPU)."""
    meta = Meta(block_list=block_list, block_groups=block_groups.clone(), block_usage=block_usage.to(dtype),
                block_size=block_size)
    return SET_MAPPING(processor(block_size), meta, token_rows, torch.device("cpu"), dtype)


def fetch(cache, blocks):
    return cache.index_select(0, blocks)


def attention_args(key_cache, value_cache, block_size, scale):
    return dict(key_cache=key_cache, value_cache=value_cache, block_size=block_size, scale=scale,
                matmul_qk_op=torch.matmul, matmul_av_op=torch.matmul, batch2block_matmul_op=torch.matmul,
                block2batch_matmul_op=torch.matmul, keys_fetch_func=fetch, values_fetch_func=fetch,
                position_bias=None, sinks=None, k_scales=None, v_scales=None)


class Case:
    """One decode batch: `n` live requests padded to `padded` rows, `width` verification tokens per row."""

    def __init__(self, rng, block_size, width, n, padded, ctx, qlen, kv_heads, q_heads, head, dtype, pool_pages=None):
        self.block_size, self.width, self.n, self.padded = block_size, width, n, padded
        self.ctx, self.qlen = ctx, qlen
        self.kv_heads, self.q_heads, self.head, self.dtype = kv_heads, q_heads, head, dtype
        self.pages = [(c + q + block_size - 1) // block_size for c, q in zip(ctx, qlen)]
        self.pad_block = pool_pages or (sum(self.pages) + 7)
        pool = rng.sample(range(self.pad_block), sum(self.pages))
        table = np.full((n + 1, max(self.pages) + 2), -5, dtype=np.int32)
        k = 0
        for i, p in enumerate(self.pages):
            table[i, :p] = pool[k:k + p]; k += p
        self.table = table
        gen = torch.Generator().manual_seed(rng.randint(0, 2**31))
        slots = (self.pad_block + 1) * block_size
        # Every slot holds data, including never-written tails and the padding page (stale KV on the device).
        self.key_cache = torch.randn(slots, kv_heads, head, generator=gen, dtype=torch.float64).to(dtype)
        self.value_cache = torch.randn(slots, kv_heads, head, generator=gen, dtype=torch.float64).to(dtype)
        self.query = torch.randn(padded * width, q_heads * head, generator=gen, dtype=torch.float64).to(dtype)
        self.scale = head ** -0.5

    def live(self):
        return [(row, tok) for row in range(self.n) for tok in range(self.qlen[row])]

    def slots(self):
        """Main-group slot mapping exactly as _create_decode_input_data lays it out: [padded * width, 1]."""
        out = torch.empty((self.padded, self.width), dtype=torch.int64)
        dummy = itertools.cycle(range(self.pad_block * self.block_size, (self.pad_block + 1) * self.block_size))
        for row in range(self.padded):
            for tok in range(self.width):
                if row < self.n and tok < self.qlen[row]:
                    pos = self.ctx[row] + tok
                    out[row, tok] = int(self.table[row, pos // self.block_size]) * self.block_size + pos % self.block_size
                elif row < self.n:
                    out[row, tok] = next(dummy)
                else:
                    out[row, tok] = self.pad_block * self.block_size   # block_number PAD, offset 0
        return out.view(-1, 1)

    def flattened_tables(self):
        """build_block_tables_list of the deployed runner: one causal page prefix per flattened token."""
        tables = []
        for i in range(self.n):
            row = self.table[i, :self.pages[i]].tolist()
            for tok in range(self.qlen[i]):
                tables.append(row[:(self.ctx[i] + tok + self.block_size) // self.block_size])
            tables.extend([[self.pad_block]] * (self.width - self.qlen[i]))
        return tables

    def runner(self, buckets=None):
        return types.SimpleNamespace(attn_block_size=self.block_size, _PAD_BLOCK_ID=self.pad_block,
                                     use_contiguous_pa=False, bucketing_manager=buckets or Buckets(),
                                     model_config=types.SimpleNamespace(dtype=self.dtype), get_dp_padding=lambda s: 0)

    def flattened_output(self):
        r = self.runner()
        blocks, groups, usage = REF_BUFFERS(r, self.flattened_tables(), self.slots().tolist(),
                                            self.padded * self.width, block_size=self.block_size)
        # Rows beyond the live requests own no page in the reference either.
        meta = device_metadata(blocks, groups, usage, self.block_size, self.padded * self.width, self.dtype)
        return OPS.flat_pa(query=self.query.view(self.padded * self.width, 1, -1), block_list=meta.block_list,
                           block_mapping=meta.block_mapping, block_bias=meta.attn_bias,
                           block_groups=meta.block_groups,
                           **attention_args(self.key_cache, self.value_cache, self.block_size, self.scale))

    def shared_buffers(self, buckets=None):
        b = buckets or Buckets()
        return sp.shared_prefix_buffers(self.ctx, self.qlen, self.width, self.block_size, self.table, self.pad_block,
                                        lambda total: b.find_decode_bucket(self.padded, total, True)[2], self.dtype)

    def shared_output(self):
        blocks, groups, usage = self.shared_buffers()
        meta = device_metadata(blocks, groups, usage, self.block_size, self.padded * self.width, self.dtype)
        assert meta.block_mapping.shape == (blocks.numel(), self.padded)
        assert meta.attn_bias.shape == (blocks.numel(), self.width, self.block_size)
        out = OPS.flat_pa_query_block(query=self.query.view(self.padded, self.width, -1),
                                      block_list=meta.block_list, block_mapping=meta.block_mapping,
                                      block_bias=meta.attn_bias, block_groups=meta.block_groups,
                                      **attention_args(self.key_cache, self.value_cache, self.block_size,
                                                       self.scale))
        # HPUAttentionImpl.forward ends with output.view(num_tokens, hidden)
        out.view(self.padded * self.width, self.q_heads * self.head)
        return out


def random_case(rng, dtype=torch.float64, block_size=None, width=None):
    bs = block_size or rng.choice([4, 8, 16])
    width = width or rng.choice([8, 8, 4, 2])
    n = rng.randint(1, 2)
    padded = rng.choice([n, 2])
    ctx = [max(0, rng.choice([0, 1, bs - 1, bs, bs + 1, rng.randint(0, 12 * bs),
                              bs * rng.randint(1, 9) - rng.randint(0, width + 1)])) for _ in range(n)]
    qlen = [rng.choice([width, width, width, rng.randint(1, width)]) for _ in range(n)]
    kv, q = rng.choice([(2, 4), (2, 12), (3, 3), (1, 2)])
    return Case(rng, bs, width, n, padded, ctx, qlen, kv, q, rng.choice([4, 8]), dtype)


# ----------------------------------------------------------------------------- attention equivalence
@pytest.mark.parametrize("seed", range(400))
def test_attention_matches_flattened_layout(seed):
    rng = random.Random(seed)
    case = random_case(rng)
    ref = case.flattened_output()
    new = case.shared_output()
    assert new.shape == ref.shape == (case.padded * case.width, case.q_heads, case.head)
    assert torch.isfinite(new).all()
    for row, tok in case.live():
        i = row * case.width + tok
        assert torch.allclose(new[i], ref[i], rtol=1e-9, atol=1e-11), (seed, row, tok)


@pytest.mark.parametrize("seed", range(40))
def test_page_crossing_blocks_are_ignored(seed):
    """Tokens before a page crossing must not see the next page, whatever stale data it holds."""
    rng = random.Random(1000 + seed)
    bs, width = rng.choice([4, 8, 16]), 8
    ctx = [bs * rng.randint(1, 6) - rng.randint(1, width - 1)]      # the block crosses into a fresh page
    case = Case(rng, bs, width, 1, 1, ctx, [width], 2, 4, 8, torch.float64)
    blocks, groups, usage = case.shared_buffers()
    assert (usage[:case.pages[0]] == 0).any()
    ref, new = case.flattened_output(), case.shared_output()
    assert torch.allclose(new, ref, rtol=1e-9, atol=1e-11)
    # Poison everything a token is not allowed to read; the result may not move.
    last = int(case.table[0, case.pages[0] - 1]) * bs
    written = (ctx[0] + width) - (case.pages[0] - 1) * bs
    case.key_cache[last + written:last + bs] = 1e6
    case.value_cache[last + written:last + bs] = -1e6
    poisoned = case.shared_output()
    assert torch.allclose(poisoned, new, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("seed", range(60))
def test_attention_matches_naive_softmax(seed):
    """Independent check: plain causal softmax attention over each token's visible prefix."""
    rng = random.Random(2000 + seed)
    case = random_case(rng)
    new = case.shared_output()
    group = case.q_heads // case.kv_heads
    for row, tok in case.live():
        visible = case.ctx[row] + tok + 1
        idx = torch.tensor([int(case.table[row, p // case.block_size]) * case.block_size + p % case.block_size
                            for p in range(visible)])
        k, v = case.key_cache[idx], case.value_cache[idx]                       # [visible, kv, head]
        q = case.query[row * case.width + tok].view(case.q_heads, case.head) * case.scale
        for h in range(case.q_heads):
            w = torch.softmax(k[:, h // group] @ q[h], dim=0)
            assert torch.allclose(new[row * case.width + tok, h], w @ v[:, h // group], rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seed", range(30))
def test_attention_low_precision(seed, dtype):
    rng = random.Random(3000 + seed)
    case = random_case(rng, dtype=dtype)
    try:
        ref, new = case.flattened_output(), case.shared_output()
    except RuntimeError as exc:   # an op without a CPU kernel for this dtype
        pytest.skip(str(exc)[:80])
    assert torch.isfinite(new).all()
    tol = 2e-4 if dtype == torch.float32 else 0.12
    for row, tok in case.live():
        i = row * case.width + tok
        assert torch.allclose(new[i].float(), ref[i].float(), rtol=tol, atol=tol), (seed, row, tok)


def test_width_one_is_the_plain_layout():
    """A [pages, 1] usage tensor reproduces the plain decode result (same pages, same order)."""
    rng = random.Random(7)
    for _ in range(50):
        case = random_case(rng, width=1)
        assert torch.allclose(case.shared_output(), case.flattened_output(), rtol=1e-9, atol=1e-11)


def test_real_shape_smoke():
    """Deployment shape: 24 query heads over 4 KV heads, head 256, 128-token pages, 8 verification tokens."""
    rng = random.Random(11)
    case = Case(rng, 128, 8, 2, 2, [128 * 3 - 5, 700], [8, 3], 4, 24, 256, torch.float32)
    ref, new = case.flattened_output(), case.shared_output()
    for row, tok in case.live():
        i = row * 8 + tok
        assert torch.allclose(new[i], ref[i], rtol=2e-4, atol=2e-4)
    assert torch.isfinite(new).all()


def test_guards_reject_other_attention_variants():
    rng = random.Random(3)
    case = random_case(rng)
    blocks, groups, usage = case.shared_buffers()
    meta = device_metadata(blocks, groups, usage, case.block_size, case.padded * case.width, case.dtype)
    args = attention_args(case.key_cache, case.value_cache, case.block_size, case.scale)
    kwargs = dict(query=case.query.view(case.padded, case.width, -1), block_list=meta.block_list,
                  block_mapping=meta.block_mapping, block_bias=meta.attn_bias, block_groups=meta.block_groups)
    for flag in ("fp32_softmax", "fused_block_softmax", "fused_block_softmax_adjustment"):
        setattr(CONFIG, flag, True)
        try:
            with pytest.raises(NotImplementedError):
                OPS.flat_pa_query_block(**kwargs, **args)
        finally:
            setattr(CONFIG, flag, False)
    for flag in ("VLLM_HPU_GQA_COMPACT_KV", "VLLM_HPU_PAGED_KV_DUAL_GATHER", "VLLM_HPU_GQA_NATIVE_MATMUL"):
        setattr(OPS_ENV, flag, True)
        try:
            with pytest.raises(NotImplementedError):
                OPS.flat_pa_query_block(**kwargs, **args)
        finally:
            setattr(OPS_ENV, flag, False)
    with pytest.raises(NotImplementedError):
        OPS.flat_pa_query_block(**kwargs, **{**args, "sinks": torch.zeros(case.q_heads)})


# ----------------------------------------------------------------------------- CPU builders
def naive_buffers(case, bucket):
    blocks, groups, usage = [], [], []
    for row in range(case.n):
        for page in range(case.pages[row]):
            blocks.append(int(case.table[row, page])); groups.append(row)
            line = []
            for tok in range(case.width):
                t = min(tok, case.qlen[row] - 1)
                line.append(min(case.block_size, max(0, case.ctx[row] + t + 1 - page * case.block_size)))
            usage.append(line)
    target = max(bucket(len(blocks)), len(blocks))
    while len(blocks) < target:
        blocks.append(case.pad_block); groups.append(-1); usage.append([1] * case.width)
    return (torch.tensor(blocks, dtype=torch.long), torch.tensor(groups, dtype=torch.long),
            torch.tensor(usage, dtype=torch.long).to(case.dtype))


@pytest.mark.parametrize("seed", range(300))
def test_builder_matches_naive(seed):
    rng = random.Random(4000 + seed)
    case = random_case(rng, dtype=torch.bfloat16, block_size=rng.choice([4, 16, 128]))
    buckets = Buckets(step=rng.choice([8, 128]))
    got = case.shared_buffers(buckets)
    want = naive_buffers(case, lambda total: buckets.find_decode_bucket(case.padded, total, True)[2])
    for a, b in zip(got, want):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    blocks, groups, usage = got
    assert usage.shape[1] == case.width and blocks.numel() >= sum(case.pages)
    # Each live token sees exactly its causal prefix.
    for row, tok in case.live():
        seen = int(usage[:sum(case.pages)].long()[groups[:sum(case.pages)] == row][:, tok].sum())
        assert seen == case.ctx[row] + tok + 1


def test_builder_pads_past_the_largest_bucket():
    rng = random.Random(5)
    case = Case(rng, 4, 8, 1, 1, [4 * 300], [8], 2, 4, 4, torch.bfloat16)
    small = Buckets(step=128, limit=256)        # the bucket table stops below the request
    blocks, groups, usage = sp.shared_prefix_buffers(case.ctx, case.qlen, 8, 4, case.table, case.pad_block,
                                                     lambda total: small.find_decode_bucket(1, total, True)[2],
                                                     case.dtype)
    assert blocks.numel() == sum(case.pages)    # never shorter than the real page list


def test_builder_rejects_bad_inputs():
    rng = random.Random(6)
    case = random_case(rng, dtype=torch.bfloat16)
    bucket = lambda total: total
    args = (case.width, case.block_size, case.table, case.pad_block, bucket, case.dtype)
    assert sp.shared_prefix_buffers(case.ctx, case.qlen, *args) is not None
    assert sp.shared_prefix_buffers([], [], *args) is None
    assert sp.shared_prefix_buffers(case.ctx, [0] * case.n, *args) is None
    assert sp.shared_prefix_buffers(case.ctx, [case.width + 1] * case.n, *args) is None
    assert sp.shared_prefix_buffers([-1] * case.n, case.qlen, *args) is None
    short = case.table[:, :max(case.pages) - 1]
    assert sp.shared_prefix_buffers(case.ctx, case.qlen, case.width, case.block_size, short, case.pad_block, bucket,
                                    case.dtype) is None
    bad = case.table.copy(); bad[0, 0] = case.pad_block
    assert sp.shared_prefix_buffers(case.ctx, case.qlen, case.width, case.block_size, bad, case.pad_block, bucket,
                                    case.dtype) is None
    bad[0, 0] = -1
    assert sp.shared_prefix_buffers(case.ctx, case.qlen, case.width, case.block_size, bad, case.pad_block, bucket,
                                    case.dtype) is None


@pytest.mark.parametrize("seed", range(300))
def test_group_slots_match_decode_group_addresses(seed):
    rng = random.Random(5000 + seed)
    case = random_case(rng, dtype=torch.bfloat16, block_size=rng.choice([4, 16, 128]))
    tables = [case.table[i, :case.pages[i]].tolist() for i in range(case.n)]
    want, _ = agr.decode_group_addresses(tables, case.ctx, case.qlen, case.padded, case.width, case.block_size,
                                         case.pad_block)
    got = sp.group_slots(case.ctx, case.qlen, case.padded, case.width, case.block_size, case.table, case.pad_block)
    assert got.dtype == want.dtype and got.shape == want.shape and torch.equal(got, want)


# ----------------------------------------------------------------------------- patched runner
class AttentionSpec:
    pass


class FakeTable:
    def __init__(self, tensor):
        self.tensor = tensor

    def get_cpu_tensor(self):
        return self.tensor


RUNNER_NAMES = ["_route_legacy_decode_groups", "_legacy_attention_tables", "_resolve_block",
                "_get_attention_group_id_for_hybrid", "_shared_prefix_decode_enabled", "_shared_prefix_page_table",
                "_shared_prefix_buffers", "_fast_prep_state_for", "_fast_prep_layout_ok", "_fast_prep",
                "get_habana_paged_attn_buffers"]


def make_runner(rng, n, width, flag, block_size=128, pad_block=5000, attn_groups=(10, 11, 12, 13), num_groups=15,
                fake_hpu=False):
    env = types.SimpleNamespace(VLLM_HPU_SHARED_PREFIX_DECODE=flag, VLLM_HPU_FAST_DECODE_PREP="",
                                VLLM_HPU_FAST_DECODE_PREP_VERIFY=0)
    funcs, logs, ns = class_methods(RUN_SRC, RUNNER_NAMES, {"gaudi_envs": env, "AttentionSpec": AttentionSpec,
                                                             "is_fake_hpu": lambda: fake_hpu})
    r = types.SimpleNamespace()
    for k, f in funcs.items():
        setattr(r, k, types.MethodType(f, r))
    r.attn_block_size, r._PAD_BLOCK_ID = block_size, pad_block
    r.use_contiguous_pa = r.interleaved_sliding_window = r.model_has_chunked_attention = False
    r.use_framework_kv_cache_layout = r.warmup_mode = False
    r.bucketing_manager = Buckets()
    r.model_config = types.SimpleNamespace(dtype=torch.bfloat16)
    r.get_dp_padding = lambda size: 0
    r.device = "cpu"
    r.num_mamba_like_layers = 48
    r._legacy_attention_group_ids = tuple(attn_groups)
    r.kv_cache_config = types.SimpleNamespace(kv_cache_groups=[
        types.SimpleNamespace(kv_cache_spec=AttentionSpec() if g in attn_groups else object())
        for g in range(num_groups)])
    ctx = [rng.choice([0, block_size - 3, block_size, rng.randint(0, 3000)]) for _ in range(n)]
    qlen = [rng.choice([width, width, rng.randint(1, width)]) for _ in range(n)]
    pages = [(c + q + block_size - 1) // block_size for c, q in zip(ctx, qlen)]
    tables = []
    for g in range(num_groups):
        t = torch.zeros((4, max(pages) + 4), dtype=torch.int32)
        pool = rng.sample(range(pad_block), sum(pages)); k = 0
        for i, p in enumerate(pages):
            t[i, :p] = torch.tensor(pool[k:k + p], dtype=torch.int32); k += p
        tables.append(FakeTable(t))
    req_ids = [f"q{i}" for i in range(n)]
    r.input_batch = types.SimpleNamespace(block_table=tables, req_ids=req_ids,
                                          req_id_to_index={q: i for i, q in enumerate(req_ids)})
    return r, ctx, qlen, pages, logs


@pytest.mark.parametrize("seed", range(120))
def test_runner_routes_groups_with_shared_prefix(seed):
    rng = random.Random(6000 + seed)
    n, width = rng.randint(1, 2), rng.choice([8, 8, 4])
    padded = rng.choice([n, 2])
    r, ctx, qlen, pages, logs = make_runner(rng, n, width, True)
    base = Meta(slot_mapping=torch.zeros(1), block_list=torch.zeros(1))
    out = r._route_legacy_decode_groups(base, n, qlen + [0] * (padded - n), np.asarray(ctx), width, padded)
    assert any("Shared-prefix decode attention enabled" in a[0] for kind, a in logs if kind == "info")
    assert out.attention_groups[10].block_list is base.block_list          # first group keeps the main metadata
    for gid in (11, 12, 13):
        g = out.attention_groups[gid]
        table = r.input_batch.block_table[gid].get_cpu_tensor().numpy()[:n, :max(pages)]
        want = sp.shared_prefix_buffers(ctx, qlen, width, 128, table, r._PAD_BLOCK_ID,
                                        lambda total: r.bucketing_manager.find_decode_bucket(padded, total, True)[2],
                                        torch.bfloat16)
        for a, b in zip((g.block_list, g.block_groups, g.block_usage), want):
            assert a.dtype == b.dtype and torch.equal(a, b)
        assert g.block_usage.shape == (g.block_list.numel(), width)
        rows = [r.input_batch.block_table[gid].get_cpu_tensor()[i, :pages[i]].tolist() for i in range(n)]
        slots, _ = agr.decode_group_addresses(rows, ctx, qlen, padded, width, 128, r._PAD_BLOCK_ID)
        assert g.slot_mapping.dtype == slots.dtype and torch.equal(g.slot_mapping, slots)


@pytest.mark.parametrize("seed", range(40))
def test_runner_keeps_flattened_layout_when_off_or_single_token(seed):
    rng = random.Random(7000 + seed)
    for flag, width in ((False, 8), (True, 1)):
        n = rng.randint(1, 2)
        state = rng.getstate()
        r, ctx, qlen, pages, _ = make_runner(rng, n, width, flag)
        rng.setstate(state)
        r0, *_ = make_runner(rng, n, width, False)
        base = Meta(slot_mapping=torch.zeros(1), block_list=torch.zeros(1))
        out = r._route_legacy_decode_groups(base, n, list(qlen), np.asarray(ctx), width, n)
        ref = r0._route_legacy_decode_groups(base, n, list(qlen), np.asarray(ctx), width, n)
        for gid in (11, 12, 13):
            for field in ("slot_mapping", "block_list", "block_groups", "block_usage"):
                a, b = getattr(out.attention_groups[gid], field), getattr(ref.attention_groups[gid], field)
                assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
            assert out.attention_groups[gid].block_usage.dim() == 1


def test_runner_ignores_flag_on_unsupported_layouts():
    rng = random.Random(8)
    for attr in ("use_contiguous_pa", "interleaved_sliding_window", "model_has_chunked_attention",
                 "use_framework_kv_cache_layout"):
        r, ctx, qlen, pages, logs = make_runner(rng, 1, 8, True)
        setattr(r, attr, True)
        assert r._shared_prefix_decode_enabled() is False
        assert any(kind == "warning" for kind, _ in logs)
    r, *_ = make_runner(rng, 1, 8, True, fake_hpu=True)
    assert r._shared_prefix_decode_enabled() is False
    r, *_ = make_runner(rng, 1, 8, False)
    assert r._shared_prefix_decode_enabled() is False


def test_runner_rounds_up_past_the_largest_bucket():
    rng = random.Random(9)
    r, ctx, qlen, pages, _ = make_runner(rng, 1, 8, True)
    r.bucketing_manager = Buckets(step=128, limit=512)
    table = np.arange(700, dtype=np.int32).reshape(1, -1)
    blocks, groups, usage = r._shared_prefix_buffers(table, [128 * 600], [8], 8, 1)
    assert blocks.numel() == 768 and int((groups >= 0).sum()) == 601
    blocks2, *_ = r._shared_prefix_buffers(table, [128 * 640], [8], 8, 1)
    assert blocks2.numel() == 768                       # same shape for neighbouring rounds
    with pytest.raises(RuntimeError):
        r._shared_prefix_buffers(table, [128 * 800], [8], 8, 1)


def test_set_block_mapping_plain_decode():
    """One-dimensional usage (plain decode, flattened layout) keeps its [pages, block_size] mask and mapping."""
    usage = torch.tensor([128, 128, 5, 1, 1], dtype=torch.bfloat16)
    groups = torch.tensor([0, 0, 0, 1, -1])
    meta = Meta(block_list=torch.arange(5), block_groups=groups.clone(), block_usage=usage)
    out = SET_MAPPING(processor(128), meta, 2, torch.device("cpu"), torch.bfloat16)
    assert out.block_mapping.shape == (5, 2) and out.attn_bias.shape == (5, 128)
    assert torch.isinf(out.attn_bias[2, 5:]).all() and (out.attn_bias[2, :5] == 0).all()
    assert out.block_mapping[:3, 0].eq(1).all() and out.block_mapping[3, 1] == 1 and out.block_mapping[4].eq(0).all()


def test_builder_speed_report(capsys):
    rng = random.Random(10)
    lines = []
    for ctx_tokens in (600, 30_000, 91_000, 262_000):
        for n in (1, 2):
            ctx = [ctx_tokens // n - 9] * n
            pages = [(c + 8 + 127) // 128 for c in ctx]
            table = np.arange(sum(pages) + 8, dtype=np.int32)[:n * (max(pages) + 1)].reshape(n, -1)
            t0 = time.perf_counter()
            for _ in range(50):
                for _group in range(4):
                    sp.shared_prefix_buffers(ctx, [8] * n, 8, 128, table, 10**6, lambda total: -(-total // 128) * 128,
                                             torch.bfloat16)
                for _group in range(3):
                    sp.group_slots(ctx, [8] * n, n, 8, 128, table, 10**6)
            lines.append(f"ctx {ctx_tokens:>7} B={n}: {(time.perf_counter() - t0) / 50 * 1e3:6.3f} ms for 4 groups")
    with capsys.disabled():
        print("\n" + "\n".join(lines))
