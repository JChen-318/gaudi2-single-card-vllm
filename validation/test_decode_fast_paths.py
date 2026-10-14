"""CPU equivalence tests: vectorised decode-prep builders (VLLM_HPU_FAST_DECODE_PREP) vs the reference code.

The reference methods are AST-extracted from the model runner in this repository and bound to a small fake
object, so the comparison is against the shipped list-based code, not a re-implementation.
Run from the repository root: python -m pytest -q validation
"""
import ast, importlib.util, itertools, os, random, sys, textwrap, types

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


RUNNER = os.path.join(WORKER, "hpu_model_runner.py")
fp = load_module("decode_fast_paths", os.path.join(WORKER, "decode_fast_paths.py"))
agr = load_module("attention_group_routing_ref", os.path.join(WORKER, "attention_group_routing.py"))

SRC = open(RUNNER, encoding="utf-8").read()
TREE = ast.parse(SRC)


def flatten(in_list):
    return list(itertools.chain(*in_list))


def gather_list(input, indices, v):
    return [input[i] if i is not None else v for i in indices]


def pad_list(input, target_len, val_generator):
    padding = target_len - len(input)
    if padding > 0:
        input.extend(itertools.islice(val_generator, padding))
    return input


def method(name):
    """Compile one HPUModelRunner method from the live source into a plain function."""
    for node in ast.walk(TREE):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    code = textwrap.dedent(ast.get_source_segment(SRC, item))
                    ns = {"torch": torch, "np": np, "itertools": itertools, "flatten": flatten, "pad_list": pad_list,
                          "gather_list": gather_list, "Any": object, "Optional": object, "Union": object,
                          "logger": types.SimpleNamespace(warning=lambda *a, **k: None)}
                    exec(code, ns)
                    return ns[name]
    raise KeyError(name)


REF_BUFFERS = method("get_habana_paged_attn_buffers")
REF_GDN = method("_prepare_mamba_state_idxs_reference")


class Buckets:
    """Linear decode buckets like the deployment: blocks 1,2,4,..,128,256,384,512 then multiples of 128."""

    def find_decode_bucket(self, batch_size, num_blocks, seed_buckets=False):
        ramp = [1, 2, 4, 8, 16, 32, 64]
        size = next((b for b in ramp if b >= num_blocks), None) or -(-num_blocks // 128) * 128
        return (batch_size, 1, size)


class Runner:
    def __init__(self, block_size=128, pad_block=4000, dtype=torch.bfloat16):
        self.attn_block_size = block_size
        self._PAD_BLOCK_ID = pad_block
        self.use_contiguous_pa = False
        self.bucketing_manager = Buckets()
        self.model_config = types.SimpleNamespace(dtype=dtype)

    def get_dp_padding(self, n):
        return 0


def same(a, b):
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert torch.equal(a, b)


def bucket_fn(runner, batch_size):
    return lambda total: runner.bucketing_manager.find_decode_bucket(batch_size, total)[2]


def random_case(rng, max_ctx):
    bs = rng.choice([16, 128])
    width = rng.choice([1, 2, 8])
    n = rng.randint(1, 3)
    padded = n + rng.randint(0, 2)
    ctx = [rng.choice([0, 1, bs - 1, bs, bs + 1, rng.randint(0, max_ctx), bs * rng.randint(1, 20) - rng.randint(0, 9)])
           for _ in range(n)]
    ctx = [max(0, c) for c in ctx]
    qlen = [rng.choice([width, width, width, rng.randint(1, width)]) for _ in range(n)]
    pages = [(c + q + bs - 1) // bs for c, q in zip(ctx, qlen)]
    pad_block = 5000
    pool = rng.sample(range(pad_block), sum(pages))
    tables, k = [], 0
    for p in pages:
        tables.append(pool[k:k + p]); k += p
    return bs, width, n, padded, ctx, qlen, pages, pad_block, tables


def table_array(tables, extra_cols=3, fill=-7):
    cols = max(len(t) for t in tables) + extra_cols
    arr = np.full((len(tables), cols), fill, dtype=np.int32)
    for i, t in enumerate(tables):
        arr[i, :len(t)] = t
    return arr


@pytest.mark.parametrize("seed", range(300))
def test_group_buffers_match_reference(seed):
    rng = random.Random(seed)
    bs, width, n, padded, ctx, qlen, pages, pad_block, tables = random_case(rng, max_ctx=6000)
    runner = Runner(bs, pad_block)
    slots, prefixes = agr.decode_group_addresses(tables, ctx, qlen, padded, width, bs, pad_block)
    blocks, owners, usage = REF_BUFFERS(runner, prefixes, slots.tolist(), padded * width, block_size=bs)
    arr = table_array(tables)[:, :max(pages)]
    out = group(arr, ctx, qlen, padded, width, bs, pad_block, bucket_fn(runner, padded * width))
    assert out is not None
    for a, b in zip(out, (slots, blocks, owners, usage)):
        same(a, b)


@pytest.mark.parametrize("seed", range(300))
def test_main_buffers_match_reference(seed):
    """The reference page lists come from build_block_tables_list, taken verbatim from _create_decode_input_data."""
    rng = random.Random(1000 + seed)
    bs, width, n, padded, ctx, qlen, pages, pad_block, tables = random_case(rng, max_ctx=6000)
    runner = Runner(bs, pad_block)
    block_table_cpu_tensor = torch.from_numpy(table_array(tables, extra_cols=5, fill=0))
    num_blocks, num_tokens, context_lens = pages, width, np.asarray(ctx)
    num_tokens_per_req = qlen + [0] * (padded - n)
    decode_block_size = bs
    loop = SRC[SRC.index("        def build_block_tables_list():"):
               SRC.index("        # The list form is only needed by the reference builder")]
    ns = dict(num_blocks=num_blocks, block_table_cpu_tensor=block_table_cpu_tensor, num_tokens_per_req=num_tokens_per_req,
              context_lens=context_lens, decode_block_size=bs, num_tokens=width, self=runner)
    exec(textwrap.dedent(loop), ns)
    block_tables_list = ns["build_block_tables_list"]()
    # slot mapping as the runner would produce it: live tokens real slots, dead tokens arbitrary pad slots
    slot = torch.full((padded, width), pad_block * bs, dtype=torch.int64)
    dummy = itertools.cycle(range(pad_block * bs, pad_block * bs + bs))
    for i in range(n):
        for t in range(width):
            pos = ctx[i] + t
            slot[i, t] = tables[i][pos // bs] * bs + pos % bs if t < qlen[i] else next(dummy)
    slot_mapping = slot.view(-1, 1) if width > 1 else slot
    ref = REF_BUFFERS(runner, block_tables_list, slot_mapping.tolist(), padded * width, block_size=bs)
    layout = fp.token_layout(ctx, qlen, n, width, bs)
    table = block_table_cpu_tensor.numpy()[:n, :max(num_blocks)]
    slots = slot_mapping.numpy().reshape(-1)[:n * width].astype(np.int64)
    out = fp.causal_prefix_buffers(layout, table, slots, pad_block, bucket_fn(runner, padded * width), torch.bfloat16)
    assert out is not None
    for a, b in zip(out, ref):
        same(a, b)


def group(table, ctx, qlen, padded, width, bs, pad_block, bucket, dtype=torch.bfloat16):
    layout = fp.token_layout(ctx, qlen, padded, width, bs)
    return None if layout is None else fp.decode_group_buffers(layout, table, pad_block, bucket, dtype)


def test_group_buffers_reject_what_reference_rejects():
    runner = Runner()
    bucket = bucket_fn(runner, 8)
    ok = np.array([[5, 6, 7]], dtype=np.int32)
    assert group(ok, [10], [8], 1, 8, 128, 4000, bucket) is not None
    for bad in (np.array([[4000, 6, 7]], dtype=np.int32), np.array([[-1, 6, 7]], dtype=np.int32)):
        assert group(bad, [10], [8], 1, 8, 128, 4000, bucket) is None
        with pytest.raises(IndexError):
            agr.decode_group_addresses([bad[0].tolist()], [10], [8], 1, 8, 128, 4000)
    # prefix needs page 1 but the table only has one column
    assert group(ok[:, :1], [125], [8], 1, 8, 128, 4000, bucket) is None
    # query longer than the verification width / negative context
    assert group(ok, [10], [9], 1, 8, 128, 4000, bucket) is None
    assert group(ok, [-1], [8], 1, 8, 128, 4000, bucket) is None
    # pages beyond the used prefix may be garbage without affecting the result
    garbage = np.array([[5, -9, 99999]], dtype=np.int32)
    a = group(garbage, [10], [8], 1, 8, 128, 4000, bucket)
    b = group(ok, [10], [8], 1, 8, 128, 4000, bucket)
    for x, y in zip(a, b):
        same(x, y)


class FakeTable:
    def __init__(self, t):
        self.t = t

    def get_cpu_tensor(self):
        return self.t


class FakeBlockTable:
    def __init__(self, tables):
        self.block_tables = tables

    def __getitem__(self, i):
        return self.block_tables[i]


def gdn_runner(rng, num_groups, compact, num_reqs, max_reqs, slots_per_req):
    r = types.SimpleNamespace()
    tables = [FakeTable(torch.randint(0, 900, (max_reqs, 6), dtype=torch.int32, generator=torch.Generator().manual_seed(
        rng.randint(0, 10**6)))) for _ in range(num_groups)]
    req_ids = [f"r{i}" for i in range(max_reqs)]
    rng.shuffle(req_ids)
    r.input_batch = types.SimpleNamespace(block_table=FakeBlockTable(tables), req_ids=req_ids)
    r._compact_gdn_group_ids = set(compact)
    r._compact_gdn_group_offset = {g: i for i, g in enumerate(sorted(compact))}
    r._gdn_max_reqs = max_reqs
    r._gdn_state_slots_per_req = slots_per_req
    slots = list(range(max_reqs)); rng.shuffle(slots)
    r._gdn_req_to_base_slot = dict(zip(req_ids, slots))
    r._MAMBA_PAD_BLOCK_ID = 12345
    return r


def fast_gdn(r, req_indices, offsets, target_bs, width, ckpt):
    base = [r._gdn_req_to_base_slot[r.input_batch.req_ids[i]] for i in req_indices] if r._compact_gdn_group_ids else []
    return fp.gdn_state_indices(len(r.input_batch.block_table.block_tables),
                                lambda g: r.input_batch.block_table[g].get_cpu_tensor(),
                                {g: r._compact_gdn_group_offset[g] for g in r._compact_gdn_group_ids}, base,
                                r._gdn_max_reqs, r._gdn_state_slots_per_req, req_indices, offsets, target_bs, width, ckpt,
                                r._MAMBA_PAD_BLOCK_ID)


@pytest.mark.parametrize("seed", range(300))
def test_gdn_state_indices_match_reference(seed):
    rng = random.Random(2000 + seed)
    num_groups = rng.choice([1, 5, 15])
    compact = [g for g in range(num_groups) if rng.random() < rng.choice([0.0, 0.7, 1.0])]
    max_reqs = rng.choice([2, 4])
    spr = rng.choice([1, 8])
    r = gdn_runner(rng, num_groups, compact, None, max_reqs, spr)
    n = rng.randint(1, max_reqs)
    req_indices = list(range(n))
    target_bs = n + rng.randint(0, 2)
    if rng.random() < 0.5:
        width, ckpt = rng.choice([1, 8]), None
    else:
        width, ckpt = 1, [rng.randint(-2, spr + 2) for _ in range(n)]
    offsets = rng.choice([[0] * n, [rng.randint(0, 5) for _ in range(n)],
                          torch.tensor([rng.randint(0, 5) for _ in range(n)]),
                          np.asarray([rng.randint(0, 5) for _ in range(n)])])
    ref = REF_GDN(r, req_indices, offsets, target_bs, width, ckpt)
    out = fast_gdn(r, req_indices, offsets, target_bs, width, ckpt)
    assert out is not None
    same(out, ref)


def test_gdn_state_indices_fall_back():
    rng = random.Random(7)
    r = gdn_runner(rng, 3, [0, 2], None, 4, 8)
    assert fast_gdn(r, [0, 1, 2], [0, 0, 0], 2, 8, None) is None          # more requests than the bucket
    assert fast_gdn(r, [0, 1], [0, 0], 2, 8, [0, 0]) is None              # checkpoint offsets need width 1
    r.input_batch.block_table.block_tables[1] = FakeTable(torch.zeros((4, 6), dtype=torch.int64))
    assert fast_gdn(r, [0, 1], [0, 0], 2, 1, None) is None                # unexpected table dtype
