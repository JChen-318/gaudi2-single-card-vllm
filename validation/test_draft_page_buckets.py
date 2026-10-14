"""Draft context page buckets keep the same attention: extra pages are padding and fully masked."""
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import types  # noqa: E402
import v6_support as v6_local_test  # noqa: E402

for name in ("vllm_gaudi", "vllm_gaudi.v1", "vllm_gaudi.v1.spec_decode"):
    sys.modules.setdefault(name, types.ModuleType(name))
utils = types.ModuleType("vllm_gaudi.utils")
utils.async_h2d_copy = lambda source, device=None, **_: source
sys.modules["vllm_gaudi.utils"] = utils

host = v6_local_test._load("v1/spec_decode/dflash_host_prepare.py", "vllm_gaudi.v1.spec_decode.dflash_host_prepare")
cpu = v6_local_test._load("v1/spec_decode/dflash_cpu_inputs.py", "dflash_cpu_inputs_v10_test")


def test_bucket_sizes():
    assert [cpu.bucket_context_blocks(n) for n in (0, 1, 2, 3, 5, 17, 32, 33, 64, 65, 2047)] == \
        [0, 1, 2, 4, 8, 32, 32, 64, 64, 96, 2048]


@pytest.mark.parametrize("starts", [[1], [127, 128], [129, 4000], [9000, 300]])
@pytest.mark.parametrize("window", [None, 2048])
def test_bucketed_inputs_equal_on_live_pages(starts, window):
    table = torch.arange(2 * 200, dtype=torch.int32).reshape(2, 200) + 7
    kw = dict(batch_size=len(starts), query_len=8, block_size=128, mask_token_id=3, pad_block=99999, pad_slot=-5,
              window=window, causal=False, dtype=torch.float32)
    rows = list(range(len(starts)))
    ref = cpu.prepare_query_cpu_inputs(table, rows, starts, [11] * len(starts), **kw)
    new = cpu.prepare_query_cpu_inputs(table, rows, starts, [11] * len(starts), bucket_blocks=cpu.bucket_context_blocks,
                                       **kw)
    for key in ("input_ids", "positions", "slot_mapping", "context_lens_tensor", "seq_lens_tensor"):
        assert torch.equal(ref[key], new[key]), key
    b = len(starts)
    old_pages = ref["block_list"].numel() // b
    new_pages = new["block_list"].numel() // b
    assert new_pages == cpu.bucket_context_blocks(old_pages)
    assert torch.equal(new["block_list"].reshape(b, -1)[:, :old_pages], ref["block_list"].reshape(b, -1))
    assert bool((new["block_list"].reshape(b, -1)[:, old_pages:] == 99999).all())
    past_old, past_new = old_pages * 128, new_pages * 128
    assert torch.equal(new["attn_bias"][..., :past_old], ref["attn_bias"][..., :past_old])
    assert bool(torch.isinf(new["attn_bias"][..., past_old:past_new]).all())
    assert torch.equal(new["attn_bias"][..., past_new:], ref["attn_bias"][..., past_old:])
