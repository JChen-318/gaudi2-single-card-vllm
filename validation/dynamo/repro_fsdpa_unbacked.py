"""Dynamo repro of the drafter attention graph break (FusedSDPA.py:105 after the block list is marked unbacked).

The functions mirror the deployed call chain: ops._include_past (context KV fetched by block_list, reshaped with
-1 and concatenated with the current KV) followed by Habana's sdpa_fwd_wrapper, whose first statement tests
`k.numel() == 0`. Usage: python repro_fsdpa.py <mark|nomark|check>
"""
import sys, warnings
warnings.filterwarnings("ignore")
import torch
import torch._dynamo as dynamo

MODE = sys.argv[1] if len(sys.argv) > 1 else "mark"
BLOCK, KV, HEAD, QH = 128, 2, 16, 4


def include_past(current, cache, block_list):
    past = cache.unflatten(0, (-1, BLOCK)).index_select(0, block_list)
    past = past.reshape(current.size(0), -1, past.shape[2], past.shape[3])
    return torch.concat((past, current), dim=1)


def sdpa_fwd_wrapper(q, k, v, bias):
    # first statement of habana_frameworks/torch/hpex/kernels/FusedSDPA.py::sdpa_fwd_wrapper
    if q.numel() == 0 or k.numel() == 0 or v.numel() == 0:
        out_shape = list(q.shape)
        out_shape[-1] = v.shape[-1]
        return q.new_empty(out_shape)
    w = torch.matmul(q, k.transpose(-1, -2)) + bias
    return torch.matmul(torch.softmax(w, dim=-1), v)


def layer(query, key, value, key_cache, value_cache, block_list, bias, seq_lens):
    # hpu_attn.HPUAttentionImpl.forward, prompt branch: flat tokens, batch taken from seq_lens_tensor
    batch_size = seq_lens.shape[0]
    num_tokens, hidden = query.shape
    seq_len = num_tokens // batch_size
    query = query.view(batch_size, seq_len, -1)
    key = key.view(-1, KV, HEAD)
    value = value.view(-1, KV, HEAD)
    q = query.view(batch_size, seq_len, QH, HEAD)
    k = include_past(key.view(batch_size, -1, KV, HEAD), key_cache, block_list)
    v = include_past(value.view(batch_size, -1, KV, HEAD), value_cache, block_list)
    if MODE == "check":
        torch._check(k.shape[1] > 0)
        torch._check(v.shape[1] > 0)
    q = q.transpose(1, 2).unflatten(1, (KV, -1))
    k = k.transpose(1, 2).unsqueeze(2)
    v = v.transpose(1, 2).unsqueeze(2)
    out = sdpa_fwd_wrapper(q, k, v, bias).flatten(1, 2).transpose(1, 2)
    return out.reshape(batch_size, seq_len, hidden)


compiled = torch.compile(layer, backend="eager")


def run(batch, pages_per_row, width=8):
    cache_k = torch.randn(64 * BLOCK, KV, HEAD); cache_v = torch.randn(64 * BLOCK, KV, HEAD)
    block_list = torch.arange(batch * pages_per_row, dtype=torch.int32)
    if MODE != "nomark":
        dynamo.decorators.mark_unbacked(block_list, 0)
    q = torch.randn(batch * width, QH * HEAD); k = torch.randn(batch * width, KV * HEAD)
    v = torch.randn(batch * width, KV * HEAD)
    bias = torch.zeros(batch, 1, 1, width, pages_per_row * BLOCK + width)
    seq_lens = torch.full((batch,), width)
    out = compiled(q, k, v, cache_k, cache_v, block_list, bias, seq_lens)
    ref = layer(q, k, v, cache_k, cache_v, block_list, bias, seq_lens)
    assert torch.allclose(out, ref, atol=1e-5)


dynamo.reset()
dynamo.utils.counters.clear()
shapes = [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (1, 8), (2, 4)]
for i, (b, p) in enumerate(shapes):
    run(b, p)
    c = dynamo.utils.counters
    print(f"{MODE:7} after shape B={b} pages/row={p}: frames ok {c['frames']['ok']:3d}  "
          f"graph breaks {sum(c['graph_break'].values()):2d}  unique graphs {c['stats']['unique_graphs']:3d}")
for reason, n in list(dynamo.utils.counters["graph_break"].items())[:3]:
    print("   break:", str(reason).strip().split("\n")[0][:170], "x", n)
