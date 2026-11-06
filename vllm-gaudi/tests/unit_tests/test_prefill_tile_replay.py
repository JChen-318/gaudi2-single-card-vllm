import torch

from vllm_gaudi.extension import prefill_tile_replay as tiles


def _sdpa(q, k, v, bias, scale, _softmax_mode):
    if k.shape[1] != q.shape[1]:
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    scores = scores + bias.float()
    row_max = scores.amax(dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - row_max)
    # sdpa_recomp_fwd's third value is the inverse sum of exp, not the sum.
    sumexp = exp_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    out = torch.matmul(exp_scores / sumexp, v.float())
    return out.to(q.dtype), row_max, 1.0 / sumexp


def _reference(query, key_ctx, value_ctx, key_q, value_q, real_ctx, real_q, scale):
    parts_k = []
    parts_v = []
    if real_ctx > 0:
        parts_k.append(key_ctx[:, :, :real_ctx, :])
        parts_v.append(value_ctx[:, :, :real_ctx, :])
    parts_k.append(key_q[:, :, :real_q, :])
    parts_v.append(value_q[:, :, :real_q, :])
    key = torch.cat(parts_k, dim=-2)
    value = torch.cat(parts_v, dim=-2)
    q_len = query.shape[-2]
    k_len = key.shape[-2]
    cols = torch.arange(k_len)
    ctx_cols = cols < real_ctx
    query_cols = (cols >= real_ctx) & ((cols - real_ctx) < real_q)
    rows = torch.arange(q_len).view(q_len, 1)
    causal = query_cols.view(1, k_len) & ((cols - real_ctx).view(1, k_len) > rows)
    visible = ctx_cols.view(1, k_len) | (query_cols.view(1, k_len) & ~causal)
    bias = torch.zeros(1, 1, q_len, k_len)
    bias = bias.masked_fill(~visible.reshape(1, 1, q_len, k_len), float("-inf"))
    out, _, _ = _sdpa(query, key, value, bias, scale, "fp32")
    return out


def test_single_bucket_matches_256k_chunk():
    assert tiles.single_prompt_bucket(1, 2048, 128, 262144) == (1, 2048, 2032)


def test_seven_bundles_cover_256k():
    assert tiles.bundle_count(6) == 6
    assert tiles.bundle_count(7) == 8
    assert tiles.bundle_count(9) == 12
    assert tiles.bundle_count(17) == 24
    assert tiles.bundle_count(32) == 32
    prompt = tiles.prompt_bundle_buckets(1, 2048, 128, 262144)
    decode = tiles.decode_bundle_buckets(128, 262144)
    assert prompt == [
        (1, 2048, 64),
        (1, 2048, 128),
        (1, 2048, 192),
        (1, 2048, 256),
        (1, 2048, 320),
        (1, 2048, 384),
        (1, 2048, 512),
        (1, 2048, 768),
        (1, 2048, 1024),
        (1, 2048, 1536),
        (1, 2048, 2032),
    ]
    assert decode == [
        (1, 1, 64),
        (1, 1, 128),
        (1, 1, 256),
        (1, 1, 512),
        (1, 1, 1024),
        (1, 1, 1536),
        (1, 1, 2048),
    ]
    assert all(bucket[0] == 1 for bucket in decode)


def test_last_context_tile_pads_to_a_full_window():
    # 256K dummy: 2032 blocks, tile 64. Tile 31 starts at 1984 and has 48 real blocks.
    ids = torch.arange(2032)
    full = tiles.pad_block_window(ids, 0, 64)
    assert torch.equal(full, torch.arange(64))
    tail = tiles.pad_block_window(ids, 1984, 64)
    assert tail.shape == (64,)
    assert torch.equal(tail[:48], torch.arange(1984, 2032))
    assert torch.equal(tail[48:], torch.zeros(16, dtype=ids.dtype))
    assert torch.equal(tiles.pad_block_window(ids, 2032, 64), torch.zeros(64, dtype=ids.dtype))


def test_context_tile_count_follows_real_length():
    assert tiles.context_tile_count(0, 8192) == 0
    assert tiles.context_tile_count(1, 8192) == 1
    assert tiles.context_tile_count(8192, 8192) == 1
    assert tiles.context_tile_count(8193, 8192) == 2


def test_tiled_attention_matches_full_attention():
    torch.manual_seed(0)
    heads_q, heads_kv, dim = 4, 2, 8
    query_len, past_len, tile = 5, 10, 4
    real_q, real_ctx = 3, 6
    scale = dim**-0.5
    query = torch.randn(1, heads_q, query_len, dim)
    key_ctx = torch.randn(1, heads_kv, past_len, dim)
    value_ctx = torch.randn(1, heads_kv, past_len, dim)
    key_q = torch.randn(1, heads_kv, query_len, dim)
    value_q = torch.randn(1, heads_kv, query_len, dim)
    calls = []

    def load_ctx_tile(index):
        start = index * tile
        valid = min(tile, real_ctx - start)
        key = torch.zeros(1, heads_kv, tile, dim)
        value = torch.zeros(1, heads_kv, tile, dim)
        key[:, :, :valid, :] = key_ctx[:, :, start:start + valid, :]
        value[:, :, :valid, :] = value_ctx[:, :, start:start + valid, :]
        return key, value, valid

    def break_graph():
        calls.append(1)

    out = tiles.run_tiled_attention(query, tiles.context_tile_count(real_ctx, tile), load_ctx_tile, key_q, value_q,
                                    real_q, _sdpa, break_graph, scale, "fp32")
    ref = _reference(query, key_ctx, value_ctx, key_q, value_q, real_ctx, real_q, scale)
    assert torch.allclose(out[:, :, :real_q, :], ref[:, :, :real_q, :], atol=1e-5, rtol=1e-5)
    assert len(calls) == tiles.context_tile_count(real_ctx, tile) + 1


def test_bundled_attention_matches_full_attention():
    torch.manual_seed(1)
    heads_q, heads_kv, dim = 4, 2, 8
    query_len, past_len, tile = 5, 28, 4
    real_q, real_ctx = 3, 25  # 7 tiles, bundle rounds up to 8
    scale = dim**-0.5
    query = torch.randn(1, heads_q, query_len, dim)
    key_ctx = torch.randn(1, heads_kv, past_len, dim)
    value_ctx = torch.randn(1, heads_kv, past_len, dim)
    key_q = torch.randn(1, heads_kv, query_len, dim)
    value_q = torch.randn(1, heads_kv, query_len, dim)
    seen = []

    def load_ctx_tile(index):
        start = index * tile
        valid = min(tile, max(real_ctx - start, 0))
        key = torch.zeros(1, heads_kv, tile, dim)
        value = torch.zeros(1, heads_kv, tile, dim)
        if valid > 0:
            key[:, :, :valid, :] = key_ctx[:, :, start:start + valid, :]
            value[:, :, :valid, :] = value_ctx[:, :, start:start + valid, :]
        return key, value, valid

    def sdpa(q, k, v, bias, scale_, mode):
        seen.append(q.shape[0])
        return _sdpa(q, k, v, bias, scale_, mode)

    out = tiles.run_bundled_attention(query, tiles.context_tile_count(real_ctx, tile), load_ctx_tile, key_q, value_q,
                                      real_q, sdpa, lambda: None, scale, "fp32")
    ref = _reference(query, key_ctx, value_ctx, key_q, value_q, real_ctx, real_q, scale)
    assert seen == [8, 1]
    assert torch.allclose(out[:, :, :real_q, :], ref[:, :, :real_q, :], atol=1e-5, rtol=1e-5)


def test_bias_dtype_matches_query():
    heads_q, heads_kv, dim = 2, 1, 4
    query = torch.randn(1, heads_q, 4, dim, dtype=torch.bfloat16)
    key_q = torch.randn(1, heads_kv, 4, dim, dtype=torch.bfloat16)
    value_q = torch.randn(1, heads_kv, 4, dim, dtype=torch.bfloat16)
    seen = []

    def load_ctx_tile(_index):
        key = torch.randn(1, heads_kv, 4, dim, dtype=torch.bfloat16)
        value = torch.randn(1, heads_kv, 4, dim, dtype=torch.bfloat16)
        return key, value, 3

    def sdpa(q, k, v, bias, scale, _softmax_mode):
        seen.append(bias.dtype)
        return _sdpa(q, k, v, bias, scale, _softmax_mode)

    tiles.run_tiled_attention(query, 1, load_ctx_tile, key_q, value_q, 2, sdpa, lambda: None, dim**-0.5, "fast")
    assert seen == [torch.bfloat16, torch.bfloat16]


def test_flag_defaults_off():
    import os
    saved = os.environ.pop("VLLM_HPU_PREFILL_TILE_REPLAY", None)
    try:
        assert tiles.enabled() is False
    finally:
        if saved is not None:
            os.environ["VLLM_HPU_PREFILL_TILE_REPLAY"] = saved


def test_lengths_round_trip():
    tiles.clear_real_lengths()
    assert tiles.real_lengths() is None
    tiles.set_real_lengths(4, 9)
    assert tiles.real_lengths() == (4, 9)
    tiles.clear_real_lengths()
    assert tiles.real_lengths() is None
