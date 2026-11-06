// SPDX-License-Identifier: Apache-2.0
// H15 Optimization 1 (epilogue): fused top-k weighted reduction after down.
//
//   out[t, h] = sum_k bf16(slot_down[t*K+k, h] * w[t, k]),   K = top_k
//
// Replaces (slot_down * w[:, None].to(bf16)) + view(T, K, H).sum(1): two
// dispatched ops on the [S, H] plane become one. Per-slot products are
// rounded to BF16 first (the multiply's output tensor), accumulated in F32
// in ascending slot order, and rounded to BF16 once for the store -- the
// accumulation form torch's sum takes on this plane (bench_moe_h15.py
// verifies the BF16 bit patterns against the unfused chain).
static inline float128 moe_bf16_round128(float128 v) {
    return v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(v));
}

void main(tensor slot_down, tensor topk_weights, tensor output)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int width = get_dim_size(output, 0);
    const int topk = get_dim_size(topk_weights, 0);
    for (int token = start[1]; token < end[1]; ++token) {
        for (int tile = start[0]; tile < end[0]; ++tile) {
            const int col = tile * 128;
            if (col >= width) break;
            float128 sum = {0};
            for (int k = 0; k < topk; ++k) {
                const int slot = token * topk + k;
                const float w = (float)s_bf16_ld_g(
                    gen_addr((int5){k, token, 0, 0, 0}, topk_weights));
                float64 w_vec = w;
                float128 w128;
                w128.v1 = w_vec;
                w128.v2 = w_vec;
                const float128 d = v_convert_bf16_to_f32_all_b(
                    v_bf16_ld_tnsr_b((int5){col, slot, 0, 0, 0}, slot_down));
                float128 prod;
                prod.v1 = d.v1 * w128.v1;
                prod.v2 = d.v2 * w128.v2;
                const float128 prod_bf = moe_bf16_round128(prod);
                sum.v1 += prod_bf.v1;
                sum.v2 += prod_bf.v2;
            }
            const bfloat128 out = v_convert_f32_to_bf16_all_b(sum);
            const int5 p = {col, token, 0, 0, 0};
            if (col + 128 <= width) {
                v_bf16_st_tnsr(p, output, out);
            } else {
                v_bf16_st_tnsr_partial(p, output, out, width - col - 1, 0);
            }
        }
    }
}
