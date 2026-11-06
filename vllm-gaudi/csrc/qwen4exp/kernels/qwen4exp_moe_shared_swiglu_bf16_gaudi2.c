// SPDX-License-Identifier: Apache-2.0
// H15 Optimization 2 (epilogue): fused shared-expert SwiGLU + gate scalar.
//
//   merged [T, 2*I+1] BF16 = [gate | up | shared_logit]  (one MME GEMM over
//   the in-memory merged weight [gate_up; shared_expert_gate])
//   out[t, i] = bf16( bf16(silu_bf16(gate) * up) * bf16(sigmoid(shared_logit)) )
//
// The per-token sigmoid scalar is applied to the activation BEFORE the down
// projection instead of to its output. sigmoid(s) is a per-token scalar and
// down is linear, so down @ (c * act) == c * (down @ act) in real arithmetic;
// only the position of one BF16 rounding moves (checked at the MoE-block
// tolerance, rel < 2e-2 / cosine > 0.999, in bench_moe_h15.py). In return the
// shared expert is: merged GEMM -> this kernel -> down GEMM (3 launches).
//
// The silu step reproduces the unfused BF16 tensor chain: silu evaluated in
// F32 on the BF16 gate, rounded to BF16 (F.silu's output tensor), multiplied
// by the BF16 up, rounded to BF16; the sigmoid scalar is likewise the BF16
// value F.sigmoid would materialize.
static inline float128 moe_bf16_round128(float128 v) {
    return v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(v));
}

void main(tensor merged, tensor output)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int inter = get_dim_size(output, 0);
    for (int token = start[1]; token < end[1]; ++token) {
        const float sg = (float)s_bf16_ld_g(gen_addr((int5){2 * inter, token, 0, 0, 0}, merged));
        float64 sg_vec = sg;
        const float64 sig = v_sigmoid_f32(sg_vec);
        float128 sig_wide;
        sig_wide.v1 = sig;
        sig_wide.v2 = sig;
        const float128 sig_bf = moe_bf16_round128(sig_wide);
        for (int tile = start[0]; tile < end[0]; ++tile) {
            const int col = tile * 128;
            if (col >= inter) break;
            const float128 g = v_convert_bf16_to_f32_all_b(
                v_bf16_ld_tnsr_b((int5){col, token, 0, 0, 0}, merged));
            const float128 u = v_convert_bf16_to_f32_all_b(
                v_bf16_ld_tnsr_b((int5){col + inter, token, 0, 0, 0}, merged));
            float128 silu;
            silu.v1 = g.v1 * v_sigmoid_f32(g.v1);
            silu.v2 = g.v2 * v_sigmoid_f32(g.v2);
            const float128 silu_bf = moe_bf16_round128(silu);
            float128 act;
            act.v1 = silu_bf.v1 * u.v1;
            act.v2 = silu_bf.v2 * u.v2;
            const float128 act_bf = moe_bf16_round128(act);
            float128 y;
            y.v1 = act_bf.v1 * sig_bf.v1;
            y.v2 = act_bf.v2 * sig_bf.v2;
            const bfloat128 out = v_convert_f32_to_bf16_all_b(y);
            const int5 p = {col, token, 0, 0, 0};
            if (col + 128 <= inter) {
                v_bf16_st_tnsr(p, output, out);
            } else {
                v_bf16_st_tnsr_partial(p, output, out, inter - col - 1, 0);
            }
        }
    }
}
