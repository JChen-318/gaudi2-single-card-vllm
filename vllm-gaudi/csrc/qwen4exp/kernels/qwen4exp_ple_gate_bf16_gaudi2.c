// SPDX-License-Identifier: Apache-2.0
// H29: fused PLE gate — the whole of ``ple_runtime.ple_gate`` in one launch.
//
// Torch (pre-H29) per forward: three grouped Gemma RMSNorm calls (the H13
// ``hc_norm`` kernel, 3 launches) + ~12 eager ops (float casts, multiply, sum,
// scalar divide, abs/clamp/sqrt, sign, sigmoid, cast, broadcast multiply,
// reshape) + one more ``hc_norm`` for the conv input, over four [T, 4*2560]
// BF16 planes with six HBM round trips.
//
// This kernel keeps the reference's rounding boundaries (each ``.to(dtype)``
// boundary of the torch chain is an explicit BF16 round here) and does it all
// in one pass structure per (token, stream) member:
//
//   pass 1  sum(x^2) of key and query          (F32, 128-wide accumulation)
//   pass 2  key_n = bf16(key * rsqrt(ms+eps) * (1+gamma_k))
//           query_n = bf16(query * rsqrt(ms+eps) * (1+gamma_q))
//           dot += f32(bf16(key_n * query_n))        -> BF16 product boundary
//   d = bf16(dot / sqrt(H));  sign/magnitude -> BF16 -> gate = bf16(sigmoid)
//   pass 3  gated = bf16(gate * value), sum(gated^2) (F32 over BF16 values)
//   pass 4  conv_input = bf16(gated * rsqrt(mean+eps) * (1 + gamma_conv))
//
// ``value`` is single-stream ([T, H]) while key/query are multi-stream
// ([T, S*H]): the gate scalar is per (token, stream), the value row is shared.
//
// Layouts (TPC dim 0 = torch's contiguous last dim):
//   key, query [T, S*H] -> (S*H, T);  value [T, H] -> (H, T)
//   gamma_* [Sg, H] -> (H, Sg);  gated, conv_input [T, S*H] -> (S*H, T)
// Index space (streams, tokens): one member owns one (stream, token) row.
#include "qwen4exp_hc_common.h"

//: RNE f32 -> bf16 -> f32 round trip. The reference materializes these
//: boundaries in BF16; ``v_convert_*`` are the same rounding the store path
//: uses, so the value matches what torch's ``.to(torch.bfloat16)`` produces.
static inline bfloat128 q4_ple_gate_round128(float128 x) {
    return v_convert_f32_to_bf16_all_b(x);
}

static inline float64 q4_ple_gate_round_scalar(float64 x) {
    float128 pair;
    pair.v1 = x;
    pair.v2 = x;
    return v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(pair)).v1;
}

void main(tensor key, tensor query, tensor value, tensor gamma_key, tensor gamma_query,
          tensor gamma_conv, tensor gated, tensor conv_input, float eps) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int width = get_dim_size(value, 0);       // H
    const int gamma_rows = get_dim_size(gamma_key, 1);
    const float inv_width = 1.0f / (float)width;
    // 1 / sqrt(H): the reference divides the F32 dot by ``float(H) ** 0.5``.
    const float64 inv_scale = v_rsqrt_f32((float)width);
    for (int token = start[1]; token < end[1]; ++token) {
        for (int stream = start[0]; stream < end[0]; ++stream) {
            const int base = stream * width;
            const int grow = gamma_rows == 1 ? 0 : stream;
            // ---- pass 1: RMS of key and query (identical to hc_norm) ------
            float128 square_key = {0};
            float128 square_query = {0};
            for (int col = 0; col < width; col += 128) {
                const int5 p = {base + col, token, 0, 0, 0};
                const float128 k = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, key));
                const float128 q = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, query));
                square_key.v1 += k.v1 * k.v1;
                square_key.v2 += k.v2 * k.v2;
                square_query.v1 += q.v1 * q.v1;
                square_query.v2 += q.v2 * q.v2;
            }
            const float64 inv_key = v_rsqrt_f32(
                v_f32_reduce_add(square_key.v1 + square_key.v2) * inv_width + eps);
            const float64 inv_query = v_rsqrt_f32(
                v_f32_reduce_add(square_query.v1 + square_query.v2) * inv_width + eps);
            // ---- pass 2: BF16-normalized dot product ----------------------
            float128 dot = {0};
            for (int col = 0; col < width; col += 128) {
                const int5 p = {base + col, token, 0, 0, 0};
                const int5 g = {col, grow, 0, 0, 0};
                const float128 k = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, key));
                const float128 q = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, query));
                const float128 wk = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(g, gamma_key));
                const float128 wq = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(g, gamma_query));
                float128 kn;
                float128 qn;
                kn.v1 = (k.v1 * inv_key) * (1.0f + wk.v1);
                kn.v2 = (k.v2 * inv_key) * (1.0f + wk.v2);
                qn.v1 = (q.v1 * inv_query) * (1.0f + wq.v1);
                qn.v2 = (q.v2 * inv_query) * (1.0f + wq.v2);
                const float128 knb = v_convert_bf16_to_f32_all_b(q4_ple_gate_round128(kn));
                const float128 qnb = v_convert_bf16_to_f32_all_b(q4_ple_gate_round128(qn));
                float128 product;
                product.v1 = knb.v1 * qnb.v1;
                product.v2 = knb.v2 * qnb.v2;
                const float128 product_bf =
                    v_convert_bf16_to_f32_all_b(q4_ple_gate_round128(product));
                dot.v1 += product_bf.v1;
                dot.v2 += product_bf.v2;
            }
            // ---- gate scalar (every boundary the reference materializes) --
            const float64 scaled = q4_ple_gate_round_scalar(
                v_f32_reduce_add(dot.v1 + dot.v2) * inv_scale);
            const float64 magnitude = v_f32_max_b(scaled, scaled * -1.0f);
            const float64 clamped = v_f32_max_b(magnitude, 1e-6f);
            const float64 root = q4_ple_gate_round_scalar(clamped * v_rsqrt_f32(clamped));
            // torch.sign(d): 0 at d == 0. The clamped magnitude is >= 1e-3
            // there, so the +root this spelling picks rounds to the same BF16
            // gate (sigmoid(1e-3) = 0.50025 -> 0.5) as the reference's 0.
            const float64 signed_root = v_f32_sel_less_f32_b(scaled, 0.0f, root * -1.0f, root);
            const float64 gate = q4_ple_gate_round_scalar(qwen4exp_hc_sigmoid(signed_root));
            // ---- pass 3: gated = bf16(gate * value), and its sum of squares
            float128 square_gated = {0};
            for (int col = 0; col < width; col += 128) {
                const int5 p = {col, token, 0, 0, 0};
                const int5 o = {base + col, token, 0, 0, 0};
                const float128 v = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, value));
                float128 gv;
                gv.v1 = gate * v.v1;
                gv.v2 = gate * v.v2;
                const bfloat128 gb = q4_ple_gate_round128(gv);
                const float128 gbf = v_convert_bf16_to_f32_all_b(gb);
                square_gated.v1 += gbf.v1 * gbf.v1;
                square_gated.v2 += gbf.v2 * gbf.v2;
                v_bf16_st_tnsr(o, gated, gb);
            }
            const float64 inv_gated = v_rsqrt_f32(
                v_f32_reduce_add(square_gated.v1 + square_gated.v2) * inv_width + eps);
            // ---- pass 4: grouped Gemma RMSNorm of the gated plane ----------
            for (int col = 0; col < width; col += 128) {
                const int5 p = {col, token, 0, 0, 0};
                const int5 g = {col, grow, 0, 0, 0};
                const int5 o = {base + col, token, 0, 0, 0};
                const float128 v = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, value));
                const float128 wc = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(g, gamma_conv));
                float128 gv;
                gv.v1 = gate * v.v1;
                gv.v2 = gate * v.v2;
                const float128 gbf =
                    v_convert_bf16_to_f32_all_b(q4_ple_gate_round128(gv));
                float128 y;
                y.v1 = (gbf.v1 * inv_gated) * (1.0f + wc.v1);
                y.v2 = (gbf.v2 * inv_gated) * (1.0f + wc.v2);
                v_bf16_st_tnsr(o, conv_input, q4_ple_gate_round128(y));
            }
        }
    }
}
