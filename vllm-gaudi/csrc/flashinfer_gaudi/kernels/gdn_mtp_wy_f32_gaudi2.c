// SPDX-License-Identifier: Apache-2.0
// WY-form full-query MTP verification core (B<=16, T=TOKENS (8 or 16), H=16, HV=48, K=V=128).
//
// Same recurrence and outputs as gdn_mtp_fused_f32_gaudi2.c, but the dot products of the evolving state with every
// k_t and q_t are not computed here. The caller precomputes on the MME (FP32 matmuls):
//   rmat [B, 48, 2T, V] : rows t < T: S0 k_t (raw k), rows T + t: S0 q_t (raw q)     (S0: gathered initial state)
//   gram [B, 16, 2T, 2T]: X X^T for X = [k_0..k_{T-1}; q_0..q_{T-1}] (raw), one per key head
//   kvals [B, T, 16, K] : raw k in FP32;  vals [B, T, 48, V]: v in FP32
// With rk = 1/|k|, rq = scale/|q| (as the fused kernel), G_t = prod_{i<=t} g_i and ds_t = rk_t delta_t:
//   ds_t  = rk_t beta_t v_t - beta_t G_t rk_t^2 (S0 k_t) - sum_{j<t} beta_t (G_t/G_j) rk_t^2 (k_t.k_j) ds_j
//   out_t = G_t rq_t (S0 q_t) + sum_{j<=t} (G_t/G_j) rq_t (q_t.k_j) ds_j
//   S_t   = g_t S_{t-1} + ds_t k_t^T                       (the checkpoint update of the fused kernel)
// Phase 1 (per head, all 128 rows as two 64-lane vectors): ds_t and out_t; out_t is stored here.
// Phase 2 (tiles of ROWS rows carried in registers across all tokens; each state row is read once): only the
// state update and checkpoint stores. Lanes are moved with lane-XOR swaps (the fused kernel's reduction moves),
// so every loop over tokens is a runtime loop: small code, low register pressure.
#define ROWS 8
#ifndef TOKENS
#define TOKENS 16
#endif
#if TOKENS != 8 && TOKENS != 16
#error "TOKENS must be 8 or 16"
#endif
// lanes of a coefficient vector: [0, T) -a_tj, [T, 2T) e_tj, then per-token scalars
#define L_RKB (2 * TOKENS)        // rk_t beta_t
#define L_B (2 * TOKENS + 1)      // beta_t G_t rk_t^2
#define L_C (2 * TOKENS + 2)      // G_t rq_t
#define L_G (2 * TOKENS + 3)      // g_t

#define XOR_LUT(name, m)                                                                                       \
    uint64 name##_bytes = 0x80808080;                                                                          \
    name##_bytes += v_u32_xor_b(read_lane_id_4b_b() * 0x01010101, (m) * 0x01010101);                          \
    name##_bytes = v_u32_mov_group_b(name##_bytes, 0xFFFFFFFF, (SW_GROUP1_EN | SW_DUAL_GROUP0_EN), name##_bytes, \
                                     1, 0);                                                                    \
    name##_bytes = v_u32_add_vb(name##_bytes, 0x20202020, 0, name##_bytes, to_bool64(pred_group_1));           \
    name##_bytes = v_u32_mov_dual_group_all_b(name##_bytes, 0xFFFFFFFF, 0, 0, 0, 0,                            \
                                              MkWrA(0b11, 0b11, 0b11, 0b11), 0);                               \
    const uchar256 name = (uchar256)name##_bytes;

#define XOR8(v) v_f32_mov_group_b((v), 0xFFFFFFFF, SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN |           \
                                  SW_DUAL_GROUP1_EN | SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN, (v), 1, 0)
#define XOR16(v) v_f32_mov_dual_group_all_b((v), 0xFFFFFFFF, 1, 0, 3, 2, MkWrA(0b11, 0b11, 0b11, 0b11), (v))
#define XOR32(v) v_f32_mov_dual_group_all_b((v), 0xFFFFFFFF, 2, 3, 0, 1, MkWrA(0b11, 0b11, 0b11, 0b11), (v))
#define P1(v) v_f32_shuffle_b((v), lut_x1, 0, (v))
#define P2(v) v_f32_shuffle_b((v), lut_x2, 0, (v))
#define P4(v) v_f32_shuffle_b((v), lut_x4, 0, (v))
#if TOKENS == 16
#define SWAP_T(v) XOR16(v)
#else
#define SWAP_T(v) XOR8(v)
#endif
// all-true / all-false predicate from a scalar condition
#define PRED(cond) v_u32_cmp_less_b(lanes, (uint64)((cond) ? 64 : 0))
// v with lane (idx ^ x) moved to lane x for every x (idx < 64, runtime): lane 0 then holds v[idx]
#define LANE_TO_0(v, idx)                                                                                     \
    ({                                                                                                        \
        float64 _v = (v);                                                                                     \
        const int _i = (idx);                                                                                 \
        _v = v_f32_mov_vb(P1(_v), 0, _v, PRED(_i & 1));                                                        \
        _v = v_f32_mov_vb(P2(_v), 0, _v, PRED(_i & 2));                                                        \
        _v = v_f32_mov_vb(P4(_v), 0, _v, PRED(_i & 4));                                                        \
        _v = v_f32_mov_vb(XOR8(_v), 0, _v, PRED(_i & 8));                                                      \
        _v = v_f32_mov_vb(XOR16(_v), 0, _v, PRED(_i & 16));                                                    \
        _v = v_f32_mov_vb(XOR32(_v), 0, _v, PRED(_i & 32));                                                    \
        _v;                                                                                                   \
    })
#define LANE(v, idx) v_broadcast_element_f32(LANE_TO_0((v), (idx)), 0)

// Per token t: lanes [0, T) -a_tj (0 for j >= t), [T, 2T) e_tj (0 for j > t), L_RKB.. per-token scalars.
volatile __local__ float64 token_coef[TOKENS];
// ds_t for rows 0..63 (index 2t) and 64..127 (index 2t + 1); zeroed once (masked terms multiply it by 0).
__local__ float64 token_delta[2 * TOKENS];

void main(tensor source, tensor kvals, tensor log_decay, tensor beta, tensor rmat, tensor gram, tensor vals,
          tensor output, tensor checkpoints)
{
    const uint64 lanes = read_lane_id_4b_b();
    const float q_scale = 0.08838834764831845f;  // 128 ** -0.5
    const int5 begin = get_index_space_offset();
    const int5 end = get_index_space_size() + begin;
    const float64 zero = 0.0f;
    for (int i = 0; i < 2 * TOKENS; ++i) token_delta[i] = zero;

    for (int batch = begin[3]; batch < end[3]; ++batch) {
        for (int head = begin[2]; head < end[2]; ++head) {
            const int key_head = head / 3;
            // ---- per-token scalars as lane vectors (lane t) ----
            // k_t.k_t is lane t of gram row t, q_t.q_t lane T + t of row T + t: select lanes, no movement
            float64 kdiag = zero, qdiag = zero, ldv = zero, betav = zero;
            int5 krow_coords = {0, 0, key_head, batch, 0};
            int5 qrow_coords = {0, TOKENS, key_head, batch, 0};
            float64 next_krow = v_f32_ld_tnsr_b(krow_coords, gram);
            float64 next_qrow = v_f32_ld_tnsr_b(qrow_coords, gram);
            for (int t = 0; t < TOKENS; ++t) {
                const float64 krow_now = next_krow, qrow_now = next_qrow;
                const int tn = t + 1 < TOKENS ? t + 1 : t;
                krow_coords[1] = tn; qrow_coords[1] = TOKENS + tn;
                next_krow = v_f32_ld_tnsr_b(krow_coords, gram);
                next_qrow = v_f32_ld_tnsr_b(qrow_coords, gram);
                const bool64 here = v_u32_cmp_eq_b(lanes, (uint64)t);
                kdiag = v_f32_mov_vb(krow_now, 0, kdiag, here);
                qdiag = v_f32_mov_vb(qrow_now, 0, qdiag, v_u32_cmp_eq_b(lanes, (uint64)(TOKENS + t)));
                int5 scalar_coords = {head, t, batch, 0, 0};
                const float64 ld = v_broadcast_element_f32(v_f32_ld_tnsr_b(scalar_coords, log_decay), 0);
                const float beta_value = (float)s_bf16_ld_g((__global__ bf16*)gen_addr(scalar_coords, beta));
                ldv = v_f32_mov_vb(ld, 0, ldv, here);
                betav = v_f32_mov_vb(zero + beta_value, 0, betav, here);
            }
            const float64 rkv = v_rsqrt_f32(kdiag + 1e-6f);
            const float64 rqv = v_rsqrt_f32(SWAP_T(qdiag) + 1e-6f) * q_scale;
            const float64 gv = v_exp_f32(ldv);
            // G_t / G_j and G_t as running products of g (no exp per token: special-function LUT reloads made
            // the T16 kernel 4x slower; products also underflow gracefully where exp(cl_t - cl_j) would not)
            float64 ratio = zero;
            float64 Gcur = zero + 1.0f;
            // ---- coefficients ----
            #pragma unroll
            for (int t = 0; t < TOKENS; ++t) {
                int5 krow_coords = {0, t, key_head, batch, 0};
                const float64 krow = v_f32_ld_tnsr_b(krow_coords, gram);        // lanes j: k_t . k_j
                int5 qrow_coords = {0, TOKENS + t, key_head, batch, 0};
                const float64 qrow = v_f32_ld_tnsr_b(qrow_coords, gram);        // lanes j: q_t . k_j
                const float64 g_t = v_broadcast_element_f32(gv, t);
                Gcur = Gcur * g_t;                                               // G_t
                ratio = ratio * g_t;                                             // lanes j < t: G_t / G_j
                ratio = v_f32_mov_vb(zero + 1.0f, 0, ratio, v_u32_cmp_eq_b(lanes, (uint64)t));
                const float64 rk_t = v_broadcast_element_f32(rkv, t);
                const float64 beta_t = v_broadcast_element_f32(betav, t);
                const float64 rq_t = v_broadcast_element_f32(rqv, t);
                const float64 neg_a = (ratio * krow) * (-(beta_t * rk_t * rk_t));
                const float64 e = (ratio * qrow) * rq_t;
                const float64 a_masked = v_f32_mov_vb(neg_a, 0, zero, v_u32_cmp_less_b(lanes, (uint64)t));
                const float64 e_masked = v_f32_mov_vb(e, 0, zero, v_u32_cmp_leq_b(lanes, (uint64)t));
                float64 coef = v_f32_mov_vb(SWAP_T(e_masked), 0, a_masked, v_u32_cmp_geq_b(lanes, (uint64)TOKENS));
                coef = v_f32_mov_vb(rk_t * beta_t, 0, coef, v_u32_cmp_eq_b(lanes, (uint64)L_RKB));
                coef = v_f32_mov_vb(beta_t * Gcur * rk_t * rk_t, 0, coef, v_u32_cmp_eq_b(lanes, (uint64)L_B));
                coef = v_f32_mov_vb(Gcur * rq_t, 0, coef, v_u32_cmp_eq_b(lanes, (uint64)L_C));
                coef = v_f32_mov_vb(g_t, 0, coef, v_u32_cmp_eq_b(lanes, (uint64)L_G));
                token_coef[t] = coef;
            }
            // ---- phase 1: ds_t and out_t for all rows (masked coefficients zero the j >= t / j > t terms) ----
            #pragma unroll
            for (int part = 0; part < 2; ++part) {
                // software pipelined: the next token's global loads are issued before this token's chain
                int5 vcoords = {64 * part, head, 0, batch, 0};
                int5 kcoords1 = {64 * part, 0, head, batch, 0};
                int5 qcoords1 = {64 * part, TOKENS, head, batch, 0};
                float64 next_values = v_f32_ld_tnsr_b(vcoords, vals);
                float64 next_rk = v_f32_ld_tnsr_b(kcoords1, rmat);
                float64 next_rq = v_f32_ld_tnsr_b(qcoords1, rmat);
                for (int t = 0; t < TOKENS; ++t) {
                    const float64 coef = token_coef[t];
                    const float64 values = next_values, rk_col = next_rk, rq_now = next_rq;
                    const int tn = t + 1 < TOKENS ? t + 1 : t;
                    vcoords[2] = tn; kcoords1[1] = tn; qcoords1[1] = TOKENS + tn;
                    next_values = v_f32_ld_tnsr_b(vcoords, vals);
                    next_rk = v_f32_ld_tnsr_b(kcoords1, rmat);
                    next_rq = v_f32_ld_tnsr_b(qcoords1, rmat);
                    float64 acc[4];
                    acc[0] = values * v_broadcast_element_f32(coef, L_RKB) - rk_col * v_broadcast_element_f32(coef, L_B);
                    acc[1] = zero; acc[2] = zero; acc[3] = zero;
                    #pragma unroll
                    for (int j = 0; j < TOKENS; ++j)
                        acc[j & 3] = v_f32_mac_b(token_delta[2 * j + part], v_broadcast_element_f32(coef, j), acc[j & 3]);
                    const float64 ds = (acc[0] + acc[1]) + (acc[2] + acc[3]);
                    token_delta[2 * t + part] = ds;
                    const float64 rq_col = rq_now;
                    float64 oc[4];
                    oc[0] = rq_col * v_broadcast_element_f32(coef, L_C);
                    oc[1] = zero; oc[2] = zero; oc[3] = zero;
                    #pragma unroll
                    for (int j = 0; j < TOKENS; ++j)
                        oc[j & 3] = v_f32_mac_b(token_delta[2 * j + part], v_broadcast_element_f32(coef, TOKENS + j),
                                                oc[j & 3]);
                    const float64 o = (oc[0] + oc[1]) + (oc[2] + oc[3]);
                    int5 out_coords = {64 * part, head, t, batch, 0};
                    v_f32_st_tnsr(out_coords, output, o);
                }
            }
#if defined(DEFER) && TOKENS == 16
            // ---- deferred: no state update here; ds_t goes to checkpoints[b, t, h, 0, :] (row 0 of each
            // checkpoint), the caller rebuilds the accepted state from S0 and these deltas in the next round ----
            for (int t = 0; t < TOKENS; ++t) {
                #pragma unroll
                for (int part = 0; part < 2; ++part) {
                    int5 coords = {64 * part, 0, head, t, batch};
                    v_f32_st_tnsr(coords, checkpoints, token_delta[2 * t + part]);
                }
            }
#elif !defined(NO_PHASE2)
            // ---- phase 2: state update and checkpoints, tiles of ROWS rows ----
            for (int tile = 0; tile < 128 / ROWS; ++tile) {
                const int part = tile >> 3;
                const int sub = tile & 7;                          // tile rows are lanes 8 sub .. 8 sub + 7
                const bool64 p8 = PRED(sub & 1), p16 = PRED(sub & 2), p32 = PRED(sub & 4);
                float64 state_lo[ROWS];
                float64 state_hi[ROWS];
                #pragma unroll
                for (int row = 0; row < ROWS; ++row) {
                    int5 coords = {0, tile * ROWS + row, head, batch, 0};
                    state_lo[row] = v_f32_ld_tnsr_b(coords, source);
                    coords[0] = 64;
                    state_hi[row] = v_f32_ld_tnsr_b(coords, source);
                }
#ifdef REPLAY_TOKENS
                for (int t = 0; t < REPLAY_TOKENS; ++t) {
#else
                for (int t = 0; t < TOKENS; ++t) {
#endif
                    float64 dv = token_delta[2 * t + part];
                    dv = v_f32_mov_vb(XOR8(dv), 0, dv, p8);
                    dv = v_f32_mov_vb(XOR16(dv), 0, dv, p16);
                    dv = v_f32_mov_vb(XOR32(dv), 0, dv, p32);       // lanes 0..7: ds_t of the tile rows
                    const float64 g = v_broadcast_element_f32(token_coef[t], L_G);
                    int5 kcoords = {0, key_head, t, batch, 0};
                    const float64 k_lo = v_f32_ld_tnsr_b(kcoords, kvals);
                    kcoords[0] = 64;
                    const float64 k_hi = v_f32_ld_tnsr_b(kcoords, kvals);
                    #pragma unroll
                    for (int row = 0; row < ROWS; ++row) {
                        const float64 dr = v_broadcast_element_f32(dv, row);
                        state_lo[row] = v_f32_mac_b(k_lo, dr, state_lo[row] * g);
                        state_hi[row] = v_f32_mac_b(k_hi, dr, state_hi[row] * g);
#ifdef STORE_LAST
#ifdef REPLAY_TOKENS
                        if (t == REPLAY_TOKENS - 1) {
#else
                        if (t == TOKENS - 1) {
#endif
#endif
                        int5 coords = {0, tile * ROWS + row, head, t, batch};
                        v_f32_st_tnsr(coords, checkpoints, state_lo[row]);
                        coords[0] = 64;
                        v_f32_st_tnsr(coords, checkpoints, state_hi[row]);
#ifdef STORE_LAST
                        }
#endif
                    }
                }
            }
#endif
        }
    }
}
