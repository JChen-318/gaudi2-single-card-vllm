// SPDX-License-Identifier: Apache-2.0
// Deterministic proposal, all-random target. RNG and softmax stay outside.
// One tile scans 1024 vocabulary entries; independent tiles run concurrently.
void main(tensor probs, tensor draft_ids, tensor cumulative, tensor uniforms,
          tensor noise, tensor tile_scores, tensor tile_ids) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int vocab = get_dim_size(probs, 0);
    const int width = get_dim_size(probs, 1);
    const int tile_width = 1024;
    const int64 lanes = (int64)read_lane_id_4b_b();
    for (int batch = start[1]; batch < end[1]; ++batch) {
        int5 c = {batch, 0, 0, 0, 0};
        const int total = s_i32_ld_g(gen_addr(c, cumulative));
        c[0] = batch > 0 ? batch - 1 : 0;
        const int previous = batch > 0 ? s_i32_ld_g(gen_addr(c, cumulative)) : 0;
        const int raw_length = total - previous;
        const int length = raw_length < 0 ? 0 : (raw_length > width ? width : raw_length);
        int first = length;
        int rejected_id = -1;
        for (int step = 0; step < length; ++step) {
            int5 d = {step, batch, 0, 0, 0};
            const int token = s_i32_ld_g(gen_addr(d, draft_ids));
            const float uniform = s_f32_ld_g(gen_addr(d, uniforms));
            bool accepted = 0;
            if (token >= 0 && token < vocab) {
                int5 p = {token, step, batch, 0, 0};
                accepted = uniform < s_f32_ld_g(gen_addr(p, probs));
            }
            if (!accepted) { first = step; rejected_id = token; break; }
        }
        for (int tile = start[0]; tile < end[0]; ++tile) {
            float64 best = as_float64((int64)0xff800000);
            int64 best_id = 2147483647;
            if (first < length) {
                for (int offset = 0; offset < tile_width; offset += 64) {
                    const int base = tile * tile_width + offset;
                    int5 p = {base, first, batch, 0, 0};
                    int5 n = {base, batch, 0, 0, 0};
                    float64 probability = v_f32_ld_tnsr_b(p, probs);
                    const int64 ids = base + lanes;
                    probability = v_f32_sel_eq_i32_b(ids, rejected_id, 0.0f, probability);
                    const float64 exponential = v_f32_ld_tnsr_b(n, noise);
                    float64 score = probability / exponential;
                    score = v_f32_sel_geq_i32_b(ids, vocab, as_float64((int64)0xff800000), score);
                    const bool64 nan = v_f32_cmp_neq_b(score, score);
                    const bool64 best_nan = v_f32_cmp_neq_b(best, best);
                    const bool64 lower_id = v_i32_cmp_less_b(ids, best_id);
                    const bool64 better = (nan & (~best_nan | lower_id)) |
                        (~nan & ~best_nan & (v_f32_cmp_grt_b(score, best) |
                        (v_f32_cmp_eq_b(score, best) & lower_id)));
                    best = v_f32_mov_vb(score, 0, best, better);
                    best_id = v_i32_mov_vb(ids, 0, best_id, better);
                }
            }
            // VLM and SLM are distinct memories: scalar loads cannot read a
            // vector-local store. Keep winner reduction entirely in vector lanes.
            const bool64 nan = v_f32_cmp_neq_b(best, best);
            const int64 nan_ids = v_i32_mov_vb(best_id, 0, (int64)2147483647, nan);
            const int64 nan_winner = v_i32_reduce_min(nan_ids);
            const float64 finite_best = v_f32_mov_vb(as_float64((int64)0xff800000), 0, best, nan);
            float64 maximum = v_f32_reduce_max(finite_best);
            const bool64 tied = v_f32_cmp_eq_b(finite_best, maximum);
            const int64 tied_ids = v_i32_mov_vb(best_id, 0, (int64)2147483647, tied);
            int64 winner = v_i32_reduce_min(tied_ids);
            const bool64 has_nan = v_i32_cmp_less_b(nan_winner, (int64)2147483647);
            winner = v_i32_mov_vb(nan_winner, 0, winner, has_nan);
            maximum = v_f32_mov_vb(as_float64((int64)0x7fc00000), 0, maximum, has_nan);
            int5 out = {tile, batch, 0, 0, 0};
            v_f32_st_tnsr_partial(out, tile_scores, maximum, 0, 0);
            v_i32_st_tnsr_partial(out, tile_ids, winner, 0, 0);
        }
    }
}
