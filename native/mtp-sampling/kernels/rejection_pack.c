// SPDX-License-Identifier: Apache-2.0
// Repeat the <=4 scalar acceptance comparisons, reduce tile winners and pack.
void main(tensor probs, tensor draft_ids, tensor cumulative, tensor uniforms,
          tensor bonus, tensor tile_scores, tensor tile_ids, tensor output) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int vocab = get_dim_size(probs, 0);
    const int width = get_dim_size(probs, 1);
    const int tiles = get_dim_size(tile_scores, 0);
    for (int batch = start[0]; batch < end[0]; ++batch) {
        int5 c = {batch, 0, 0, 0, 0};
        const int total = s_i32_ld_g(gen_addr(c, cumulative));
        c[0] = batch > 0 ? batch - 1 : 0;
        const int previous = batch > 0 ? s_i32_ld_g(gen_addr(c, cumulative)) : 0;
        const int raw_length = total - previous;
        const int length = raw_length < 0 ? 0 : (raw_length > width ? width : raw_length);
        int first = length;
        for (int step = 0; step < length; ++step) {
            int5 d = {step, batch, 0, 0, 0};
            const int token = s_i32_ld_g(gen_addr(d, draft_ids));
            const float uniform = s_f32_ld_g(gen_addr(d, uniforms));
            bool accepted = 0;
            if (token >= 0 && token < vocab) {
                int5 p = {token, step, batch, 0, 0};
                accepted = uniform < s_f32_ld_g(gen_addr(p, probs));
            }
            if (!accepted) { first = step; break; }
        }
        float maximum = as_float(0xff800000);
        int winner = 0;
        if (first < length) {
            for (int tile = 0; tile < tiles; ++tile) {
                int5 t = {tile, batch, 0, 0, 0};
                const float value = s_f32_ld_g(gen_addr(t, tile_scores));
                const int id = s_i32_ld_g(gen_addr(t, tile_ids));
                if ((value != value && (maximum == maximum || id < winner)) ||
                    (value == value && maximum == maximum &&
                     (value > maximum || (value == maximum && id < winner)))) { maximum = value; winner = id; }
            }
        }
        for (int step = 0; step <= width; ++step) {
            int value = -1;
            int5 d = {step, batch, 0, 0, 0};
            if (step < first) value = s_i32_ld_g(gen_addr(d, draft_ids));
            if (step == first) {
                int5 b = {0, batch, 0, 0, 0};
                value = first < length ? winner : s_i32_ld_g(gen_addr(b, bonus));
            }
            s_i32_st_g(gen_addr(d, output), value);
        }
    }
}
