// SPDX-License-Identifier: MIT
void main(tensor input, tensor output) {
    int5 start = get_index_space_offset();
    int5 end = start + get_index_space_size();
    for (int row = start[1]; row < end[1]; ++row)
        for (int block = start[0]; block < end[0]; ++block) {
            int5 p = {block * 128, row, 0, 0, 0};
            bfloat128 x = v_bf16_ld_tnsr_b(p, input);
            v_bf16_st_tnsr(p, output, v_bf16_max_b(x, (bf16)0));
        }
}
