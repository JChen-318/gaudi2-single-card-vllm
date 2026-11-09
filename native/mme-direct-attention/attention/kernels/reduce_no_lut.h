// SPDX-License-Identifier: MIT
// FP32 tree reduction using explicit lane permutations, without coefficient LUTs.
static float64 sum64_no_lut(float64 x) {
    const int all_groups = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 |
        SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |
        SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 |
        SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3;
    x += v_f32_mov_dual_group_all_b(x, 0xffffffff, 1, 0, 3, 2, all_groups, x);
    x += v_f32_mov_dual_group_all_b(x, 0xffffffff, 2, 3, 0, 1, all_groups, x);
    x += v_f32_mov_group_b(x, 0xffffffff, SW_GROUP0_EN | SW_GROUP1_EN |
        SW_DUAL_GROUP0_EN | SW_DUAL_GROUP1_EN | SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN, x);
    uint64 lanes = read_lane_id_4b_b() & 7;
    uint64 c4 = ((lanes ^ 4) | 0x80) * 0x01010101;
    uint64 c2 = ((lanes ^ 2) | 0x80) * 0x01010101;
    uint64 c1 = ((lanes ^ 1) | 0x80) * 0x01010101;
    x += v_f32_shuffle_b(x, *((uchar256 *)&c4), 0, x);
    x += v_f32_shuffle_b(x, *((uchar256 *)&c2), 0, x);
    x += v_f32_shuffle_b(x, *((uchar256 *)&c1), 0, x);
    return x;
}
