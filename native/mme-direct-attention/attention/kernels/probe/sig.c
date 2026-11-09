void main(tensor in, tensor out) {
    int5 p = {0, 0, 0, 0, 0};
    bfloat128 x = v_bf16_ld_tnsr_b(p, in);
    ushort128 i = v_u16_lookup_1c(x);
    bfloat128 n = v_bf16_nearbyint_b(x);
    short128 c = v_convert_bf16_to_i16_b(x);
    ushort128 d = v_u16_sub_b(x);
    v_bf16_st_tnsr(p, out, x);
}
