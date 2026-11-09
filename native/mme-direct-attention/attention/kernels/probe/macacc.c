void main(tensor in, tensor out) {
    int5 p = {0, 0, 0, 0, 0};
    bfloat128 x = v_bf16_ld_tnsr_b(p, in);
    float128 acc = {0};
    acc = v_bf16_mac_acc32_b(x, (bfloat128)1.0f, acc, 0, 1, 0);
    v_f32_st_tnsr(p, out, acc.v1 + acc.v2);
}
