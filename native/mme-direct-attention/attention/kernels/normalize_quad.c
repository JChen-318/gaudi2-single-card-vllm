// SPDX-License-Identifier: MIT
// Merge two key halves with per-half maxima (FlashAttention-style):
//   m = max(mA, mB), wA = 2^((mA - m) log2 e), wB likewise,
//   out = (wA * a + wB * b) / (wA * lA + wB * lB).
// stats rows r and r + 128 hold {m, l} of halves A and B (FP32, 256 x 2).
// No special-function LUT: base-2 cubic (rel. err. 7.5e-5) and Newton reciprocal.
#ifndef TILE_ROWS
#define TILE_ROWS 128
#endif
static float64 exp_nonpos(float64 x) {
    float64 t = v_f32_mac_b(x, 1.4426950408889634f, (float64)127.0f);
    t = v_f32_max_b(t, 1.0f);
    int64 n = v_convert_f32_to_i32_b(t, SW_RHNE);
    float64 f = t - v_convert_i32_to_f32_b(n, 0);
    float64 p = 0.05517604947090149f;
    p = v_f32_mac_b(p, f, 0.24261149764060974f);
    p = v_f32_mac_b(p, f, 0.6932601928710938f);
    p = v_f32_mac_b(p, f, 0.9999280571937561f);
    int64 bits = n << 23;
    return p * *((float64 *)&bits);
}

void main(tensor partials, tensor stats, tensor output) {
    int5 start=get_index_space_offset(),end=start+get_index_space_size();
    int queries=get_dim_size(partials,1)/4;
    for (int row=start[1];row<end[1];++row) {
        int sr=(row/512)*2048+row%512;
        float64 s0=v_f32_ld_tnsr_b((int5){0,sr,0,0,0},stats);
        float64 s1=v_f32_ld_tnsr_b((int5){0,sr+512,0,0,0},stats);
        float64 s2=v_f32_ld_tnsr_b((int5){0,sr+1024,0,0,0},stats);
        float64 s3=v_f32_ld_tnsr_b((int5){0,sr+1536,0,0,0},stats);
        float64 m0=v_broadcast_element_f32(s0,0),m1=v_broadcast_element_f32(s1,0);
        float64 m2=v_broadcast_element_f32(s2,0),m3=v_broadcast_element_f32(s3,0);
        float64 m=v_f32_max_b(v_f32_max_b(m0,m1),v_f32_max_b(m2,m3));
        float64 w0=exp_nonpos(m0-m),w1=exp_nonpos(m1-m),w2=exp_nonpos(m2-m),w3=exp_nonpos(m3-m);
        float64 l=(w0*v_broadcast_element_f32(s0,1)+w1*v_broadcast_element_f32(s1,1))+
                  (w2*v_broadcast_element_f32(s2,1)+w3*v_broadcast_element_f32(s3,1));
        l=v_f32_max_b(l,1e-30f);
        int64 bits=0x7ef311c3-*((int64 *)&l);float64 inv=*((float64 *)&bits);
        inv*=2.0f-l*inv;inv*=2.0f-l*inv;inv*=2.0f-l*inv;
        w0*=inv;w1*=inv;w2*=inv;w3*=inv;
        float128 x0=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){0,row,0,0,0},partials));
        float128 x1=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){0,row+queries,0,0,0},partials));
        float128 x2=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){0,row+queries*2,0,0,0},partials));
        float128 x3=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){0,row+queries*3,0,0,0},partials));
        x0.v1=(x0.v1*w0+x1.v1*w1)+(x2.v1*w2+x3.v1*w3);
        x0.v2=(x0.v2*w0+x1.v2*w1)+(x2.v2*w2+x3.v2*w3);
        v_bf16_st_tnsr((int5){0,row,0,0,0},output,v_convert_f32_to_bf16_all_b(x0));
    }
}
