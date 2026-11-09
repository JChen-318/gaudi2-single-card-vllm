// SPDX-License-Identifier: MIT
// Fixed0 only: merge four unnormalized key chunks using sum/PV numerator.
// No row-max weighting is needed when every chunk uses the same fixed shift.
#ifndef QUAD_ROWS
#define QUAD_ROWS 512
#endif
void main(tensor partials, tensor stats, tensor output) {
    int5 start=get_index_space_offset(),end=start+get_index_space_size();
    int queries=get_dim_size(partials,1)/4;
    for (int row=start[1];row<end[1];++row) {
        int sr=(row/QUAD_ROWS)*(4*QUAD_ROWS)+row%QUAD_ROWS;
        float64 s0=v_f32_ld_tnsr_b((int5){0,sr,0,0,0},stats);
        float64 s1=v_f32_ld_tnsr_b((int5){0,sr+QUAD_ROWS,0,0,0},stats);
        float64 s2=v_f32_ld_tnsr_b((int5){0,sr+2*QUAD_ROWS,0,0,0},stats);
        float64 s3=v_f32_ld_tnsr_b((int5){0,sr+3*QUAD_ROWS,0,0,0},stats);
        float64 l=(v_broadcast_element_f32(s0,1)+v_broadcast_element_f32(s1,1))+
                  (v_broadcast_element_f32(s2,1)+v_broadcast_element_f32(s3,1));
        l=v_f32_max_b(l,1e-30f);
        int64 bits=0x7ef311c3-*((int64 *)&l);float64 inv=*((float64 *)&bits);
        inv*=2.0f-l*inv;inv*=2.0f-l*inv;inv*=2.0f-l*inv;
        float128 x0=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){0,row,0,0,0},partials));
        float128 x1=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){0,row+queries,0,0,0},partials));
        float128 x2=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){0,row+queries*2,0,0,0},partials));
        float128 x3=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){0,row+queries*3,0,0,0},partials));
        x0.v1=((x0.v1+x1.v1)+(x2.v1+x3.v1))*inv;
        x0.v2=((x0.v2+x1.v2)+(x2.v2+x3.v2))*inv;
        v_bf16_st_tnsr((int5){0,row,0,0,0},output,v_convert_f32_to_bf16_all_b(x0));
    }
}
