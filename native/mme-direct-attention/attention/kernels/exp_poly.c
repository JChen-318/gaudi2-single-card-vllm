// exp(x) for bounded nonpositive scores, no LUT; degree-five residual polynomial.
static float64 approx_exp(float64 x) {
    x=v_f32_max_b(x,-87.0f);
    x=v_f32_min_b(x,0.0f);
    int64 n=v_convert_f32_to_i32_b(x*1.4426950408889634f,SW_RHNE);
    float64 r=x-v_convert_i32_to_f32_b(n,0)*0.6931471805599453f;
    float64 p=0.008333333333333333f;
    p=v_f32_mac_b(p,r,0.041666666666666664f);
    p=v_f32_mac_b(p,r,0.16666666666666666f);
    p=v_f32_mac_b(p,r,0.5f);
    p=v_f32_mac_b(p,r,1.0f);
    p=v_f32_mac_b(p,r,1.0f);
    int64 bits=(n+127)<<23;
    float64 scale=*((float64*)&bits);
    return p*scale;
}
void main(tensor input,tensor output) {
    int5 start=get_index_space_offset(),end=start+get_index_space_size();
    for(int row=start[1];row<end[1];++row)
      for(int block=start[0];block<end[0];++block) {
        int5 p={block*128,row,0,0,0};
        float128 x=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p,input));
        float128 y;
        y.v1=approx_exp(x.v1);y.v2=approx_exp(x.v2);
        v_bf16_st_tnsr(p,output,v_convert_f32_to_bf16_all_b(y));
      }
}
