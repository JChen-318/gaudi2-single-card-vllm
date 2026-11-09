// SPDX-License-Identifier: MIT
// FP16 scores -> BF16 probabilities in the same 2-byte storage. Tensors have
// 16-bit elements; ld_tnsr.f16 reads score bits, st_tnsr.bf16 writes P bits.
// FP32 half maxima/sums, LUT128 slot14 logq correction. Caller bound <=80.
#include "reduce_no_lut.h"
#include "reduce_max_no_lut.h"
#ifndef MAX_STRIDE
#define MAX_STRIDE 1
#endif
#ifndef FIXED0_MULTIPLIER
#define FIXED0_MULTIPLIER 184.625f
#endif
#define LD(t,c) v_f16_ld_tnsr_b((int5){(c),row,0,0,0}, (t))
#define ST(t,c,y) v_bf16_st_tnsr((int5){(c),row,0,0,0}, (t), (y))
#define ACC(s,y) ((s)=v_bf16_mac_acc32_b((y),(bfloat128)1.0f,(s),0,1,0))

#ifdef F16_AFFINE
// FP16-affine variant (opt-in): 128-lane f16 sub/mul instead of 2x64-lane FP32
// mac; the 16256 BF16 exponent bias is folded into LUT128 slot15 (FID 0x8f,
// gen_lut --logq2). |d|<11 lands exactly on the 1/128 log2 grid; the f16
// constant 184.625 vs 184.665 is a 2.2e-4 relative temperature error.
// CPU model: sim_exp_f16_affine.py. q is clamped to >=-16128 so bits >=128.
static bfloat128 probability(half128 score, half128 mh, half128 ceiling) {
#ifdef F16_FIXED0
    // Gated by caller: |QK|<=80; exp/PV upper-bound log<86. No row max,
    // ceiling or underflow clamp is needed for these bounded inputs.
    half128 p = v_f16_mul_b(score, (half128)FIXED0_MULTIPLIER);
#else
    score = v_f16_min_b(score, ceiling);
    half128 p = v_f16_mul_b(v_f16_sub_b(score, mh), (half128)184.625f);
#endif
    short128 bits = v_convert_f16_to_i16_b(p, SW_RHNE);
#ifndef F16_FIXED0
    bits = v_i16_max_b(bits, (short128)-16128);
#endif
    ushort128 idx = *(ushort128 *)&bits & (ushort128)127;
#ifdef F16_INCOME
    // idx is dead after the lookup: reusing it as income lets the destination
    // share its register (no per-vector zeroing move). All lanes predicated on.
    bits += v_i16_lookup_1c(idx, 143, 1, *(short128 *)&idx, 1, 0);
#else
    bits += v_i16_lookup_1c(idx, 143, 1, (short128)0, 1, 0);
#endif
    return *(bfloat128 *)&bits;
}
#else
static bfloat128 probability(half128 score, float64 bias, half128 ceiling) {
    score = v_f16_min_b(score, ceiling);
    float128 t = v_convert_f16_to_f32_all_b(score);
    t.v1 = v_f32_mac_b(t.v1, 184.6649652337873f, bias);
    t.v2 = v_f32_mac_b(t.v2, 184.6649652337873f, bias);
    short128 bits = v_convert_f32_to_i16_all_b(t, SW_RHNE);
    bits = v_i16_max_b(bits, (short128)128);
    ushort128 idx = *(ushort128 *)&bits & (ushort128)127;
    bits += v_i16_lookup_1c(idx, 142, 1, (short128)0, 1, 0);
    return *(bfloat128 *)&bits;
}
#endif

#ifdef F16_AFFINE
// mx is a max of FP16 values, so the f32->f16 conversion is exact.
#define BIAS_DECL float128 mb;mb.v1=mx;mb.v2=mx;half128 bias=v_convert_f32_to_f16_all_b(mb);
#else
#define BIAS_DECL float64 bias=v_f32_mac_b(mx,-184.6649652337873f,(float64)16256.0f);
#endif

#if defined(F16_PIPE2) && defined(F16_AFFINE)
// Two-stage software pipeline over 512-column groups. Stage 1 (min/sub/mul/
// cvt/max/and) of group c+512 is independent of stage 2 (lookup/add/st/acc)
// of group c, so the lookups (load slot) fill VPU latency gaps. The next
// group's load is issued after stage 1 consumed x, so x needs no copy.
// Loads of columns >= full only feed values that are never stored/accumulated.
// Opt-in experiments (default off, default code unchanged):
//  F16_NOAND   : FAILED on device 2026-11-09 (the 128-entry lookup does NOT
//                ignore upper index bits; evidence/ablation-xknoand-g16.log). Not built.
//  F16_LKEARLY : issue the lookup at the end of stage 1, so its latency overlaps
//                the next group's work; stage 2 only adds/stores/accumulates.
#ifdef F16_NOAND
#define LKIDX(b) (*(ushort128 *)&(b))
#else
#define LKIDX(b) (*(ushort128 *)&(b) & (ushort128)127)
#endif
#ifdef F16_FIXED0
#define S1(x,b,i) do { \
    half128 _p = v_f16_mul_b((x),(half128)FIXED0_MULTIPLIER); \
    (b) = v_convert_f16_to_i16_b(_p,SW_RHNE); \
    (i) = LKIDX(b); } while (0)
#define S2(t,col,b,i,s) do { \
    short128 _r = v_i16_lookup_1c((i),143,1,*(short128 *)&(i),1,0); \
    short128 _y = (b) + _r; bfloat128 _yb = *(bfloat128 *)&_y; \
    ST(t,col,_yb); ACC(s,_yb); } while (0)
#elif defined(F16_LKEARLY)
#define S1(x,b,i) do { \
    half128 _p = v_f16_mul_b(v_f16_sub_b(v_f16_min_b((x),ceiling),bias),(half128)184.625f); \
    (b) = v_i16_max_b(v_convert_f16_to_i16_b(_p,SW_RHNE),(short128)-16128); \
    ushort128 _ix = LKIDX(b); \
    short128 _r = v_i16_lookup_1c(_ix,143,1,*(short128 *)&_ix,1,0); \
    (i) = *(ushort128 *)&_r; } while (0)
#define S2(t,col,b,i,s) do { \
    short128 _y = (b) + *(short128 *)&(i); bfloat128 _yb = *(bfloat128 *)&_y; \
    ST(t,col,_yb); ACC(s,_yb); } while (0)
#else
#define S1(x,b,i) do { \
    half128 _p = v_f16_mul_b(v_f16_sub_b(v_f16_min_b((x),ceiling),bias),(half128)184.625f); \
    (b) = v_i16_max_b(v_convert_f16_to_i16_b(_p,SW_RHNE),(short128)-16128); \
    (i) = LKIDX(b); } while (0)
#define S2(t,col,b,i,s) do { \
    short128 _r = v_i16_lookup_1c((i),143,1,*(short128 *)&(i),1,0); \
    short128 _y = (b) + _r; bfloat128 _yb = *(bfloat128 *)&_y; \
    ST(t,col,_yb); ACC(s,_yb); } while (0)
#endif
#ifdef F16_IDX
// Opt-in: persistent coordinate registers advanced by 512 columns per
// iteration, instead of rebuilding an int5 for every load/store.
#define STI(t,p,y) v_bf16_st_tnsr((p), (t), (y))
#define S2I(t,p,b,i,s) do { \
    short128 _r = v_i16_lookup_1c((i),143,1,*(short128 *)&(i),1,0); \
    short128 _y = (b) + _r; bfloat128 _yb = *(bfloat128 *)&_y; \
    STI(t,p,_yb); ACC(s,_yb); } while (0)
#define INNER4(t) \
        { \
        int5 p0={c,row,0,0,0},p1={c+128,row,0,0,0},p2={c+256,row,0,0,0},p3={c+384,row,0,0,0}; \
        int5 q0={c+1024,row,0,0,0},q1={c+1152,row,0,0,0},q2={c+1280,row,0,0,0},q3={c+1408,row,0,0,0}; \
        for (;c+1024<=full;c+=512) { \
            S2I(t,p0,b0,i0,s0);S2I(t,p1,b1,i1,s1);S2I(t,p2,b2,i2,s2);S2I(t,p3,b3,i3,s3); \
            S1(x0,b0,i0);S1(x1,b1,i1);S1(x2,b2,i2);S1(x3,b3,i3); \
            x0=v_f16_ld_tnsr_b(q0,(t));x1=v_f16_ld_tnsr_b(q1,(t));x2=v_f16_ld_tnsr_b(q2,(t));x3=v_f16_ld_tnsr_b(q3,(t)); \
            p0[0]+=512;p1[0]+=512;p2[0]+=512;p3[0]+=512;q0[0]+=512;q1[0]+=512;q2[0]+=512;q3[0]+=512; \
        } \
        }
#else
#define INNER4(t) \
        for (;c+1024<=full;c+=512) { \
            S2(t,c,b0,i0,s0);S2(t,c+128,b1,i1,s1);S2(t,c+256,b2,i2,s2);S2(t,c+384,b3,i3,s3); \
            S1(x0,b0,i0);S1(x1,b1,i1);S1(x2,b2,i2);S1(x3,b3,i3); \
            x0=LD(t,c+1024);x1=LD(t,c+1152);x2=LD(t,c+1280);x3=LD(t,c+1408); \
        }
#endif
#define MAIN_LOOP(t) MAIN4(t)
#define MAIN4(t) \
    if (c+512<=full) { \
        short128 b0,b1,b2,b3; ushort128 i0,i1,i2,i3; \
        half128 x0=LD(t,c),x1=LD(t,c+128),x2=LD(t,c+256),x3=LD(t,c+384); \
        S1(x0,b0,i0);S1(x1,b1,i1);S1(x2,b2,i2);S1(x3,b3,i3); \
        x0=LD(t,c+512);x1=LD(t,c+640);x2=LD(t,c+768);x3=LD(t,c+896); \
        INNER4(t) \
        S2(t,c,b0,i0,s0);S2(t,c+128,b1,i1,s1);S2(t,c+256,b2,i2,s2);S2(t,c+384,b3,i3,s3); \
        c+=512; \
    } \
    TAIL256(t)
#ifdef F16_TAIL256
// One two-vector step for a remaining 256..511-column tail (instead of two
// latency-bound single-vector iterations). Same per-element arithmetic.
#define TAIL256(t) \
    if (c+256<=full) { \
        short128 tb0,tb1; ushort128 ti0,ti1; \
        half128 tx0=LD(t,c),tx1=LD(t,c+128); \
        S1(tx0,tb0,ti0);S1(tx1,tb1,ti1); \
        S2(t,c,tb0,ti0,s0);S2(t,c+128,tb1,ti1,s1); \
        c+=256; \
    }
#else
#define TAIL256(t)
#endif
#elif defined(F16_PIPE)
// Software-pipelined: the next 512 columns are loaded while this group is
// computed. The final prefetch may touch columns >= full (still inside the
// tensor or padded by tensor bounds); those values are never used.
#define MAIN_LOOP(t) \
    if (c+512<=full) { \
        half128 x0=LD(t,0),x1=LD(t,128),x2=LD(t,256),x3=LD(t,384); \
        for (;c+512<=full;c+=512) { \
            half128 n0=LD(t,c+512),n1=LD(t,c+640),n2=LD(t,c+768),n3=LD(t,c+896); \
            bfloat128 y0=probability(x0,bias,ceiling),y1=probability(x1,bias,ceiling); \
            bfloat128 y2=probability(x2,bias,ceiling),y3=probability(x3,bias,ceiling); \
            ST(t,c,y0);ST(t,c+128,y1);ST(t,c+256,y2);ST(t,c+384,y3); \
            ACC(s0,y0);ACC(s1,y1);ACC(s2,y2);ACC(s3,y3); \
            x0=n0;x1=n1;x2=n2;x3=n3; \
        } \
    }
#else
#define MAIN_LOOP(t) \
    for (;c+512<=full;c+=512) { \
        half128 x0=LD(t,c),x1=LD(t,c+128),x2=LD(t,c+256),x3=LD(t,c+384); \
        bfloat128 y0=probability(x0,bias,ceiling),y1=probability(x1,bias,ceiling); \
        bfloat128 y2=probability(x2,bias,ceiling),y3=probability(x3,bias,ceiling); \
        ST(t,c,y0);ST(t,c+128,y1);ST(t,c+256,y2);ST(t,c+384,y3); \
        ACC(s0,y0);ACC(s1,y1);ACC(s2,y2);ACC(s3,y3); \
    }
#endif

#if defined(F16_FIXED0_G8) || defined(F16_FIXED0_G15) || defined(F16_FIXED0_G16)
// Fixed0 needs only mul/cvt/and before lookup, so an eight-chain body has
// much lower register pressure than the archived affine/max eight-chain code.
// All eight FP16 loads precede all stores; one coordinate per load/store pair.
#define MAIN8_FIXED(t) \
    { \
      int5 p0={c,row,0,0,0},p1={c+128,row,0,0,0},p2={c+256,row,0,0,0},p3={c+384,row,0,0,0}; \
      int5 p4={c+512,row,0,0,0},p5={c+640,row,0,0,0},p6={c+768,row,0,0,0},p7={c+896,row,0,0,0}; \
      for (;c+1024<=full;c+=1024) { \
        half128 x0=v_f16_ld_tnsr_b(p0,t),x1=v_f16_ld_tnsr_b(p1,t),x2=v_f16_ld_tnsr_b(p2,t),x3=v_f16_ld_tnsr_b(p3,t); \
        half128 x4=v_f16_ld_tnsr_b(p4,t),x5=v_f16_ld_tnsr_b(p5,t),x6=v_f16_ld_tnsr_b(p6,t),x7=v_f16_ld_tnsr_b(p7,t); \
        short128 b0,b1,b2,b3,b4,b5,b6,b7;ushort128 i0,i1,i2,i3,i4,i5,i6,i7; \
        S1(x0,b0,i0);S1(x1,b1,i1);S1(x2,b2,i2);S1(x3,b3,i3); \
        S1(x4,b4,i4);S1(x5,b5,i5);S1(x6,b6,i6);S1(x7,b7,i7); \
        S2I(t,p0,b0,i0,s0);S2I(t,p1,b1,i1,s1);S2I(t,p2,b2,i2,s2);S2I(t,p3,b3,i3,s3); \
        S2I(t,p4,b4,i4,s0);S2I(t,p5,b5,i5,s1);S2I(t,p6,b6,i6,s2);S2I(t,p7,b7,i7,s3); \
        p0[0]+=1024;p1[0]+=1024;p2[0]+=1024;p3[0]+=1024;p4[0]+=1024;p5[0]+=1024;p6[0]+=1024;p7[0]+=1024; \
      } \
    }
#undef MAIN_LOOP
#ifdef F16_FIXED0_PAD
#define MAIN_LOOP(t) MAIN8_FIXED(t)
#else
#define MAIN_LOOP(t) MAIN8_FIXED(t) MAIN4(t)
#endif
#endif

#ifdef F16_FIXED0_G8PIPE
#undef MAIN8_FIXED
#define MAIN8_FIXED(t) \
    if(c+1024<=full) { \
    short128 b0,b1,b2,b3,b4,b5,b6,b7; ushort128 i0,i1,i2,i3,i4,i5,i6,i7; \
    half128 x0=LD(t,c+0),x1=LD(t,c+128),x2=LD(t,c+256),x3=LD(t,c+384),x4=LD(t,c+512),x5=LD(t,c+640),x6=LD(t,c+768),x7=LD(t,c+896); \
    S1(x0,b0,i0);S1(x1,b1,i1);S1(x2,b2,i2);S1(x3,b3,i3);S1(x4,b4,i4);S1(x5,b5,i5);S1(x6,b6,i6);S1(x7,b7,i7); \
    x0=LD(t,c+1024);x1=LD(t,c+1152);x2=LD(t,c+1280);x3=LD(t,c+1408);x4=LD(t,c+1536);x5=LD(t,c+1664);x6=LD(t,c+1792);x7=LD(t,c+1920); \
    { \
    int5 p0={c+0,row,0,0,0},p1={c+128,row,0,0,0},p2={c+256,row,0,0,0},p3={c+384,row,0,0,0},p4={c+512,row,0,0,0},p5={c+640,row,0,0,0},p6={c+768,row,0,0,0},p7={c+896,row,0,0,0}; \
    int5 q0={c+2048,row,0,0,0},q1={c+2176,row,0,0,0},q2={c+2304,row,0,0,0},q3={c+2432,row,0,0,0},q4={c+2560,row,0,0,0},q5={c+2688,row,0,0,0},q6={c+2816,row,0,0,0},q7={c+2944,row,0,0,0}; \
    for(;c+2048<=full;c+=1024) { \
    S2I(t,p0,b0,i0,s0);S2I(t,p1,b1,i1,s1);S2I(t,p2,b2,i2,s2);S2I(t,p3,b3,i3,s3);S2I(t,p4,b4,i4,s0);S2I(t,p5,b5,i5,s1);S2I(t,p6,b6,i6,s2);S2I(t,p7,b7,i7,s3); \
    S1(x0,b0,i0);S1(x1,b1,i1);S1(x2,b2,i2);S1(x3,b3,i3);S1(x4,b4,i4);S1(x5,b5,i5);S1(x6,b6,i6);S1(x7,b7,i7); \
    x0=v_f16_ld_tnsr_b(q0,t);x1=v_f16_ld_tnsr_b(q1,t);x2=v_f16_ld_tnsr_b(q2,t);x3=v_f16_ld_tnsr_b(q3,t);x4=v_f16_ld_tnsr_b(q4,t);x5=v_f16_ld_tnsr_b(q5,t);x6=v_f16_ld_tnsr_b(q6,t);x7=v_f16_ld_tnsr_b(q7,t); \
    p0[0]+=1024;q0[0]+=1024;p1[0]+=1024;q1[0]+=1024;p2[0]+=1024;q2[0]+=1024;p3[0]+=1024;q3[0]+=1024;p4[0]+=1024;q4[0]+=1024;p5[0]+=1024;q5[0]+=1024;p6[0]+=1024;q6[0]+=1024;p7[0]+=1024;q7[0]+=1024; \
    } \
    } \
    S2(t,c+0,b0,i0,s0);S2(t,c+128,b1,i1,s1);S2(t,c+256,b2,i2,s2);S2(t,c+384,b3,i3,s3);S2(t,c+512,b4,i4,s0);S2(t,c+640,b5,i5,s1);S2(t,c+768,b6,i6,s2);S2(t,c+896,b7,i7,s3); \
    c+=1024; \
    }
#endif

#ifdef F16_FIXED0_G16
#undef MAIN8_FIXED
#define MAIN8_FIXED(t) \
    { \
    int5 p0={c+0,row,0,0,0},p1={c+128,row,0,0,0},p2={c+256,row,0,0,0},p3={c+384,row,0,0,0},p4={c+512,row,0,0,0},p5={c+640,row,0,0,0},p6={c+768,row,0,0,0},p7={c+896,row,0,0,0},p8={c+1024,row,0,0,0},p9={c+1152,row,0,0,0},p10={c+1280,row,0,0,0},p11={c+1408,row,0,0,0},p12={c+1536,row,0,0,0},p13={c+1664,row,0,0,0},p14={c+1792,row,0,0,0},p15={c+1920,row,0,0,0}; \
    for(;c+2048<=full;c+=2048) { \
    half128 x0=v_f16_ld_tnsr_b(p0,t),x1=v_f16_ld_tnsr_b(p1,t),x2=v_f16_ld_tnsr_b(p2,t),x3=v_f16_ld_tnsr_b(p3,t),x4=v_f16_ld_tnsr_b(p4,t),x5=v_f16_ld_tnsr_b(p5,t),x6=v_f16_ld_tnsr_b(p6,t),x7=v_f16_ld_tnsr_b(p7,t),x8=v_f16_ld_tnsr_b(p8,t),x9=v_f16_ld_tnsr_b(p9,t),x10=v_f16_ld_tnsr_b(p10,t),x11=v_f16_ld_tnsr_b(p11,t),x12=v_f16_ld_tnsr_b(p12,t),x13=v_f16_ld_tnsr_b(p13,t),x14=v_f16_ld_tnsr_b(p14,t),x15=v_f16_ld_tnsr_b(p15,t); \
    short128 b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15;ushort128 i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15; \
    S1(x0,b0,i0);S1(x1,b1,i1);S1(x2,b2,i2);S1(x3,b3,i3);S1(x4,b4,i4);S1(x5,b5,i5);S1(x6,b6,i6);S1(x7,b7,i7);S1(x8,b8,i8);S1(x9,b9,i9);S1(x10,b10,i10);S1(x11,b11,i11);S1(x12,b12,i12);S1(x13,b13,i13);S1(x14,b14,i14);S1(x15,b15,i15); \
    S2I(t,p0,b0,i0,s0);S2I(t,p1,b1,i1,s1);S2I(t,p2,b2,i2,s0);S2I(t,p3,b3,i3,s1);S2I(t,p4,b4,i4,s0);S2I(t,p5,b5,i5,s1);S2I(t,p6,b6,i6,s0);S2I(t,p7,b7,i7,s1);S2I(t,p8,b8,i8,s0);S2I(t,p9,b9,i9,s1);S2I(t,p10,b10,i10,s0);S2I(t,p11,b11,i11,s1);S2I(t,p12,b12,i12,s0);S2I(t,p13,b13,i13,s1);S2I(t,p14,b14,i14,s0);S2I(t,p15,b15,i15,s1); \
    p0[0]+=2048;p1[0]+=2048;p2[0]+=2048;p3[0]+=2048;p4[0]+=2048;p5[0]+=2048;p6[0]+=2048;p7[0]+=2048;p8[0]+=2048;p9[0]+=2048;p10[0]+=2048;p11[0]+=2048;p12[0]+=2048;p13[0]+=2048;p14[0]+=2048;p15[0]+=2048; \
    } \
    }
#endif

#ifdef F16_FIXED0_G15
#undef MAIN8_FIXED
#define MAIN8_FIXED(t) \
    { \
    int5 p0={c+0,row,0,0,0},p1={c+128,row,0,0,0},p2={c+256,row,0,0,0},p3={c+384,row,0,0,0},p4={c+512,row,0,0,0},p5={c+640,row,0,0,0},p6={c+768,row,0,0,0},p7={c+896,row,0,0,0},p8={c+1024,row,0,0,0},p9={c+1152,row,0,0,0},p10={c+1280,row,0,0,0},p11={c+1408,row,0,0,0},p12={c+1536,row,0,0,0},p13={c+1664,row,0,0,0},p14={c+1792,row,0,0,0}; \
    for(;c+1920<=full;c+=1920) { \
    half128 x0=v_f16_ld_tnsr_b(p0,t),x1=v_f16_ld_tnsr_b(p1,t),x2=v_f16_ld_tnsr_b(p2,t),x3=v_f16_ld_tnsr_b(p3,t),x4=v_f16_ld_tnsr_b(p4,t),x5=v_f16_ld_tnsr_b(p5,t),x6=v_f16_ld_tnsr_b(p6,t),x7=v_f16_ld_tnsr_b(p7,t),x8=v_f16_ld_tnsr_b(p8,t),x9=v_f16_ld_tnsr_b(p9,t),x10=v_f16_ld_tnsr_b(p10,t),x11=v_f16_ld_tnsr_b(p11,t),x12=v_f16_ld_tnsr_b(p12,t),x13=v_f16_ld_tnsr_b(p13,t),x14=v_f16_ld_tnsr_b(p14,t); \
    short128 b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14;ushort128 i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14; \
    S1(x0,b0,i0);S1(x1,b1,i1);S1(x2,b2,i2);S1(x3,b3,i3);S1(x4,b4,i4);S1(x5,b5,i5);S1(x6,b6,i6);S1(x7,b7,i7);S1(x8,b8,i8);S1(x9,b9,i9);S1(x10,b10,i10);S1(x11,b11,i11);S1(x12,b12,i12);S1(x13,b13,i13);S1(x14,b14,i14); \
    S2I(t,p0,b0,i0,s0);S2I(t,p1,b1,i1,s1);S2I(t,p2,b2,i2,s0);S2I(t,p3,b3,i3,s1);S2I(t,p4,b4,i4,s0);S2I(t,p5,b5,i5,s1);S2I(t,p6,b6,i6,s0);S2I(t,p7,b7,i7,s1);S2I(t,p8,b8,i8,s0);S2I(t,p9,b9,i9,s1);S2I(t,p10,b10,i10,s0);S2I(t,p11,b11,i11,s1);S2I(t,p12,b12,i12,s0);S2I(t,p13,b13,i13,s1);S2I(t,p14,b14,i14,s0); \
    p0[0]+=1920;p1[0]+=1920;p2[0]+=1920;p3[0]+=1920;p4[0]+=1920;p5[0]+=1920;p6[0]+=1920;p7[0]+=1920;p8[0]+=1920;p9[0]+=1920;p10[0]+=1920;p11[0]+=1920;p12[0]+=1920;p13[0]+=1920;p14[0]+=1920; \
    } \
    }
#endif

// FAILED full-head RMS gate (2026-11-09); not built or exposed by CLI.
#ifdef F16_FIXED0_G19
#undef MAIN8_FIXED
#define MAIN8_FIXED(t) \
    { \
    int5 p0={c+0,row,0,0,0},p1={c+128,row,0,0,0},p2={c+256,row,0,0,0},p3={c+384,row,0,0,0},p4={c+512,row,0,0,0},p5={c+640,row,0,0,0},p6={c+768,row,0,0,0},p7={c+896,row,0,0,0},p8={c+1024,row,0,0,0},p9={c+1152,row,0,0,0},p10={c+1280,row,0,0,0},p11={c+1408,row,0,0,0},p12={c+1536,row,0,0,0},p13={c+1664,row,0,0,0},p14={c+1792,row,0,0,0},p15={c+1920,row,0,0,0},p16={c+2048,row,0,0,0},p17={c+2176,row,0,0,0},p18={c+2304,row,0,0,0}; \
    for(;c+2432<=full;c+=2432) { \
    half128 x0=v_f16_ld_tnsr_b(p0,t),x1=v_f16_ld_tnsr_b(p1,t),x2=v_f16_ld_tnsr_b(p2,t),x3=v_f16_ld_tnsr_b(p3,t),x4=v_f16_ld_tnsr_b(p4,t),x5=v_f16_ld_tnsr_b(p5,t),x6=v_f16_ld_tnsr_b(p6,t),x7=v_f16_ld_tnsr_b(p7,t),x8=v_f16_ld_tnsr_b(p8,t),x9=v_f16_ld_tnsr_b(p9,t),x10=v_f16_ld_tnsr_b(p10,t),x11=v_f16_ld_tnsr_b(p11,t),x12=v_f16_ld_tnsr_b(p12,t),x13=v_f16_ld_tnsr_b(p13,t),x14=v_f16_ld_tnsr_b(p14,t),x15=v_f16_ld_tnsr_b(p15,t),x16=v_f16_ld_tnsr_b(p16,t),x17=v_f16_ld_tnsr_b(p17,t),x18=v_f16_ld_tnsr_b(p18,t); \
    short128 b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18;ushort128 i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18; \
    S1(x0,b0,i0);S1(x1,b1,i1);S1(x2,b2,i2);S1(x3,b3,i3);S1(x4,b4,i4);S1(x5,b5,i5);S1(x6,b6,i6);S1(x7,b7,i7);S1(x8,b8,i8);S1(x9,b9,i9);S1(x10,b10,i10);S1(x11,b11,i11);S1(x12,b12,i12);S1(x13,b13,i13);S1(x14,b14,i14);S1(x15,b15,i15);S1(x16,b16,i16);S1(x17,b17,i17);S1(x18,b18,i18); \
    S2I(t,p0,b0,i0,s0);S2I(t,p1,b1,i1,s0);S2I(t,p2,b2,i2,s0);S2I(t,p3,b3,i3,s0);S2I(t,p4,b4,i4,s0);S2I(t,p5,b5,i5,s0);S2I(t,p6,b6,i6,s0);S2I(t,p7,b7,i7,s0);S2I(t,p8,b8,i8,s0);S2I(t,p9,b9,i9,s0);S2I(t,p10,b10,i10,s0);S2I(t,p11,b11,i11,s0);S2I(t,p12,b12,i12,s0);S2I(t,p13,b13,i13,s0);S2I(t,p14,b14,i14,s0);S2I(t,p15,b15,i15,s0);S2I(t,p16,b16,i16,s0);S2I(t,p17,b17,i17,s0);S2I(t,p18,b18,i18,s0); \
    p0[0]+=2432;p1[0]+=2432;p2[0]+=2432;p3[0]+=2432;p4[0]+=2432;p5[0]+=2432;p6[0]+=2432;p7[0]+=2432;p8[0]+=2432;p9[0]+=2432;p10[0]+=2432;p11[0]+=2432;p12[0]+=2432;p13[0]+=2432;p14[0]+=2432;p15[0]+=2432;p16[0]+=2432;p17[0]+=2432;p18[0]+=2432; \
    } \
    }
#endif

#ifdef PROFILE_NO_MAIN
// Timing-only (outputs invalid): skip all per-element exp work of a unit.
#define PROFILE_SKIP(n) c=(n);
#else
#define PROFILE_SKIP(n)
#endif
#define UNIT(t,cols,lim) do { \
    int full=(lim)/128*128; \
    half128 m0=low,m1=low,m2=low,m3=low; \
    int c=0; \
    for (; c+384*MAX_STRIDE<full; c+=512*MAX_STRIDE) { \
        m0=v_f16_max_b(m0,LD(t,c)); m1=v_f16_max_b(m1,LD(t,c+128*MAX_STRIDE)); \
        m2=v_f16_max_b(m2,LD(t,c+256*MAX_STRIDE)); m3=v_f16_max_b(m3,LD(t,c+384*MAX_STRIDE)); \
    } \
    for (;c<full;c+=128*MAX_STRIDE) m0=v_f16_max_b(m0,LD(t,c)); \
    if (MAX_STRIDE==1 || !full) \
        for (;c<(cols);c+=128) m0=v_f16_max_b(m0,v_f16_sel_less_u16_b(read_lane_id_2b_b()+c,(unsigned short)(lim),LD(t,c),low)); \
    m0=v_f16_max_b(v_f16_max_b(m0,m1),v_f16_max_b(m2,m3)); \
    float128 fm=v_convert_f16_to_f32_all_b(m0); \
    mx=max64_no_lut(v_f32_max_b(fm.v1,fm.v2)); \
    BIAS_DECL \
    float128 cm;cm.v1=mx+64.0f;cm.v2=cm.v1; \
    half128 ceiling=v_convert_f32_to_f16_all_b(cm); \
    float128 s0={0},s1={0},s2={0},s3={0}; \
    c=0; \
    PROFILE_SKIP(cols) \
    MAIN_LOOP(t) \
    for (;c<full;c+=128) { bfloat128 y=probability(LD(t,c),bias,ceiling);ST(t,c,y);ACC(s0,y); } \
    for (;c<(cols);c+=128) { \
        bfloat128 y=probability(LD(t,c),bias,ceiling); \
        y=v_bf16_sel_less_u16_b(read_lane_id_2b_b()+c,(unsigned short)(lim),y,(bfloat128)0); \
        ST(t,c,y);ACC(s0,y); \
    } \
    lsum=((s0.v1+s0.v2)+(s1.v1+s1.v2))+((s2.v1+s2.v2)+(s3.v1+s3.v2)); \
} while(0)

#if defined(F16_SKEW) && defined(SINGLE_TILE)
// Skewed rows (opt-in, single-tile kernels): the sampled-max loads of row u+1
// follow row u's exp loop, then row u's row-sum reduction and row u+1's lane
// max reduction run back to back as independent chains. One copy of the main
// loop (no code growth); per-row arithmetic identical to UNIT.
#define MAXVEC(t,cols,lim,mv) do { \
    int full=(lim)/128*128; \
    half128 m0=low,m1=low,m2=low,m3=low; \
    int c=0; \
    for (; c+384*MAX_STRIDE<full; c+=512*MAX_STRIDE) { \
        m0=v_f16_max_b(m0,LD(t,c)); m1=v_f16_max_b(m1,LD(t,c+128*MAX_STRIDE)); \
        m2=v_f16_max_b(m2,LD(t,c+256*MAX_STRIDE)); m3=v_f16_max_b(m3,LD(t,c+384*MAX_STRIDE)); \
    } \
    for (;c<full;c+=128*MAX_STRIDE) m0=v_f16_max_b(m0,LD(t,c)); \
    if (MAX_STRIDE==1 || !full) \
        for (;c<(cols);c+=128) m0=v_f16_max_b(m0,v_f16_sel_less_u16_b(read_lane_id_2b_b()+c,(unsigned short)(lim),LD(t,c),low)); \
    (mv)=v_f16_max_b(v_f16_max_b(m0,m1),v_f16_max_b(m2,m3)); \
} while(0)
#define LANEMAX(h) ({ float128 _f=v_convert_f16_to_f32_all_b(h); max64_no_lut(v_f32_max_b(_f.v1,_f.v2)); })
#ifdef F16_FIXED0_PAD
// Tensor bounds suppress stores and return an FP16 sentinel whose fixed0
// affine/LUT mapping is exactly BF16 zero (host/runtime checked).
#if defined(F16_FIXED0_G15)
#define EXP_FULL(lim) (((lim)+1919)/1920*1920)
#elif defined(F16_FIXED0_G16)
#define EXP_FULL(lim) (((lim)+2047)/2048*2048)
#else
#define EXP_FULL(lim) (((lim)+1023)/1024*1024)
#endif
#else
#define EXP_FULL(lim) ((lim)/128*128)
#endif
#define EXPPART(t,cols,lim) do { \
    int full=EXP_FULL(lim); \
    int c=0; \
    BIAS_DECL \
    float128 cm;cm.v1=mx+64.0f;cm.v2=cm.v1; \
    half128 ceiling=v_convert_f32_to_f16_all_b(cm); \
    float128 s0={0},s1={0},s2={0},s3={0}; \
    PROFILE_SKIP(cols) \
    MAIN_LOOP(t) \
    for (;c<full;c+=128) { bfloat128 y=probability(LD(t,c),bias,ceiling);ST(t,c,y);ACC(s0,y); } \
    for (;c<(cols);c+=128) { \
        bfloat128 y=probability(LD(t,c),bias,ceiling); \
        y=v_bf16_sel_less_u16_b(read_lane_id_2b_b()+c,(unsigned short)(lim),y,(bfloat128)0); \
        ST(t,c,y);ACC(s0,y); \
    } \
    lsum=((s0.v1+s0.v2)+(s1.v1+s1.v2))+((s2.v1+s2.v2)+(s3.v1+s3.v2)); \
} while(0)
#endif

void main(tensor stats, tensor sa, tensor sb, int valid_cols) {
    int5 start=get_index_space_offset(),end=start+get_index_space_size();
    int ca=get_dim_size(sa,0);
    int la=valid_cols<ca?valid_cols:ca;
#ifndef SINGLE_TILE
    int cb=get_dim_size(sb,0),lb=valid_cols>ca?valid_cols-ca:0;
#endif
    const half128 low=(half128)(-65504.0f);
#if defined(F16_FIXED0) && defined(SINGLE_TILE)
    for (int u=start[1];u<end[1];++u) {
        float64 mx=(float64)0.0f,lsum;
        { int row=u; EXPPART(sa,ca,la); }
        float64 sum=sum64_no_lut(lsum);
        float64 out=v_f32_sel_eq_u32_b(read_lane_id_4b_b(),0,(float64)0.0f,sum);
        v_f32_st_tnsr((int5){0,u,0,0,0},stats,out);
    }
    return;
#endif
#if defined(F16_SKEW) && defined(SINGLE_TILE)
    if (start[1] < end[1]) {
        half128 mv;
        { int row=start[1]; MAXVEC(sa,ca,la,mv); }
        float64 mnext = LANEMAX(mv);
        for (int u=start[1];u<end[1];++u) {
            float64 mx = mnext, lsum;
            { int row=u; EXPPART(sa,ca,la); }
            mv = low;
            if (u + 1 < end[1]) { int row=u+1; MAXVEC(sa,ca,la,mv); }
            float64 sum=sum64_no_lut(lsum);
            mnext = LANEMAX(mv);
            float64 out=v_f32_sel_eq_u32_b(read_lane_id_4b_b(),0,mx,sum);
            v_f32_st_tnsr((int5){0,u,0,0,0},stats,out);
        }
    }
    return;
#endif
    for (int u=start[1];u<end[1];++u) {
        #ifdef SINGLE_TILE
        int row=u;
#else
        int row=u&127;
#endif
        float64 mx,lsum;
        #ifdef SINGLE_TILE
        UNIT(sa,ca,la);
#else
        if (u<128) UNIT(sa,ca,la); else UNIT(sb,cb,lb);
#endif
        float64 sum=sum64_no_lut(lsum);
        float64 out=v_f32_sel_eq_u32_b(read_lane_id_4b_b(),0,mx,sum);
        v_f32_st_tnsr((int5){0,u,0,0,0},stats,out);
    }
}
