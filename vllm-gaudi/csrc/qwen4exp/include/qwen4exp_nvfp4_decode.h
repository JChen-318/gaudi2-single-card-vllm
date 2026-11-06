// SPDX-License-Identifier: Apache-2.0
// Shared NVFP4 nibble decode. Exact integer bit construction of e2m1 * e4m3
// (at most 5 significant mantissa bits, so the product is representable in
// BF16 without rounding). The per-expert scale2 is not applied here.
//
//   value = fixed/32 * 2^(efield-127)
//   normal scale    (e >= 1): fixed0 = 4*(8+m)*(1 or 1.5), efield0 = e + (mag>>1) + 119
//   subnormal scale (e == 0): fixed0 = 4*m*(1 or 1.5),     efield0 = (mag>>1) + 120
//   odd mag == 1 forces the 1.5x factor off. The scale sign XORs into the
//   result sign. mag == 0 or a zero scale emit zeros.

static inline ushort128 nvfp4_mul_bits(ushort128 nibble, ushort128 scale)
{
    const ushort128 mag = v_u16_and_b(nibble, 7);
    const ushort128 odd = v_u16_and_b(nibble, 1);
    const ushort128 sign = v_u16_xor_b(
        v_u16_shl_b(v_u16_and_b(nibble, 8), 12),
        v_u16_shl_b(v_u16_and_b(scale, 128), 8));

    const ushort128 e = v_u16_and_b(v_u16_shr_b(scale, 3), 15);
    const ushort128 m = v_u16_and_b(scale, 7);

    const ushort128 base_n = v_u16_shl_b(v_u16_add_b(m, 8), 2);
    const ushort128 base_s = v_u16_shl_b(m, 2);
    const ushort128 base = v_u16_sel_eq_u16_b(e, 0, base_s, base_n);
    const ushort128 extra = v_u16_sel_eq_u16_b(
        odd, 1, v_u16_sel_eq_u16_b(mag, 1, 0, v_u16_shr_b(base, 1)), 0);
    ushort128 fixed0 = v_u16_add_b(base, extra);

    ushort128 efield = v_u16_add_b(e, v_u16_add_b(v_u16_shr_b(mag, 1), 119));
    efield = v_u16_add_b(efield, v_u16_sel_eq_u16_b(e, 0, 1, 0));

    const ushort128 over = v_u16_sel_geq_u16_b(fixed0, 64, 1, 0);
    fixed0 = v_u16_sel_geq_u16_b(fixed0, 64, v_u16_shr_b(fixed0, 1), fixed0);
    efield = v_u16_add_b(efield, over);

    const ushort128 under1 = v_u16_sel_less_u16_b(fixed0, 32, 1, 0);
    fixed0 = v_u16_sel_less_u16_b(fixed0, 32, v_u16_shl_b(fixed0, 1), fixed0);
    efield = v_u16_sub_b(efield, under1);
    const ushort128 under2 = v_u16_sel_less_u16_b(fixed0, 32, 1, 0);
    fixed0 = v_u16_sel_less_u16_b(fixed0, 32, v_u16_shl_b(fixed0, 1), fixed0);
    efield = v_u16_sub_b(efield, under2);
    const ushort128 under3 = v_u16_sel_less_u16_b(fixed0, 32, 1, 0);
    fixed0 = v_u16_sel_less_u16_b(fixed0, 32, v_u16_shl_b(fixed0, 1), fixed0);
    efield = v_u16_sub_b(efield, under3);

    ushort128 bits = v_u16_shl_b(v_u16_sub_b(fixed0, 32), 2);
    bits = v_u16_or_b(bits, v_u16_shl_b(efield, 7));
    bits = v_u16_sel_eq_u16_b(mag, 0, 0, bits);
    bits = v_u16_sel_eq_u16_b(v_u16_and_b(v_u16_sel_eq_u16_b(m, 0, 1, 0),
                                          v_u16_sel_eq_u16_b(e, 0, 1, 0)),
                              1, 0, bits);
    bits = v_u16_or_b(bits, sign);
    return bits;
}
