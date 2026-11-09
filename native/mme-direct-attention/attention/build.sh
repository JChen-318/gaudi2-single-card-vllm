#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
T=${HLTHUNK_SOURCE:-/opt/habanalabs/src/hl-thunk}
L=${HLTHUNK_LIB:-/usr/lib/habanalabs}
mkdir -p "$HERE/build"
for name in relu exp_poly exp_sum normalize normalize2 normalize3 normalize_batch normalize_quad normalize_quad_fixed noop4 noop_exp3; do
    tpc-clang -O2 -Wall -Werror -march=gaudi2 "$HERE/kernels/$name.c" -c -o "$HERE/build/$name.o"
    tpc-clang -O2 -march=gaudi2 -S "$HERE/kernels/$name.c" -o "$HERE/build/$name.s"
    # Standalone launchers do not install coefficient LUTs. Reject implicit use.
    if grep -E '\blookup(_|\.)|\bld_g\b|\bst_g\b' "$HERE/build/$name.s"; then
        echo "Unexpected LUT/global access in $name" >&2; exit 1
    fi
    python3 "$HERE/extract_text.py" "$HERE/build/$name.o" "$HERE/build/$name.bin"
done
# Interleaved exp variants (EXP_GROUP independent 128-column chunks per iteration).
# exp_sum_g4b3/b4: base-2 evaluation with a degree-3/4 polynomial.
for variant in g2:-DEXP_GROUP=2 g4:-DEXP_GROUP=4 g4b3:"-DEXP_GROUP=4 -DEXP_BASE2=3" g4b4:"-DEXP_GROUP=4 -DEXP_BASE2=4"; do
    name=exp_sum_${variant%%:*}
    read -r -a defs <<< "${variant#*:}"
    tpc-clang -O2 -Wall -Werror -march=gaudi2 "${defs[@]}" "$HERE/kernels/exp_sum.c" -c -o "$HERE/build/$name.o"
    tpc-clang -O2 -march=gaudi2 "${defs[@]}" -S "$HERE/kernels/exp_sum.c" -o "$HERE/build/$name.s"
    if grep -E '\blookup(_|\.)|\bld_g\b|\bst_g\b' "$HERE/build/$name.s"; then
        echo "Unexpected LUT/global access in $name" >&2; exit 1
    fi
    python3 "$HERE/extract_text.py" "$HERE/build/$name.o" "$HERE/build/$name.bin"
done
# LUT-based exp (uses lookup_1c into the Gaudi2 special-function tables).
tpc-clang -O2 -Wall -Werror -march=gaudi2 "$HERE/kernels/exp_lut.c" -c -o "$HERE/build/exp_lut.o"
tpc-clang -O2 -march=gaudi2 -S "$HERE/kernels/exp_lut.c" -o "$HERE/build/exp_lut.s"
if grep -E '\bld_g\b|\bst_g\b' "$HERE/build/exp_lut.s"; then echo "Unexpected global access in exp_lut" >&2; exit 1; fi
python3 "$HERE/extract_text.py" "$HERE/build/exp_lut.o" "$HERE/build/exp_lut.bin"
tpc-clang -O2 -Wall -Werror -march=gaudi2 -DLUT_FAST "$HERE/kernels/exp_lut.c" -c -o "$HERE/build/exp_lut2.o"
tpc-clang -O2 -march=gaudi2 -DLUT_FAST -S "$HERE/kernels/exp_lut.c" -o "$HERE/build/exp_lut2.s"
if grep -E '\bld_g\b|\bst_g\b' "$HERE/build/exp_lut2.s"; then echo "Unexpected global access in exp_lut2" >&2; exit 1; fi
python3 "$HERE/extract_text.py" "$HERE/build/exp_lut2.o" "$HERE/build/exp_lut2.bin"
tpc-clang -O2 -Wall -Werror -march=gaudi2 "$HERE/kernels/exp_lut_split.c" -c -o "$HERE/build/exp_lut_split.o"
tpc-clang -O2 -march=gaudi2 -S "$HERE/kernels/exp_lut_split.c" -o "$HERE/build/exp_lut_split.s"
if grep -E '\bld_g\b|\bst_g\b' "$HERE/build/exp_lut_split.s"; then echo "Unexpected global access in exp_lut_split" >&2; exit 1; fi
python3 "$HERE/extract_text.py" "$HERE/build/exp_lut_split.o" "$HERE/build/exp_lut_split.bin"
tpc-clang -O2 -Wall -Werror -march=gaudi2 "$HERE/kernels/exp_lut_half.c" -c -o "$HERE/build/exp_lut_half.o"
tpc-clang -O2 -march=gaudi2 -S "$HERE/kernels/exp_lut_half.c" -o "$HERE/build/exp_lut_half.s"
if grep -E '\bld_g\b|\bst_g\b' "$HERE/build/exp_lut_half.s"; then echo "Unexpected global access in exp_lut_half" >&2; exit 1; fi
python3 "$HERE/extract_text.py" "$HERE/build/exp_lut_half.o" "$HERE/build/exp_lut_half.bin"
tpc-clang -O2 -Wall -Werror -march=gaudi2 -DEXP_MANUAL "$HERE/kernels/exp_lut_half.c" -c -o "$HERE/build/exp_lut_half_m.o"
tpc-clang -O2 -march=gaudi2 -DEXP_MANUAL -S "$HERE/kernels/exp_lut_half.c" -o "$HERE/build/exp_lut_half_m.s"
python3 "$HERE/extract_text.py" "$HERE/build/exp_lut_half_m.o" "$HERE/build/exp_lut_half_m.bin"
tpc-clang -O2 -Wall -Werror -march=gaudi2 -DEXP_MANUAL -DEXP_MAX_STRIDE=8 "$HERE/kernels/exp_lut_half.c" -c -o "$HERE/build/exp_lut_half_ms8.o"
tpc-clang -O2 -march=gaudi2 -DEXP_MANUAL -DEXP_MAX_STRIDE=8 -S "$HERE/kernels/exp_lut_half.c" -o "$HERE/build/exp_lut_half_ms8.s"
python3 "$HERE/extract_text.py" "$HERE/build/exp_lut_half_ms8.o" "$HERE/build/exp_lut_half_ms8.bin"

for variant in logq:-DEXP_LOGQ logq_ms8:"-DEXP_LOGQ -DEXP_MAX_STRIDE=8" logq_b80_ms8:"-DEXP_LOGQ -DEXP_MAX_STRIDE=8 -DEXP_LOGQ_BOUND80"; do
    name=exp_lut_half_${variant%%:*}
    read -r -a defs <<< "${variant#*:}"
    tpc-clang -O2 -Wall -Werror -march=gaudi2 "${defs[@]}" "$HERE/kernels/exp_lut_half.c" -c -o "$HERE/build/$name.o"
    tpc-clang -O2 -march=gaudi2 "${defs[@]}" -S "$HERE/kernels/exp_lut_half.c" -o "$HERE/build/$name.s"
    if grep -E '\bld_g\b|\bst_g\b' "$HERE/build/$name.s"; then echo "Unexpected global access in $name" >&2; exit 1; fi
    python3 "$HERE/extract_text.py" "$HERE/build/$name.o" "$HERE/build/$name.bin"
done
for variant in exp_f16_logq:1 exp_f16_logq_ms8:8; do
    name=${variant%%:*}
    tpc-clang -O2 -Wall -Werror -march=gaudi2 -DMAX_STRIDE=${variant#*:} "$HERE/kernels/exp_f16_logq.c" -c -o "$HERE/build/$name.o"
    tpc-clang -O2 -march=gaudi2 -DMAX_STRIDE=${variant#*:} -S "$HERE/kernels/exp_f16_logq.c" -o "$HERE/build/$name.s"
    if grep -E '\bld_g\b|\bst_g\b' "$HERE/build/$name.s"; then echo "Unexpected global access in $name" >&2; exit 1; fi
    python3 "$HERE/extract_text.py" "$HERE/build/$name.o" "$HERE/build/$name.bin"
done

# Opt-in FP16-affine exp (LUT128 slot15 / lut_gen6_logq2.bin).
# The archived max/affine G8 failed. Fixed0 G8/15/16 are separately validated;
# G19 failed the full-head gate, and G8PIPE was slower, so neither is built.
for variant in exp_f16_tile_fixed0_pow2q_g15pad:"-DFIXED0_MULTIPLIER=261.25f -DMAX_STRIDE=8 -DF16_AFFINE -DF16_PIPE2 -DF16_TAIL256 -DSINGLE_TILE -DF16_SKEW -DF16_IDX -DF16_FIXED0 -DF16_FIXED0_G8 -DF16_FIXED0_G15 -DF16_FIXED0_PAD" \
               exp_f16_tile_fixed0_g15pad:"-DMAX_STRIDE=8 -DF16_AFFINE -DF16_PIPE2 -DF16_TAIL256 -DSINGLE_TILE -DF16_SKEW -DF16_IDX -DF16_FIXED0 -DF16_FIXED0_G8 -DF16_FIXED0_G15 -DF16_FIXED0_PAD" \
               exp_f16_tile_fixed0_g16pad:"-DMAX_STRIDE=8 -DF16_AFFINE -DF16_PIPE2 -DF16_TAIL256 -DSINGLE_TILE -DF16_SKEW -DF16_IDX -DF16_FIXED0 -DF16_FIXED0_G8 -DF16_FIXED0_G16 -DF16_FIXED0_PAD" \
               exp_f16_tile_fixed0_g8pad:"-DMAX_STRIDE=8 -DF16_AFFINE -DF16_PIPE2 -DF16_TAIL256 -DSINGLE_TILE -DF16_SKEW -DF16_IDX -DF16_FIXED0 -DF16_FIXED0_G8 -DF16_FIXED0_PAD" \
               exp_f16_tile_fixed0_g8:"-DMAX_STRIDE=8 -DF16_AFFINE -DF16_PIPE2 -DF16_TAIL256 -DSINGLE_TILE -DF16_SKEW -DF16_IDX -DF16_FIXED0 -DF16_FIXED0_G8" \
               exp_f16_tile_fixed0_idx:"-DMAX_STRIDE=8 -DF16_AFFINE -DF16_PIPE2 -DF16_TAIL256 -DSINGLE_TILE -DF16_SKEW -DF16_IDX -DF16_FIXED0" \
               exp_f16_affine_np_ms8:"-DMAX_STRIDE=8 -DF16_AFFINE" \
               exp_f16_affine_ms8:"-DMAX_STRIDE=8 -DF16_AFFINE -DF16_PIPE2 -DF16_TAIL256" \
               exp_f16_tile_affine_ms8:"-DMAX_STRIDE=8 -DF16_AFFINE -DF16_PIPE2 -DF16_TAIL256 -DSINGLE_TILE" \
               exp_f16_tile_affine_nomain_ms8:"-DMAX_STRIDE=8 -DF16_AFFINE -DF16_PIPE2 -DF16_TAIL256 -DSINGLE_TILE -DPROFILE_NO_MAIN" exp_f16_tile_affine_skew_ms8:"-DMAX_STRIDE=8 -DF16_AFFINE -DF16_PIPE2 -DF16_TAIL256 -DSINGLE_TILE -DF16_SKEW" exp_f16_tile_affine_skewidx_ms8:"-DMAX_STRIDE=8 -DF16_AFFINE -DF16_PIPE2 -DF16_TAIL256 -DSINGLE_TILE -DF16_SKEW -DF16_IDX"; do
    name=${variant%%:*}
    read -r -a defs <<< "${variant#*:}"
    tpc-clang -O2 -Wall -Werror -march=gaudi2 "${defs[@]}" "$HERE/kernels/exp_f16_logq.c" -c -o "$HERE/build/$name.o"
    tpc-clang -O2 -march=gaudi2 "${defs[@]}" -S "$HERE/kernels/exp_f16_logq.c" -o "$HERE/build/$name.s"
    if grep -E '\bld_g\b|\bst_g\b' "$HERE/build/$name.s"; then echo "Unexpected global access in $name" >&2; exit 1; fi
    python3 "$HERE/extract_text.py" "$HERE/build/$name.o" "$HERE/build/$name.bin"
done

# Opt-in 256-query/key-half experiment, same explicit logq LUT contract.
for variant in exp_f16_tile_logq_ms8 normalize_tile256; do
    if [[ $variant == exp_f16_tile_logq_ms8 ]]; then
        source_file=exp_f16_logq; defs=(-DMAX_STRIDE=8 -DSINGLE_TILE)
    else
        source_file=normalize_batch; defs=(-DTILE_ROWS=256)
    fi
    tpc-clang -O2 -Wall -Werror -march=gaudi2 "${defs[@]}" "$HERE/kernels/$source_file.c" -c -o "$HERE/build/$variant.o"
    tpc-clang -O2 -march=gaudi2 "${defs[@]}" -S "$HERE/kernels/$source_file.c" -o "$HERE/build/$variant.s"
    if grep -E '\bld_g\b|\bst_g\b' "$HERE/build/$variant.s"; then echo "Unexpected global access in $variant" >&2; exit 1; fi
    python3 "$HERE/extract_text.py" "$HERE/build/$variant.o" "$HERE/build/$variant.bin"
done

# Validated Conv[1] two-row-tile experiment uses the same exp kernel ABI.
tpc-clang -O2 -Wall -Werror -march=gaudi2 -DQUAD_ROWS=1024 "$HERE/kernels/normalize_quad_fixed.c" -c -o "$HERE/build/normalize_quad_fixed1024.o"
tpc-clang -O2 -march=gaudi2 -DQUAD_ROWS=1024 -S "$HERE/kernels/normalize_quad_fixed.c" -o "$HERE/build/normalize_quad_fixed1024.s"
if grep -E '\blookup(_|\.)|\bld_g\b|\bst_g\b' "$HERE/build/normalize_quad_fixed1024.s"; then echo 'Unexpected normalization global/LUT access' >&2;exit 1;fi
python3 "$HERE/extract_text.py" "$HERE/build/normalize_quad_fixed1024.o" "$HERE/build/normalize_quad_fixed1024.bin"

g++ -std=gnu++17 -O2 -I"$HERE/third_party/synapseai-core/lut" "$HERE/gen_lut.cpp" \
    "$HERE/third_party/synapseai-core/lut/SpecialFuncCoefficientsGen6.cpp" -o "$HERE/build/gen_lut"
"$HERE/build/gen_lut" "$HERE/build/lut_gen6.bin"
"$HERE/build/gen_lut" "$HERE/build/lut_gen6_logq.bin" --logq
"$HERE/build/gen_lut" "$HERE/build/lut_gen6_logq2.bin" --logq2
"$HERE/build/gen_lut" "$HERE/build/lut_gen6_logq2_shift8.bin" --logq2-shift8
# logq2 must equal logq except inside LUT128 slot15 (FID 0x8f, 256 bytes).
python3 - "$HERE/build/lut_gen6_logq.bin" "$HERE/build/lut_gen6_logq2.bin" <<'PY'
import struct, sys
a, b = (open(x, 'rb').read() for x in sys.argv[1:])
assert len(a) == len(b)
sizes = struct.unpack_from('<4I', a, 12)
t1 = 28 + (sizes[0] + 8191) // 8192 * 8192
diff = [i for i in range(len(a)) if a[i] != b[i]]
slot = 13 * 128  # SPECIAL_FUNCS_NUM_OF_CACHE_LINES_IN_FUNC_128 * CACHE_LINE_SIZE_IN_BYTES
assert diff and all(t1 + 15 * slot <= i < t1 + 15 * slot + 256 for i in diff), (diff[:4], t1)
print('LOGQ2_DIFF_OK bytes=%d range=0x%x..0x%x' % (len(diff), diff[0], diff[-1]))
PY
# Shift8 changes only slot15 by an exact BF16 exponent offset.
python3 - "$HERE/build/lut_gen6_logq2.bin" "$HERE/build/lut_gen6_logq2_shift8.bin" <<'PY'
import pathlib,struct,sys
a,b=[pathlib.Path(x).read_bytes() for x in sys.argv[1:]]
sizes=struct.unpack_from('<4I',a,12);off=28+((sizes[0]+8191)//8192)*8192+15*13*128
assert a[:off]==b[:off] and a[off+256:]==b[off+256:]
assert all(y-x==-1024 for x,y in zip(struct.unpack_from('<128h',a,off),struct.unpack_from('<128h',b,off)))
print('LOGQ2_SHIFT8_DIFF_OK')
PY
for name in tpc_runner attention_runner; do
    g++ -std=gnu++17 -O2 -Wall -Wextra -Werror \
        -I"$HERE/third_party/synapseai-core" -I"$T/include/uapi" -I"$T/include" \
        -I"$T/include/specs" -I"$T/include/specs/common" -I/usr/include/drm \
        "$HERE/$name.cpp" -L"$L" -Wl,-rpath,"$L" -lhl-thunk-tests -lhl-thunk -o "$HERE/build/$name"
done
if [[ ${BUILD_CONTEXT_PROBE:-0} == 1 ]]; then
    g++ -std=gnu++17 -O2 -Wall -Wextra -Werror -fPIC -shared \
        -I/usr/include/habanalabs -I/usr/include/drm -I"$T/include/uapi" -I"$T/include/specs" \
        "$HERE/context_bridge.cpp" -L"$L" -Wl,-rpath,"$L" \
        -lSynapse -lscal -lhl-thunk -o "$HERE/build/libcontext_bridge.so"
fi
