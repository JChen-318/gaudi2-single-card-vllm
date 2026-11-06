// SPDX-License-Identifier: Apache-2.0
// Scalar host reference for the Qwen4Exp NVFP4 MoE decode GEMV
// (custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2), compiled and run on the host:
// no device, no TPC compiler, no Habana runtime.
//
// Deliberately independent of the TPC kernel's integer bit construction
// (nvfp4_mul_bits): every weight is decoded from its E2M1 value-table entry and
// the E4M3 (float8_e4m3fn) definition in double precision, multiplied by the
// exact BF16 activation, accumulated in double in ascending K order, and scaled
// by the per-expert scale2 after the sum. That is the exact value the kernel's
// F32 lane accumulation approximates; validation/qwen4exp/
// test_nvfp4_moe_reference.py compares its own numpy emulation of the kernel
// against this file's output.
//
// The operator semantics reproduced here, including the edge cases:
//   out[s, n] = scale2[e] * sum_k e2m1[nibble(e,n,k)] * e4m3(scales(e,n,k/16)) * x[s,k]
//   e = ids[s]; ids[s] < 0 or ids[s] >= experts  ->  out[s, :] = +0.0
//   low nibble of a packed byte is the even k, one E4M3 scale covers 16 k
//
// File format (little-endian, packed; offsets in bytes):
//   in:  u32 magic 1024; u32 experts, rows, k, slots          (16 bytes header)
//        i32 ids[slots]
//        u8  packed[experts * rows * (k / 2)]
//        u8  scales[experts * rows * (k / 16)]
//        u16 x[slots * k]                                     (BF16 bits)
//        f32 scale2[experts]
//   out: u32 magic 1024; u32 slots, rows
//        f64 golden[slots * rows]
//
// usage: nvfp4_moe_gemv_reference <inputs.bin> <golden.bin>
// exit:  0 ok, 2 usage, 3 I/O error, 4 malformed input
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QMOE_MAGIC 1024u

// E2M1 value table: the magnitude nibble's three bits, sign applied separately.
static const double kE2M1[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};

// E4M3 (float8_e4m3fn) -> double. 0x7F/0xFF are NaN and are rejected above.
static double e4m3_to_double(uint8_t bits)
{
    const double sign = (bits & 0x80u) ? -1.0 : 1.0;
    const int exponent = (bits >> 3) & 0xF;
    const int mantissa = bits & 0x7;
    const double significand = exponent == 0 ? mantissa / 8.0 : 1.0 + mantissa / 8.0;
    const double scale = exponent == 0 ? ldexp(1.0, -6) : ldexp(1.0, exponent - 7);
    return sign * significand * scale;
}

static double bf16_to_double(uint16_t bits)
{
    const uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return (double)value;
}

static int read_exact(FILE* file, void* buffer, size_t bytes, const char* what)
{
    if (bytes == 0) return 0;
    if (fread(buffer, 1, bytes, file) != bytes) {
        fprintf(stderr, "reference: short read of %s (%zu bytes)\n", what, bytes);
        return -1;
    }
    return 0;
}

static int write_exact(FILE* file, const void* buffer, size_t bytes, const char* what)
{
    if (bytes == 0) return 0;
    if (fwrite(buffer, 1, bytes, file) != bytes) {
        fprintf(stderr, "reference: short write of %s (%zu bytes)\n", what, bytes);
        return -1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <inputs.bin> <golden.bin>\n", argv[0]);
        return 2;
    }
    FILE* in = fopen(argv[1], "rb");
    if (!in) {
        perror("reference: open inputs");
        return 3;
    }
    uint32_t header[5] = {0};
    if (read_exact(in, header, sizeof(header), "header") != 0) return 3;
    if (header[0] != QMOE_MAGIC || header[2] == 0 || header[3] == 0 || header[4] == 0) {
        fprintf(stderr, "reference: bad magic or empty shape\n");
        return 4;
    }
    const uint32_t experts = header[1];
    const uint32_t rows = header[2];
    const uint32_t k = header[3];
    const uint32_t slots = header[4];
    if (experts == 0 || k % 16 != 0) {
        fprintf(stderr, "reference: experts == 0 or k %% 16 != 0\n");
        return 4;
    }

    const size_t ids_bytes = (size_t)slots * sizeof(int32_t);
    const size_t packed_bytes = (size_t)experts * rows * (k / 2);
    const size_t scale_bytes = (size_t)experts * rows * (k / 16);
    const size_t x_bytes = (size_t)slots * k * sizeof(uint16_t);
    const size_t scale2_bytes = (size_t)experts * sizeof(float);

    int32_t* ids = (int32_t*)malloc(ids_bytes);
    uint8_t* packed = (uint8_t*)malloc(packed_bytes);
    uint8_t* scales = (uint8_t*)malloc(scale_bytes);
    uint16_t* x = (uint16_t*)malloc(x_bytes);
    float* scale2 = (float*)malloc(scale2_bytes);
    double* golden = (double*)malloc((size_t)slots * rows * sizeof(double));
    if (!ids || !packed || !scales || !x || !scale2 || !golden) {
        fprintf(stderr, "reference: out of memory\n");
        return 3;
    }
    if (read_exact(in, ids, ids_bytes, "ids") != 0 ||
        read_exact(in, packed, packed_bytes, "packed") != 0 ||
        read_exact(in, scales, scale_bytes, "scales") != 0 ||
        read_exact(in, x, x_bytes, "x") != 0 ||
        read_exact(in, scale2, scale2_bytes, "scale2") != 0)
        return 3;
    fclose(in);

    const size_t packed_stride = (size_t)rows * (k / 2);
    const size_t scale_stride = (size_t)rows * (k / 16);
    size_t nan_scales = 0;
    double checksum = 0.0;

    for (uint32_t s = 0; s < slots; ++s) {
        const int32_t expert = ids[s];
        const int valid = expert >= 0 && (uint32_t)expert < experts;
        for (uint32_t n = 0; n < rows; ++n) {
            double total = 0.0;
            if (valid) {
                const uint8_t* row_packed = packed + (size_t)expert * packed_stride +
                                            (size_t)n * (k / 2);
                const uint8_t* row_scales = scales + (size_t)expert * scale_stride +
                                            (size_t)n * (k / 16);
                for (uint32_t kk = 0; kk < k; ++kk) {
                    const uint8_t byte = row_packed[kk / 2];
                    const uint8_t nibble = (kk % 2) ? (uint8_t)((byte >> 4) & 0xF)
                                                    : (uint8_t)(byte & 0xF);
                    const uint8_t scale_byte = row_scales[kk / 16];
                    if (scale_byte == 0x7F || scale_byte == 0xFF) {
                        ++nan_scales;
                        continue;
                    }
                    const double sign = (nibble & 0x8) ? -1.0 : 1.0;
                    const double weight = sign * kE2M1[nibble & 0x7] * e4m3_to_double(scale_byte);
                    total += weight * bf16_to_double(x[(size_t)s * k + kk]);
                }
                total *= (double)scale2[expert];
            }
            golden[(size_t)s * rows + n] = total;
            checksum += fabs(total);
        }
    }
    if (nan_scales != 0) {
        fprintf(stderr, "reference: %zu NaN E4M3 scales in the input\n", nan_scales);
        return 4;
    }

    FILE* out = fopen(argv[2], "wb");
    if (!out) {
        perror("reference: open output");
        return 3;
    }
    const uint32_t out_header[3] = {QMOE_MAGIC, slots, rows};
    const int failed =
        write_exact(out, out_header, sizeof(out_header), "output header") != 0 ||
        write_exact(out, golden, (size_t)slots * rows * sizeof(double), "golden") != 0;
    if (fclose(out) != 0) return 3;
    if (failed) return 3;

    printf("reference: E=%u N=%u K=%u S=%u outputs=%u checksum=%.17g\n", experts, rows, k, slots,
           slots * rows, checksum);
    free(ids);
    free(packed);
    free(scales);
    free(x);
    free(scale2);
    free(golden);
    return 0;
}
