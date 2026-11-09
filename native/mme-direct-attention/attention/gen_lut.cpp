// SPDX-License-Identifier: MIT
// Writes the Gaudi2 TPC special-function coefficient tables (the same generator
// SynapseAI_Core's runner uses) to one file:
//   "G2LUT6\0\0", u32 count=4, u32 sizes[4], then the tables, each padded to 8 KiB.
// Table i is for LUT_FUNC{256,128,64,32}_BASE_ADDR, i = 0..3.
#include <cstdint>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstring>
#include "SpecialFuncCoefficients.h"
#include "gaudi2/SpecialFuncCoefficients_defGen6.h"

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3 || (argc == 3 && strcmp(argv[2], "--logq") && strcmp(argv[2], "--logq2") && strcmp(argv[2], "--logq2-shift8"))) {
        fprintf(stderr, "usage: %s out.bin [--logq|--logq2|--logq2-shift8]\n", argv[0]); return 2;
    }
    std::vector<std::vector<uint8_t>> tables(SPECIAL_FUNC_NUM_OF_DIFFERENT_INTERVALS);
    tpc_gaudi2::buildSpecialFunctionCoefficients(tables);
    if (argc == 3) {
        // FID 0x8e: 128-entry slot 14, unused by this pinned generator/header.
        // Same consecutive int16 packing as addTableLookup(elementSize=2).
        const unsigned offset = 14 * SPECIAL_FUNCS_NUM_OF_CACHE_LINES_IN_FUNC_128 * CACHE_LINE_SIZE_IN_BYTES;
        for (unsigned j = 0; j < 256; ++j) {
            if (tables[SPECIAL_FUNC_128_ENTRIES][offset + j]) {
                fprintf(stderr, "LUT128 slot14 is occupied; refusing to replace it\n"); return 3;
            }
        }
        for (unsigned j = 0; j < 128; ++j) {
            int16_t correction = (int16_t)std::nearbyint((std::exp2(j / 128.0) - 1.0) * 128.0) - j;
            memcpy(tables[SPECIAL_FUNC_128_ENTRIES].data() + offset + j * 2, &correction, 2);
        }
        printf("LOGQ_TABLE fid=0x8e slot=14 entries=128\n");
        if (!strcmp(argv[2], "--logq2") || !strcmp(argv[2], "--logq2-shift8")) {
            // FID 0x8f: slot 15, the same correction with the BF16 exponent bias
            // 16256 (=127<<7) folded in, for the FP16-affine exp kernels.
            const unsigned off15 = 15 * SPECIAL_FUNCS_NUM_OF_CACHE_LINES_IN_FUNC_128 * CACHE_LINE_SIZE_IN_BYTES;
            for (unsigned j = 0; j < 256; ++j) {
                if (tables[SPECIAL_FUNC_128_ENTRIES][off15 + j]) {
                    fprintf(stderr, "LUT128 slot15 is occupied; refusing to replace it\n"); return 3;
                }
            }
            for (unsigned j = 0; j < 128; ++j) {
                int16_t value = (int16_t)(16256 - (!strcmp(argv[2], "--logq2-shift8") ? 1024 : 0) + (int)std::nearbyint((std::exp2(j / 128.0) - 1.0) * 128.0) - (int)j);
                memcpy(tables[SPECIAL_FUNC_128_ENTRIES].data() + off15 + j * 2, &value, 2);
            }
            printf("LOGQ_BIAS_TABLE fid=0x8f slot=15 entries=128 offset=%u\n", off15);
        }
    }
    FILE *f = fopen(argv[1], "wb");
    if (!f) return 1;
    const char magic[8] = {'G', '2', 'L', 'U', 'T', '6', 0, 0};
    uint32_t count = tables.size(), sizes[4] = {};
    for (unsigned i = 0; i < 4 && i < count; ++i) sizes[i] = tables[i].size();
    fwrite(magic, 1, 8, f); fwrite(&count, 4, 1, f); fwrite(sizes, 4, 4, f);
    std::vector<uint8_t> pad(8192, 0);
    for (auto &t : tables) {
        fwrite(t.data(), 1, t.size(), f);
        fwrite(pad.data(), 1, (8192 - t.size() % 8192) % 8192, f);
    }
    fclose(f);
    printf("LUT_TABLES count=%u sizes=%u,%u,%u,%u\n", count, sizes[0], sizes[1], sizes[2], sizes[3]);
    return 0;
}
