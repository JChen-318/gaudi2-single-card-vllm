# NVFP4 MoE TPC kernels for Gaudi2

Two kinds of TPC-C kernels for the Qwen3.8-Flash-Next MoE of this checkpoint,
plus their host GlueCode and PyTorch operators:

| GUID | what it does |
| --- | --- |
| `custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2` | decode: grouped NVFP4 GEMV over the selected expert group, one launch, F32 accumulate, `scale2` after the block sum, 4 output rows per TPC member |
| `custom_qwen4exp_nvfp4_moe_count_i32_gaudi2` | prefill bucketing 1/3: routed slots per expert per 64-slot block |
| `custom_qwen4exp_nvfp4_moe_prefix_i32_gaudi2` | prefill bucketing 2/3: exclusive running count over blocks + per-expert totals |
| `custom_qwen4exp_nvfp4_moe_scatter_i32_gaudi2` | prefill bucketing 3/3: expert-major bucket rows + the inverse `place` map |

The kernels are a port of the STRATA reference engine's 4-bit MoE path
(`hpu/tpc/iq4q8_gemv_packed.c` for the decode GEMV structure,
`moe_mme_count.c` / `moe_mme_prefix.c` / `moe_mme_scatter.c` for the bucketing),
re-expressed for NVFP4: **E2M1 nibbles, one E4M3 scale per 16 K elements and one
F32 `scale2` per expert, all weights kept 4-bit packed.** Nothing is requantized
to IQ4/FP8 and no expert is ever materialized as BF16.

The existing port (`vllm-gaudi/csrc/qwen4exp/`) is untouched: this library is a
new, separate build with its own GUID names, so both can be loaded through
`native/tpc-kernel-shim` at the same time.

## Build

Host only - compiles TPC ELFs with `tpc-clang -march=gaudi2` and never opens the
device:

```bash
cmake -S native/qwen4exp-nvfp4-moe -B build/native/qwen4exp-nvfp4-moe -DCMAKE_BUILD_TYPE=Release
cmake --build build/native/qwen4exp-nvfp4-moe --parallel
```

`scripts/build_native.sh` does the same and copies the results into
`build/native/`:

- `libqwen4exp_nvfp4_moe_kernels.so` - the four TPC ELFs (embedded with
  `objcopy`), the GlueCode and the registration entry points
- `hpu_qwen4exp_nvfp4_moe_pt2.cpython-312-*.so` - the PyTorch operators
- `qwen4exp_nvfp4_moe_geometry_test` - host geometry/caps test (no device)
- `qwen4exp_nvfp4_moe_reference` - scalar C reference for the decode GEMV math
  (`tests/nvfp4_moe_gemv_reference.c`), run by
  `validation/qwen4exp/test_nvfp4_moe_reference.py`
- `qwen4exp_nvfp4_moe_sim` - TPC-simulator harness (no device), built only where
  the Habana simulator libraries (`libtpcsim_shared`, `libtpc_tests_core_ext`)
  are installed
- `qwen4exp_nvfp4_moe_latency` - per-call latency estimate at the checkpoint's
  layer shapes (`tests/qwen4exp_nvfp4_moe_latency.cpp`), same simulator, about
  40 s

`ctest` in the build directory runs the simulator harness and the latency
harness; `validation/qwen4exp/test_nvfp4_moe_{geometry,reference,ops}.py` cover
the glue, the numerics and the Python contract.

## Wiring

The recipe cache needs `GC_KERNEL_PATH` to be one regular file, so the library
is loaded behind the shim. `scripts/build_native.sh` extends the list:

```bash
GC_KERNEL_PATH=/path/to/libkiro_tpc_shim.so
KIRO_TPC_LIBS=/path/libmtp_sampling_kernels_v2.so:/path/libflashinfer_gaudi_kernels.so:/path/libqwen4exp_nvfp4_moe_kernels.so:/usr/lib/habanalabs/libtpc_kernels.so
```

`shim_check` proves every GUID of every component is reachable through the
front-end. Adding this library takes the merged, de-duplicated list from
`3262` to `3266` GUIDs (`+4`, `missing_from_merged=0`).

The PyTorch operators are registered by loading the extension once with
`torch.ops.load_library(...)`: it is a plain `TORCH_LIBRARY_FRAGMENT(custom_op,
...)` registration with `HPU` and `Meta` implementations, not a Python module
(the same shape as `vllm-gaudi/csrc/qwen4exp/pytorch`, which also has no
`PyInit`). Loading needs `habana_frameworks.torch` imported first so the `HPU`
dispatch key exists:

```python
import torch, habana_frameworks.torch
torch.ops.load_library("build/native/hpu_qwen4exp_nvfp4_moe_pt2.cpython-312-x86_64-linux-gnu.so")
torch.ops.custom_op.custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2  # now registered
```

With meta tensors the `Meta` implementations run on the host, so the shape and
parameter contract (and the argument validation) can be checked without a
device; the `HPU` paths need the Gaudi2 device.

## Python contract (Track 4 selection)

`qwen4exp_nvfp4_moe_ops.py` in this directory is the importable contract for a
decode path that wants to use the fused GEMV:

```python
import sys
sys.path.insert(0, "native/qwen4exp-nvfp4-moe")   # the directory name has a dash
from qwen4exp_nvfp4_moe_ops import GemvSpec, available, available_for_tensors, gemv

spec = GemvSpec(slots=10, experts=512, rows=1280, k=2560, device="hpu")
supported, why = available(spec, op_registered=True)   # op_registered defaults to a live check
if supported:
    y = gemv(ids, packed, scales, x, scale2)           # [S,N] BF16
```

- `OP_GEMV` = `custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2`; the other three
  operators are exported as `OP_COUNT` / `OP_PREFIX` / `OP_SCATTER`.
- Weight layout name: `LAYOUT_PACKED` = `nvfp4_e_n_khalf` (`[E,N,K/2]` uint8,
  low nibble = even k, the checkpoint's native layout - no TPC preshuffle).
- Scale layout name: `LAYOUT_SCALES` = `e4m3_block16_e_n_k16` (`[E,N,K/16]`
  bytes of float8_e4m3fn, one per 16 K elements). A float8_e4m3fn scale tensor
  is accepted by `available()` and reinterpreted to uint8 by `gemv()`.
- `available()` mirrors the host glue's caps and never imports `torch.hpu` or
  `habana_frameworks`, so it is host-testable; `available_for_tensors()` adds
  the exact plane-shape checks, `gemv()` raises `RuntimeError` with the reason
  when the call cannot be served (fall back instead of aborting a serve).
- Env: `VLLM_QWEN4EXP_MOE_LIB` points `load_library()` at the built op;
  `VLLM_QWEN4EXP_TRACK_C_MOE=0` disables selection.


## Operators

Argument order is the host glue's tensor order, then the kernel's scalar
arguments in declaration order (packed into node params).

```text
custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2(ids, packed, scales, x, scale2) -> Tensor
  ids    [S]        I32   selected expert per routed slot (-1 -> zero rows)
  packed [E,N,K/2]  U8    E2M1 nibbles, low nibble = even k
  scales [E,N,K/16] U8    per-16 E4M3 block scales
  x      [S,K]      BF16  routed activations
  scale2 [E]        F32   per-expert global scale
  -> [S,N] BF16

custom_qwen4exp_nvfp4_moe_count_i32_gaudi2(ids, int experts) -> Tensor
  ids [S] I32             (< 0 = skip)
  -> counts [blocks, E] I32, blocks = ceil(S/64)

custom_qwen4exp_nvfp4_moe_prefix_i32_gaudi2(counts) -> (Tensor, Tensor)
  counts [blocks, E] I32
  -> offsets [blocks, E] I32, totals [E] I32

custom_qwen4exp_nvfp4_moe_scatter_i32_gaudi2(ids, offsets, int experts,
                                             int slots, int cap) -> (Tensor, Tensor)
  ids [S] I32, offsets [blocks, E] I32, cap >= slots
  -> bucket [E*cap] I32 (view as [E, cap]), place [S] I32 (-1 = skipped)
```

Python-side use for one MoE layer:

```python
counts = custom_qwen4exp_nvfp4_moe_count_i32_gaudi2(ids, experts)      # [blocks, E]
offsets, totals = custom_qwen4exp_nvfp4_moe_prefix_i32_gaudi2(counts)
bucket, place = custom_qwen4exp_nvfp4_moe_scatter_i32_gaudi2(ids, offsets, experts, S, cap)
bucket = bucket.view(experts, cap)          # row e*cap+m is a routed slot of expert e
# one grouped GEMM over bucket[:totals[e]] rows per expert instead of an S-loop per expert
out = custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2(ids, packed, scales, x, scale2)
```

`counts`/`offsets` are block-major (`[blocks, E]`) because the expert axis is
the contiguous one - the kernels vectorize 64 experts per 64-lane vector, which
is also why `E` must be a multiple of 64. The bucket is expert-major because
that is the layout a grouped GEMM consumes.

## Geometry

TPC sees tensors with Synapse's outermost-dimension-first order, i.e. dim 0 is
torch's contiguous axis:

| tensor | torch | TPC |
| --- | --- | --- |
| packed | `[E,N,K/2]` | `[K/2,N,E]` |
| scales | `[E,N,K/16]` | `[K/16,N,E]` |
| x | `[S,K]` | `[K,S]` |
| output | `[S,N]` | `[N,S]` |
| counts / offsets | `[blocks,E]` | `[E,blocks]` |
| bucket / place / totals | `[E*cap]` / `[S]` / `[E]` | same |

Index spaces:

| kernel | index space | member |
| --- | --- | --- |
| decode GEMV | `(N/4, S)` | one slot and 4 output rows |
| count | `(blocks, E/64)` | one 64-slot block and 64 experts |
| prefix | `(E/64)` | 64 experts, loop over blocks |
| scatter | `(blocks)` | one 64-slot block |

Caps enforced by the glue (each returns `GLUE_INCOMPATIBLE_*` /
`GLUE_UNSUPPORTED_LAYER_CONFIGURATION` instead of miscomputing):

- decode GEMV: `K % 128 == 0`, `N % 4 == 0` (rows per member), `1 <= S <= 64`,
  `64 <= E <= 4096` with `E % 64 == 0`
- count: `64 <= E <= 4096` with `E % 64 == 0`, `1 <= S <= 65536`
- prefix: `E % 64 == 0`, `1 <= blocks <= 1024`
- scatter: `E % 64 == 0`, `1 <= S <= 65536`, `slots == ids.numel()`, `cap >= slots`

## Numerics

The decode is the same integer bit construction as the shipped
`vllm-gaudi/csrc/qwen4exp/include/qwen4exp_nvfp4_decode.h`
(`nvfp4_mul_bits`, copied into `include/`): `e2m1 * e4m3` has at most 5
significant mantissa bits, so the product is built directly as a BF16 bit
pattern with no rounding. The per-expert `scale2` multiplies the finished F32
block sum, and the result is stored as BF16 - so `scale2` is a factor of the
whole dot product rather than of every element, which changes F32 rounding but
not the decoded weights.

## Verification (host only, no device)

Three pytest modules and the binaries above cover everything that can be checked
without the Gaudi2 device:

| check | where | result |
| --- | --- | --- |
| GUID registration, index spaces, caps, re-derived shapes | `qwen4exp_nvfp4_moe_geometry_test` / `validation/qwen4exp/test_nvfp4_moe_geometry.py` | all glue cases pass |
| the four GUIDs survive the shim merge | `shim_check` (same `KIRO_TPC_LIBS` as `build_native.sh`) | `missing_from_merged=0`, 3262 -> 3266 |
| the real TPC ELF executes correctly | `qwen4exp_nvfp4_moe_sim` (TPC simulator) | one-hot decode bit-exact (256/256); power-of-two data bit-exact against the exact double sum (16/16); bucketing integer-exact |
| decode + F32 accumulate + `scale2` vs a compiled C reference | `validation/qwen4exp/test_nvfp4_moe_reference.py` + `qwen4exp_nvfp4_moe_reference` | decode bit-exact; accumulation within 2e-3, exact on power-of-two data |
| the Python contract Track 4 selects on | `validation/qwen4exp/test_nvfp4_moe_ops.py` | 32 cases: accept/reject reasons, tensor shapes, float8 scales, no HPU import |
| per-call latency estimate | `qwen4exp_nvfp4_moe_latency` (TPC simulator) | see the table below |

The numeric evidence in detail:

- `nvfp4_mul_bits` == E2M1 LUT x E4M3 scale for all 16 x 254 (nibble, scale)
  pairs, bit for bit.
- The decoded values are bit-identical to vllm-core's
  `dequantize_to_dtype` (NVFP4 reference) on random packed/scales.
- `tests/nvfp4_moe_gemv_reference.c` is an independent scalar implementation
  (E2M1 table x E4M3 value in double, double accumulation, `scale2` after the
  sum). The numpy emulation of the kernel - same accumulation order (lane j takes
  k = j mod 128, 128-lane F32 accumulator, 64+64 fold, BF16 round-to-nearest-even
  store) - reproduces the vectorized exact-math evaluation to 1e-12, and on
  random K = 1280 data the maximum relative deviation from the exact sum is
  3.1e-5 (bound 2e-3 = the BF16 output granularity; F32 rounding amplified by
  cancellation) with all 48 BF16 outputs bit-identical. On data where every
  product and partial sum is a power of two the agreement is bit for bit, and so
  is the BF16 output.
- In the TPC simulator, one-hot activations make every output element a single
  decoded weight times `scale2`: 256/256 outputs are bit-exact. The only
  permitted difference is a signed zero (a lone `-0.0` weight summed with 127
  zero lanes is `+0.0`).
- In the TPC simulator, a power-of-two data set (nibble magnitudes 0/0.5/1,
  E4M3 scales 2**-1/2**0, activations 0/+-0.25/+-0.5/+-1, `scale2` 2**-2/2**0)
  makes every product and every partial sum exact in F32, so the ELF's stored
  BF16 must equal the round-to-nearest BF16 of the exact double sum: 16/16
  outputs bit-identical, `max_abs=0.000e+00`.
- Random data at the checkpoint's K values, kernel-order emulation vs the
  compiled C reference: max relative deviation 3.140e-05 (K=1280), 1.414e-06
  (K=2560), 5.581e-06 (K=640); all 48/48 BF16 outputs bit-identical in every
  case.
- In the TPC simulator, the three bucketing kernels reproduce the same integer
  arithmetic exactly, including `ids < 0` (skip), an out-of-range id, and a tail
  block that is not 64 slots.
- The operator library loads through `torch.ops.load_library` and its `Meta`
  implementations produce the documented shapes for all four operators; the
  argument checks reject a slots/ids mismatch. The `HPU` implementations are the
  only part that needs the device.

### Per-call latency estimate

`qwen4exp_nvfp4_moe_latency` runs the real ELF at the checkpoint's layer shapes
and converts the simulator cycle count the way `native/gdn-fused-kernel` does:
whole members per TPC (one member = one thread), 24 TPCs, 1.7 GHz nominal TPC
clock, 1 IPC (the simulator issues one instruction per cycle and does not model
HBM latency). E is the minimum the glue accepts (64): a member's instruction
stream does not depend on E, and the harness measures that (the same shape at
E=64 and E=128 both run 74,884 cycles).

| call | K | N | S | cycles | cycles/member | per call |
| --- | --- | --- | --- | --- | --- | --- |
| gate/up | 2560 | 1280 | 10 | 16,724,148 | 5226.3 | 0.412 ms |
| down | 640 | 2560 | 10 | 9,165,748 | 1432.1 | 0.225 ms |
| gate/up | 2560 | 1280 | 1 | 1,672,506 | 5226.6 | 0.043 ms |

One decode token with 10 routed slots is 48 layers x (2 gate/up + 1 down) =
`48 x (2 x 0.412 + 0.225) = 50.3 ms` of GEMV time in this model; at S=1 it is
`48 x (2 x 0.043 + 0.0225) = 5.2 ms`. That is an instruction-throughput
estimate for the compute-side cost of the integer NVFP4 decode, not a device
measurement: the simulator does not model memory stalls, and the 1 IPC
assumption is the convention the GDN harness validated against a device trace.

### Software pipelining: measured, not adopted

Strata's `iq4q8_gemv_packed_pf.c` also prefetches the *next* column pair's
loads before the current pair's MACs. Porting that to this kernel (next tile's
four packed and four scale vectors held in shadow registers) was measured in
the same simulator harness: at K=1280/N=320/S=10 it runs 2,397,732 cycles
against 2,156,948 for the shipped load-then-consume loop (+11.2%; +18.3% at
K=128/N=4/S=64). The register pressure pushes local-memory traffic from 125 to
184 `st_l_v`/`ld_l_v` instructions, so the extra prefetch costs more than it
hides. The shipped kernel keeps the within-tile form: all eight 256-byte
weight loads of a tile are issued before that tile's first MAC (verified in the
generated assembly), which already overlaps four K parts x four rows of
arithmetic with the loads.

## The kernels

**Decode GEMV** (`kernels/qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.c`). One index
space member owns one routed slot and four output rows: it loads the four rows'
packed bytes and E4M3 scales (8 independent 256-byte loads) before the first
MAC, then reuses one activation vector for all four rows. That is Strata's
latency hiding (`iq4q8_gemv_packed.c` keeps five row tiles and four K parts in
flight per member), applied to NVFP4's per-16 scales. Per 128-element K part:
replicate the 64 packed bytes into all four dual groups, shuffle each byte to
the four lanes that hold its two nibbles, look the E4M3 scale up per 16 lanes,
build the BF16 weight with `nvfp4_mul_bits`, and accumulate `weight * x` in F32
(`v_bf16_mac_acc32_b`). After the K loop: F32 tree reduction, `scale2`, one BF16
store. Per (slot, row) the MAC order is K ascending, so the result is
bit-identical to the shipped one-row kernel. In the generated assembly all
eight `ld_tnsr` weight loads of a tile precede its first `mac.bf16`, and the
kernel uses 40 vector registers with 125 local-memory instructions (no
cross-tile prefetch - see the measured note above).

**Count** (`..._count_i32_gaudi2.c`). One member loads a 64-slot block's ids as
one vector and masks the lanes that are not part of the block (a tail block
loads the last 64 slots instead of reading past the tensor), then runs 64
compare + reduce pairs eight at a time - eight independent dependent chains,
Strata's `moe_mme_count.c` unroll. Ids below zero are skipped by construction.

**Prefix** (`..._prefix_i32_gaudi2.c`). 64 experts per member; one 64-lane
vector is the running exclusive count over blocks, so a block costs one load,
one store and one add. Integer arithmetic end to end (`moe_mme_prefix.c` uses
F32; counts are exact in both, I32 removes the question).

**Scatter** (`..._scatter_i32_gaudi2.c`). One member per 64-slot block: the ids
and per-block offsets are fetched in batches of eight (a scalar global load
waits a full memory round trip - Strata's `moe_mme_scatter.c` batches for the
same reason), "earlier slots of this block routed to the same expert" is a
64-entry open-addressed table in local memory, and a full block's 64 `place`
entries leave as one vector store. `bucket[e*cap + offsets[e][block] + earlier]`
= the slot, and `place[slot]` is its inverse (-1 when the slot is skipped), so
the Python side can run one grouped GEMM per expert-range and route the results
back without a second search.

## What is not covered

- Only the decode GEMV and the bucketing are ported. The grouped GEMM that
  consumes `bucket` (the prefill MME path, Strata's `u4_moe_gu_epi` /
  `u4_moe_down_epi` / `u4_moe_join`) is not part of this library; the bucketing
  produces exactly the inputs such a GEMM needs (expert-major rows, per-expert
  counts, inverse map).
- No device run: everything here is host-compiled, shim-verified and
  simulator-verified. Cycle counts in the simulator are instruction counts, not
  device timings: HBM latency/bandwidth, L1 behaviour and co-issue are not
  modelled, so the latency table is a compute-side estimate.
- The Python contract module is host-tested only; `load_library()` and the HPU
  implementations of the operators need the device.
- The shared expert and the dense/attention layers keep using the existing
  `vllm-gaudi/csrc/qwen4exp` kernels.
- The 256-byte packed/scale loads keep the shipped kernel's convention of
  reading whole vectors (a K that is not a multiple of 512 over-reads the row by
  less than one 256-byte vector). The ids reads are the same: one 64-lane
  window, so a slot list shorter than 64 reads one window from slot 0.
