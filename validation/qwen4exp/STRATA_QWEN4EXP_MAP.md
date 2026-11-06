# STRATA qwen4_exp → Gaudi2: module map, 4-bit MoE GEMV analysis, call contracts

**Track 1 deliverable — read-only.** Neither tree was modified: this file, two extraction helpers
(`validation/qwen4exp/tools/strata_module_index.py`, `validation/qwen4exp/tools/check_strata_map.py`) and their
test module are the only additions. No device was opened, no `torch.hpu` import, no HPU allocation.

| tree | root | revision |
|---|---|---|
| STRATA reference engine | `$STRATA_ROOT` (default `~/gaudi2-single-card-strata`) | `874502d` (`git log -1`) |
| our port (this repo) | this repository | working tree |
| original NVIDIA CSF export (real CSF bytes for tests) | `~/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6…/` | 147,456 `nvfp4_csf_*` tensors, verified by the command in §7 |

**Anchor discipline.** Every `path:line` in backticks is machine-checked by
`tools/check_strata_map.py`: paths under `hpu/`, `src/`, `include/`, `docs/` resolve in **STRATA**; paths under
`vllm-gaudi/`, `validation/`, `native/` resolve in **this repo**. The checker also pins 55 load-bearing lines by
substring (the LUT multiply, the address formulas, the caps, the store, the PLE constants, the chain struct), so a
line that moves or gets reworded fails the check instead of silently invalidating the map. Run:

```bash
.venv/bin/python validation/qwen4exp/tools/check_strata_map.py    # anchors + pinned lines
.venv/bin/python validation/qwen4exp/tools/strata_module_index.py # every count/table below
```

**Relationship to `PORTING_MAP.md`.** That document (75 KB) is the *design* extraction: the QSA selection law, the
HC math, the PLE hash, the NVFP4 arithmetic deltas and the kernel port plan. This document is the *inventory and
contract* map: which artifact implements which stage, what we already have for it, where exactly the NVFP4 formula
replaces the IQ4 one, how to stream the PLE table from host RAM, and the five call contracts. Where the two overlap
they agree; §7 records what was re-verified here.

---

## 0. The port in one glance

| | count | how obtained |
|---|---:|---|
| STRATA TPC kernel sources (`hpu/tpc/*.c`) | 108 | `strata_module_index.py --section kernels` |
| STRATA kernels registered in `hpu/tpc/glue.cpp` | 112 | same (ids are the registration order) |
| STRATA host modules named in this map (`hpu/src/*.cpp`) | 35 (11,596 LOC) | `--section host` |
| our native TPC kernels | 4 (338 LOC) | `--section ours` |
| our native host/glue | 5 (539 LOC) | same |
| our Python port (`models/qwen4exp/`) | 18 modules (5,079 LOC) | same |
| our shipped `csrc/qwen4exp` (indexed dequant + GEMV) | 12 files (1,741 LOC) | same |
| our registered custom GUIDs | 4 | same |

The four GUIDs are the only kernels this port ships for the MoE:
`custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2`, `…_count_i32_gaudi2`, `…_prefix_i32_gaudi2`,
`…_scatter_i32_gaudi2` (`native/qwen4exp-nvfp4-moe/host/entry_points.cpp:114-126`).

Model geometry (from `config.json`, restated by STRATA at `include/strata/kernels/ngram.hpp:31-42`): `hidden=2560`,
`hc=4` streams, `hc_lowrank=320`, 48 layers = 36 GDN + 12 QSA, 512 routed experts, top-10 + 1 shared, `ff=640`
per expert, `head_dim=256`, indexer `4 heads × 128`, `block=4`, `budget=2048`, PLE `ngram=3, heads_per_ngram=8`.

---

## 1. Module structure of the STRATA Qwen3.8-Flash-Next port

### 1.1 One layer, in execution order

The host's layer order is fixed and short — read it once, because every kernel below hangs off it:
`hpu/src/layer_flow.cpp:13-32` (`execute_layer`):

```
PLE (layer 1 only) → HC read(attn) → mixer (QSA | GDN) → HC write(attn)
                   → HC read(ffn)  → router + MoE      → HC write(ffn)
```
with the window/batched twin at `hpu/src/layer_flow.cpp:33-88`. The decode path compiles that whole sequence per
layer into **one** recipe (`Layer::decode_chain`, `hpu/src/layer.cpp:1384-1460`) whose only host waits are the
router logits and (for QSA) the attention output; the prefill/verify path shares one workspace name for every
recipe of every row count (`hpu/src/layer.cpp:32-34`).

| stage | graph builder | kernels it binds | our counterpart |
|---|---|---|---|
| PLE (layer 1) | `hpu/src/hc_ple.cpp:132-150` (`define_ple_block`), window: `:160-219` | `strata_ple_gate_*` (7, 78, 103), `strata_ple_conv_*` (8, 104), SDK `row_norm` | `models/qwen4exp/ple_layer.py`, `ple_runtime.py` (torch) |
| HC read | `hpu/src/hc_ple.cpp:14-70` | `row_norm_dual` (39), `hc_down_bf16` (100) or SDK GEMM, `hc_mix_dual` (101) | `hyperconnection_amd.py`, `hyperconnection_base.py` |
| QSA | `hpu/src/qsa.cpp:115-144` (decode), `:278-309`, `:323-356` (windows) | `qsa_prep` (26), `qsa_gather*` (46/95/97/106/109), `qsa_select_mask` (91), `qsa_decode_blocks` (96) | `qsa.py`, `qsa_math.py` (torch) |
| GDN | `hpu/src/gdn.cpp:142-183` (block), `:187-240` (rows) | `gdn_conv*` (1/30/45/48), `gdn_prepare*` (2/31), `gdn_norm*` (3/32/…), `gdn_step*` (0/21/29/55) | vllm-core `Qwen3Next*` + `model.py` |
| router + MoE | `hpu/src/resident_experts.cpp:136-246` (decode), `hpu/src/moe_mme.cpp:37-192` (prefill) | `route_topk` (24), `q8_quant_packed` (34), `iq4q8_gemv*` (22/33/35/47/53/54/67), `moe_mme_*` (69–93), `iq4_expand_tiles*` (77/81) | `packed_moe.py` + `native/qwen4exp-nvfp4-moe` + shipped `csrc/qwen4exp` |
| HC write | `hpu/src/hc_ple.cpp:66-70`, window `:126-131` | `hc_write` (6), `hc_write_moe` (42), `hc_write_moe_routed` (88) | `hyperconnection_*.py` |
| lm head | `hpu/src/linear_graph.cpp:6-13` | SDK BF16 GEMM | `model.py` head |

### 1.2 QSA: the indexer, and the kernels that restate it on the device

**The law** (host, `hpu/src/qsa_state.cpp`):

1. *Pooling.* A block of `block=4` cells has one pooled key = **mean of the block's raw index keys** → RMS norm with
   the index-key gamma → NeoX RoPE at the **block's first cell** (`hpu/src/qsa_state.cpp:143-150`, `:365-372`;
   the window twin is `hpu/tpc/qsa_index_cells.c:1-11`).
2. *Score.* `Σ over indexer heads of max(q·k, 0)`, summed in `double` on the host
   (`hpu/src/qsa_state.cpp:430`), F32 on the device (`hpu/tpc/qsa_select_mask.c:55`).
3. *Order.* `(score desc, block id asc)` (`hpu/src/qsa_state.cpp:458-472`), and the row's **own unfinished block is
   forced first with +1e9** (`hpu/src/qsa_state.cpp:432`, device `hpu/tpc/qsa_select_mask.c:50-57`).
4. *Width.* `min(cells, top_k + block - 1)` = 2048 + 4 − 1 = 2051 cells at the checkpoint's budget
   (`hpu/src/qsa_state.cpp:164-167`).
5. *Cut.* The device finds the threshold score as a **fixed 11-pass 8-way bit search** over the sorted-as-bits F32
   scores (`hpu/tpc/qsa_select_mask.c:61-90`), then emits per block how many of its cells are kept; the mask is
   materialised by `qsa_select_expand` (92). The header comment `hpu/tpc/qsa_select_mask.c:1-19` is the compact
   statement of the whole law.

**The device pipeline** (used when `STRATA_HPU_QSA_DEVICE*` is on): index-query norm+RoPE
(`hpu/tpc/qsa_index_prep.c:1-6`) → pooled keys for the window's blocks (`hpu/tpc/qsa_index_cells.c:1-11`) → mask
(`qsa_select_mask`) → gather (`qsa_gather_fp8`, 95) → attention (SDK SDPA or F32 GEMM+softmax+GEMM). Decode-side
selection without a projection wait is `qsa_decode_blocks` (96) stitched into the layer graph
(`hpu/src/layer.cpp:1433-1460`). The KV cache can be FP8 with one F32 scale per (cell, KV head):
`hpu/src/qsa.cpp:11-57` (read/write/decode), 4× HBM reduction on a 256k context.

### 1.3 GDN: the gated delta rule

- Short conv, kernel 4, state `[taps-1, channels]` oldest-first, weights transposed from GGUF
  (`hpu/tpc/gdn_conv.c:1-4`).
- Prepare: `q,k` L2-normalised per head and scaled, `v`, `gate` (log decay) and `beta` (sigmoid) from the merged
  projection (`hpu/tpc/gdn_step.c:1-3`, `gdn_prepare.c`).
- Step, per value head, state `[S, value_heads, S]` last axis fastest (`hpu/tpc/gdn_step.c:1-3`):
  `decay=exp(gate)`, `contraction = Σ_i (old_i·decay)·k_i`, `delta = (v − contraction)·beta`,
  `new_i = old_i·decay + delta·k_i`, `y = Σ_i new_i·q_i` (`hpu/tpc/gdn_step.c:10-39`).
- Prefill: the same recurrence chunked at `kGdnChunk=64` with a hierarchical `(I+A)^{-1}`, every product a batched
  MME GEMM — the exact formulas and the numerical reason for the hierarchical inverse are in
  `hpu/src/gdn_chunked.cpp:1-17`; window variants `gdn_step_window` (55), `gdn_step_rows4` (43).

### 1.4 MoE: two completely different engines behind one router

- **Decode** (`hpu/src/resident_experts.cpp:136-246`): `route_topk` on the device (24) → packed-Q8 activation
  (34) → gate/up GEMV in `parts` column slices (`:205`) → fused `SiLU(gate)·up` + re-quantise (38) or
  split/silu/quant → down GEMV (`:226`) → route weighting and the per-row sum (`:235-245`). All weights resident as
  IQ4 codes + F32 per-32 scales; **no BF16 expert copy anywhere**. This is the engine §2 dissects.
- **Prefill** (`hpu/src/moe_mme.cpp:37-192`): the window's slots are bucketed on the device
  (count 72 → prefix 73 → tile-map 76 → place 93 / scatter 74, prefix-summed through an MME product with a constant
  upper triangle at `:73`), each expert's 4-bit codes are **expanded to BF16 once per window** as the B operand
  (`strata_iq4_expand_tiles_bf16`, 77; FP8 variant 81), then two batched GEMMs per chunk with the fused activation
  (69) between them, and a join that undoes the tile permutation (`moe_mme_join_perm`, 71). Overflowing buckets are
  *designed* to produce non-finite rows and are replayed on the decode path (`hpu/src/moe_mme.cpp:88-95` comment).
- **Alternatives** in the same tree: FP8 per-slot experts (`hpu/src/moe.cpp:43-100`), a single grouped FP8 decode
  GEMM over all slots (`hpu/src/moe.cpp:117-160`), TPC FP8 experts (`hpu/src/moe.cpp:137-142`), CPU experts.
  The per-expert global scale of an NVFP4 checkpoint is read here: `hpu/src/expert_scales.cpp:3-17`, with
  `hpu/src/expert_scales.cpp:13` insisting on one F32 per expert.

### 1.5 PLE: an n-gram lookup whose table never enters HBM

Sixteen rows per token (`ngram=3` → 2 n-grams × 8 heads, `include/strata/kernels/ngram.hpp:31-42`), each row 160
values, flattened head-slowest into `n_embd=2560`. The hash and its EOS cut are `src/kernels/ngram.cpp:72-98`; the
table is 320,001,536 rows (`include/strata/kernels/ngram.hpp:46`). Only the projections live on the device — the
`w_key [10240,2560]`/`w_value [2560,2560]` BF16 matrices and the norms/conv kernel of
`hpu/src/hc_ple.cpp:132-150`. Streaming is §4.

### 1.6 Hyper-connections (4 streams)

Read (`hpu/src/hc_ple.cpp:14-70`): one RMS norm over the **stacked** `r·h` vector, a `[lr+r, n]` fused
down+inject GEMM, `lo = SiLU(down/r)`, `gate = w_up·lo`, `mixed = (1/r)·Σ_s xn_s·σ(gate_s)`. Write
(`hpu/src/hc_ple.cpp:66-70`): `out_s = residual_s + block·2σ(inject_s/r)`. The `/r` and the `2` are the contract's,
not an implementation detail (`hpu/tpc/hc_mix.c:1-12`, `hpu/tpc/hc_write.c:1-13`). Fused tails exist:
`hc_mix_window_dual` (40), `hc_write_moe` (42) / `hc_write_moe_routed` (88) fold route weighting, the expert sum and
the HC write into one node.

### 1.7 Launch chaining

`DecodeChain` (`hpu/include/strata/hpu/decode_chain.hpp:14-37`) is the whole mechanism: **one stream**, the
residual in two device buffers with `in()/out()/advance()` (`:26-36`), the attention half's outputs in three
buffers (`mid`, `x`, `inject`, bound as `attn_write.out`, `ffn.mixed`, `ffn.inject`), expert-weight leases released
after the *next* wait (`:18-19`, `std::vector<ExpertLease> leases` at `:29`), and one pinned staging buffer for the
token's input/output. `VerifyChain` (`:38-64`) extends it to a window of rows and adds the shared `scratch` list —
one buffer per internal tensor *name* across every row count and key bucket (`:56-63`).

---

## 2. The 4-bit MoE decode GEMV, exactly as it is

The kernel family is `iq4q8_gemv` (22) → `…_pipe` (33) → `…_packed` (35) → `…_packed_pf` (54); the other three
(`_tile` 47, `_group` 53, `_pair` 67, `_rows` 28) are the same arithmetic with a different member tiling or a
grouped read order. `_packed_pf` is the production decode kernel (it is what
`hpu/deploy/strata-hpu-production.json` enables via `STRATA_HPU_Q8_PACKED` + `STRATA_HPU_GEMV_PF`), so it is the
one quoted below; the differences between the four are listed in §2.4.

### 2.1 What one member owns (the four numbers a port must keep consistent)

`hpu/tpc/iq4q8_gemv.c:20`:

```c
const int tiles=rows/256, groups=tiles/5, blocks=columns/32, per_part=blocks/parts;
```

* `rows` = output rows of one **expert slab**; `tiles = rows/256`; `groups = tiles/5` — i.e. an expert's rows are
  grouped in **1280-row super-tiles** (5 sub-tiles of 256). The glue enforces `rows % 1280 == 0`
  (`hpu/tpc/glue.cpp:935-936`).
* `blocks = columns/32` — the K axis in 32-column blocks; `parts` splits *those* between members.
* One member = **(slot, column part, group of five 256-row tiles)**
  (`hpu/tpc/iq4q8_gemv.c:2`, `hpu/src/resident_experts.cpp:187-193`), and the index space is
  `slots · parts · rows/1280` (`hpu/tpc/glue.cpp:943`).
* `parts` defaults: gate/up **2**, down **1** (`hpu/src/resident_experts.cpp:199`); a window may set its own
  (`:201`). Parts exist only to give the 24 TPCs more members; their F32 outputs are summed by the caller
  (`hpu/src/resident_experts.cpp:213-218`, `:228-232`).
* `x_group = per_row = top_k+1` (`hpu/src/resident_experts.cpp:205`): every `per_row` slots share one activation
  row, and the **last slot of each row is the shared expert** — `is_shared = slot % period == period-1`
  (`hpu/tpc/iq4q8_gemv_packed_pf.c:18`), which reads a separate one-expert slab at `erow=0` and is why the weight
  load takes a 5th dimension (`W=5`) for it.

### 2.2 The weight layout the kernel assumes (a real preshuffle)

One `U8` tensor per role: codes first, then F32 scales (`hpu/tpc/glue.cpp:938` —
`bytes = r/2*c + r*(c/32)*4`). Flattened, with `B = columns/32`, `G = rows/1280`:

```
codes  : (((g*B + b)*16 + p) * 5 + t) * 256 + L        p = column pair, t = 256-row sub-tile, L = byte
scales : codes_bytes + (((g*B + b)*5 + t) * 1024 + r*4)
```

`L` is **not** the row: `L = 4*(i%64) + i/64` for row `i` inside the sub-tile — the lane swizzle that makes the four
int32 sub-vectors of an accumulator register hold four *contiguous* 64-row blocks
(`hpu/tpc/iq4q8_gemv.c:7-9` states it, `hpu/src/resident_experts.cpp:44` implements it,
`hpu/src/resident_experts.cpp:48-49` writes both planes). This is exactly the "layout preshuffle for TPC vector
loads" that architecture decision B permits: the checkpoint's `[N, K/2]` row-major nibbles **cannot** be read with
the 256-byte vector that this kernel wants, because that vector runs along **rows**, not along K.

Inside a byte, the **low nibble is the even column** of the pair and the high nibble the odd one — that is what makes
`STEP`'s `w0` multiply `x0` (column `c`) and `w1` multiply `x1` (column `c+1`)
(`hpu/tpc/iq4q8_gemv_packed.c:34-36` and `:42-47`; the packer agrees: `codes[32*b+2*p]` is the low nibble at
`hpu/src/resident_experts.cpp:48`).

Scales are **one F32 per (row, 32-column block)**, stored tile-major so that the four 64-row chunks of a sub-tile are
four consecutive 256-byte loads (`hpu/tpc/iq4q8_gemv.c:10`).

### 2.3 Activation packing

`x` is not BF16: it is Q8_1 codes packed for broadcast (`STRATA_HPU_Q8_PACKED`, 34).

* `x   {256, columns/16, xrows} U8` — per 32-column block **two** 256-byte vectors, each carrying 16 column codes
  replicated in every 16-lane group, so a column is broadcast with one `v_u8_shuffle_b` instead of a 256-byte load
  (`hpu/tpc/glue.cpp:939` — `(kernel_id==35 || kernel_id==47 || kernel_id==54)?c/16:c`;
  `hpu/tpc/iq4q8_gemv_packed.c:24-26`, `:34-36`).
* `dx  {64, blocks, xrows} F32` — one block scale per (block, row), broadcast to 64 lanes
  (`hpu/tpc/iq4q8_gemv_packed.c:21-22`, `:79-86`).
* Quantiser: `q8_quant_packed` (34), one scale per 32 columns, codes replicated for the shuffle
  (`hpu/src/resident_experts.cpp:152-157`).

### 2.4 The pipelined load structure

Per column pair (2 columns) the member issues **five 256-byte weight loads** — one per 256-row sub-tile — and the
`packed_pf` variant issues the *next* pair's five loads before consuming the current pair's
(`hpu/tpc/iq4q8_gemv_packed_pf.c:29-44`):

```
            pair c/2            pair c/2+1
          ┌─────────┐         ┌─────────┐
load v0 ──┤ tile 0  │         │ tile 0  │ ...  (256 rows each, codes of both columns)
load v1 ──┤ tile 1  │         │ tile 1  │
load v2 ──┤ tile 2  │         │ tile 2  │
load v3 ──┤ tile 3  │         │ tile 3  │
load v4 ──┤ tile 4  │         │ tile 4  │
          └─────────┘         └─────────┘
             issue 5             issue 5      ← n0..n4 shadow registers
             MAC 5 × (w0·x0 + w1·x1)          ← v0..v4 consumed
```

The four variants differ only here:

| variant (id) | member | load scheduling |
|---|---|---|
| `iq4q8_gemv` (22) | (slot, part, 5-tile group) | one load then its MAC (`:41-46`) |
| `iq4q8_gemv_pipe` (33) | same | pair's 5 loads + 2 x loads issued before the MACs (`:25-40`) |
| `iq4q8_gemv_packed` (35) | same | packed x; optional `use_pf=0` branch (`:28-30`, `:49-78`) |
| `iq4q8_gemv_packed_pf` (54) | same | **next pair's 5 loads in flight** (`:29-44`) — bit-identical to 35 (`:1-5`) |
| `iq4q8_gemv_tile` (47) | one 256-row tile | one accumulator, eight loads in flight |
| `iq4q8_gemv_group` (53) / `_pair` (67) | (part, tile, slot position) | slots reordered by routed expert, so an expert's rows are read once per window |

### 2.5 The lines that implement the 4-bit multiply — and the only lines NVFP4 must replace

These five statements are the entire codec of the kernel. Everything around them is addressing, load scheduling and
the store, and is reusable verbatim.

```c
/* hpu/tpc/iq4q8_gemv.c:41-46 — STEP */
const uchar256 v = v_u8_ld_tnsr_b(at,W); at[0]+=256;                 /* 256 rows of 2 packed columns */
const char256 w0=(char256)v_u8_shuffle_b(lut, v|0x80, 0, lut);        /* low nibble  = column c   */
const char256 w1=(char256)v_u8_shuffle_b(lut,(v>>4)|0x80,0, lut);     /* high nibble = column c+1 */
ACC=v_i8_mac_b(w1,x1, v_i8_mac_b(w0,x0, ACC, 0,1,0), 0,1,0);          /* int8 dot -> int32, exact */
```

```c
/* hpu/tpc/iq4q8_gemv.c:49-57 — SCALE (one F32 block scale per 64-row chunk, times the activation scale) */
int5 sp={codes_bytes+((group*blocks+b)*5)*1024,erow,0,0,0};
const float64 s0=(float64)v_u8_ld_tnsr_b(sp,W); sp[0]+=256;   /* 4 x 256 B = 256 rows' F32 scales */
...
partial[T][k]=v_f32_mac_b(v_convert_i32_to_f32_b(ACC.vK,0)*sK, dxv, partial[T][k]);
```

The `lut` is a 256-byte vector holding the 16 IQ4 codebook values in every lane (`hpu/tpc/iq4q8_gemv.c:12`,
`hpu/tpc/iq4q8_gemv.c:13` explains the `|0x80` shuffle-mask rule: bits 0-4 index, bit 5 swaps the group, bit 7 must
be set). Scale bytes are widened by a *reinterpret*, not a conversion: `(float64)v_u8_ld_tnsr_b(...)` loads 256 bytes
= 64 F32 (`hpu/tpc/iq4q8_gemv.c:51-52`).

**For NVFP4, the formula to substitute is exactly this pair of statements, and only its *scale granularity* changes:**

| IQ4_XS (STRATA) | NVFP4 (ours) |
|---|---|
| code → value: 16-entry int8 codebook (`−127…113`) | code → value: 16-entry E2M1 LUT (`0,1,2,3,4,6,8,12,0,−1,…`) — **also 16 entries**, so the shuffle-LUT trick survives |
| one F32 scale per **32** columns, applied **after** the 32-value int8 dot | one **E4M3** scale per **16** columns, applied **inside** the K loop |
| activation is int8 (Q8_1), so the dot is an exact int8 MAC | activation is **BF16** (W4A16, decision C), so the dot must be F32 |
| per-expert scale fused into `d` | separate F32 `scale2[expert]`, **once after the whole K reduction** (decision B) |

So the replacement is:

```
for each 16-value group g of row r:
    d   = e4m3_to_f32(scales[r,g])            /* 0x00 and 0x7F decode to 0, see below */
    acc = Σ_{j<16} e2m1[nibble(r,16g+j)] * x[16g+j]        /* F32, W4A16 */
y[r] = scale2[e] * acc                                     /* once, then BF16 store */
```

Two traps, both already measured in this repo (`tools/check_fp4_lut.py`):

1. **`packed_ue4m3` is not E4M3FN.** STRATA's GGML-40 reference decoder
   (`hpu/tpc/packed_matmul.c:469-477`) ignores bit 7 (it decodes magnitudes only), maps `0x00` and `0x7F` to 0, and
   uses `2^(exp−8)` — half the standard value — because GGML's NVFP4 block folds that half into its own `d`
   (`hpu/tpc/packed_matmul.c:472` comment). Our CSF planes are signed E4M3 with `2^(exp−7)`
   (`vllm-gaudi/vllm_gaudi/ops/modelopt_csf.py:184-196`). Copying it verbatim halves every group scale and drops
   the sign of 126 codes.
2. **`packed_nvfp4_dot`'s nibble order is not ours.** GGML-40 splits each 16-value group as
   low-nibbles `0..7` then high-nibbles `8..15` (`hpu/tpc/packed_matmul.c:485-489`); our checkpoint (and our kernel)
   store consecutive K in consecutive nibbles, low nibble first. The *structure* of `packed_nvfp4_dot` is the right
   reference; its index arithmetic is not.

The doubled E2M1 table that makes the int8 path possible is `hpu/tpc/fp4_values.inc:8`; if a profile ever shows the
F32 multiply is the bottleneck, `Σ e2m1·xq` can be done as an int8 MAC against Q8 codes and multiplied by
`(d/2)·dx` per 16 — the same shape as the IQ4 kernel, at W4A8 precision instead of the contract's W4A16.

### 2.6 Our kernel, side by side

| | STRATA `iq4q8_gemv_packed_pf` | ours `qwen4exp_nvfp4_moe_gemv_bf16_gaudi2` |
|---|---|---|
| member | (slot, column part, 1280 rows) | (slot, 4 output rows) — `…gemv_bf16_gaudi2.c:30`, `:93` |
| index space | `slots·parts·rows/1280` | `(N/4, S)` — two-dimensional, slots on dim 1 |
| vector axis | 256 **rows** × 2 columns per load | 512 **K** values of 1 row per load (`:31`, `:86`) |
| preshuffle | required (lane `4*(i%64)+i/64`) | **none** — identity layout (`vllm-gaudi/vllm_gaudi/ops/modelopt_csf.py:692-717`) |
| code → value | shuffle-LUT int8 | `nvfp4_mul_bits` → BF16 bit pattern, exact (`:46`) |
| dot | `v_i8_mac_b` → int32 | `v_bf16_mac_acc32_b` → F32 (`:48`) |
| group scale | F32 per 32, after the dot | E4M3 per 16, **inside** the dot (bit-built into the weight) |
| per-expert scale | folded into `d` | `scale2`, once after the reduction (`:64`) |
| output | `{rows, slots, parts}` F32, host sums parts | `{N, S}` BF16 (`:70-71`) |
| slots per launch | ≤ 90112 (`kMaxMoeSlots`) | **≤ 64** (`host/…gemv_bf16_gaudi2.cpp:73`) |
| shared expert | last slot of each row, same launch | separate module, added after (`model.py:218-219`) |

**The arithmetic cost of the different vector axis, for one 2560×2560 projection (gate or up), one slot:**

* ours: `N/4 = 640` row groups × `K/128 = 20` K parts = **12,800 activation loads of 256 B = 3.28 MB** — the same
  bytes as the weights, so the projection reads 2× its minimum;
* STRATA: a member covers 1280 rows and its part of K (40 blocks of 32 columns); per block it issues two 256-byte
  code vectors plus one 256-byte `dx` broadcast = 768 B, so `40 × 768 B` = **30.7 KB per member, ~61 KB per slot**;
* ratio **≈53×** in activation-load bytes. The per-slot activation row is 5 KiB and should stay in L2, so the real
  cost is most likely load-issue/L2 bandwidth rather than HBM — but that is an assumption, not a measurement
  (no device here), which is why §3.2 keeps it as a "measure first" item.

**The scale plane, for the same projection:** our kernel loads 256 B per (row, 512-K tile) and uses 32 bytes
(8 per 128-K part × 4), i.e. **8× the bytes it consumes**; the surplus is the neighbouring rows' scale bytes, so HBM
traffic is probably fine but the load count is one per 32 scales. STRATA's layout gives one 256-byte load per 64
scales × 4 chunks, i.e. one load per 64 rows of one block — the same bytes, fewer issues.

---

## 3. Mapping table and gaps

### 3.1 Artifact → artifact

Status: **have** = our port implements the same function; **partial** = implemented with a different granularity or
in torch; **missing** = no equivalent.

| STRATA artifact | what it is | our artifact | status |
|---|---|---|---|
| `hpu/tpc/iq4q8_gemv*.c` + `hpu/src/resident_experts.cpp:136-246` | decode MoE GEMV (4-bit resident, no BF16 copy) | `native/…/kernels/qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.c`; `packed_moe.py:491-531` | **have**, different tiling (§2.6) |
| `hpu/tpc/q8_quant_packed.c` (34) | activation quantiser for the GEMV | none (BF16 activation, decision C) | **n/a** — W4A16 |
| `hpu/tpc/route_topk.c` (24) | device top-k + softmax routing | `packed_moe.py:278-289` (torch) | partial |
| `hpu/tpc/moe_mme_count.c` (72), `_prefix.c` (73), `_scatter.c` (74) | prefill bucketing | `native/…/count`, `prefix`, `scatter` | **have** |
| `hpu/tpc/moe_mme_place.c` (93), `_tilemap.c` (76), `u4_bucket_blocks` (62) | bucketing helpers (vector place, tile map, stable blocks) | none | missing |
| `hpu/tpc/iq4_expand_tiles.c` (77/81) + `hpu/src/moe_mme.cpp:88-158` | prefill B-operand expansion + 2 GEMMs + act | `packed_moe.py:242-250` (`dequant` + torch bmm) | partial (no MME path) |
| `hpu/tpc/moe_mme_join_perm.c` (71) / `u4_moe_join.c` (60) | inverse permutation of the tile order | `packed_moe.py:319-393` (gather/scatter in torch) | partial |
| `hpu/tpc/u4_moe_gu_epi.c` (58) / `u4_moe_down_epi.c` (59) | affine epilogues for the U4 MME path | none | missing (not on our path) |
| `hpu/tpc/moe_act_packed.c` (38) | fused GU part-sum + SiLU·up + quantise | none (3 torch ops) | missing |
| `hpu/tpc/hc_write_moe.c` (42), `hc_write_moe_routed.c` (88) | route weight + expert sum + HC write in one node | `model.py:218-219` + torch | missing |
| `hpu/src/qsa_block.cpp:107-155`, `hpu/tpc/qsa_select_mask.c` (91), `qsa_gather_fp8` (95) | device QSA select/gather | `qsa.py`, `qsa_math.py` (torch) | partial |
| `hpu/src/qsa.cpp:11-57` | FP8 KV cache + scale slots | `qsa.py` (torch KV cache) | partial |
| `hpu/tpc/gdn_step*.c`, `gdn_conv*`, `gdn_prepare*` | GDN kernels | vllm-core `Qwen3Next*` + `model.py` | partial (SDK ops) |
| `hpu/src/gdn_chunked.cpp` | chunked prefill recurrence (MME) | none | missing |
| `hpu/tpc/ple_gate*.c` (7/78/103), `ple_conv*.c` (8/104) | PLE gate (SiLU-of-sqrt-sigmoid) and 4-tap conv | `ple_runtime.py:227-324` (torch) | partial |
| `src/kernels/ngram.cpp:72-98` + `src/kernels/ngram.cpp:217-545` | PLE hash + host row reader | `ple_runtime.py:374-435`, `ple_embedding.py:47-129` | partial (no reader/prefetch, §4) |
| `hpu/include/strata/hpu/decode_chain.hpp` | decode chain / workspace sharing | none (per-op launches) | missing |
| `hpu/src/linear_graph.cpp:6-13` | lm head BF16 GEMM | `model.py` head | have (SDK) |
| `hpu/src/expert_scales.cpp` | per-expert F32 `.scale` | `modelopt_csf.py:323-355` (`detect_nvfp4_scale`) + `packed_moe.py:99-115` (`_fold_scale2`) | have |
| `hpu/tpc/packed_matmul.c:463-496` | scalar NVFP4 dot (GGML-40) — a *reference*, not a fast path | `modelopt_csf.py:460-493` (`dequant_nvfp4`) | have (reference) |

### 3.2 The gaps, concretely

| # | gap | evidence | severity | blocks |
|---|---|---|---|---|
| G1 | decode GEMV is capped at **S ≤ 64 slots** and 4 rows/member | `native/…/host/…gemv_bf16_gaudi2.cpp:73` | high | a 4-row DFlash verify (44 slots at top_k=10+1) just fits; a 16-row window (176 slots) or any prefill needs a new kernel |
| G2 | no 1280-row member ⇒ ≈53× STRATA's activation-load bytes per slot (3.28 MB vs ~61 KB for a 2560² projection) | §2.6 arithmetic | medium (measure) | decode load-issue; HBM impact depends on the 5 KiB row staying in L2 |
| G3 | scale plane read 8× its used bytes per load | §2.6 | low/medium | decode load count; a 32-byte scale load would need the row-tiled layout |
| G4 | no expert-grouped read (`_group` 53 / `_pair` 67) | `hpu/src/resident_experts.cpp:169-186` | medium | verify windows re-read each expert once per slot |
| G5 | shared expert is a separate module + extra launch | `model.py:218-219`, `is_fused_shared_expert_enabled` returns `False` (`model.py:223-224`) | medium | one launch and one `[10240,2560]` read per layer per token |
| G6 | no prefill MME path (bucketing only) | `native/…/README.md` "What is not covered" | high | prompt throughput; the bucketing exists precisely to feed it |
| G7 | no fused GU epilogue / act+quant (38) | mapping table | medium | 3 extra launches and a `[slots,2·ff]` F32 round-trip per layer |
| G8 | no fused MoE tail + HC write (42/88) | mapping table | medium | route weight, sum and HC write as separate ops |
| G9 | no device QSA select/gather, no FP8 KV cache | `qsa.py` | medium | long-context prefill and 4× KV HBM |
| G10 | no chunked GDN prefill | `hpu/src/gdn_chunked.cpp` | medium | prefill of 36 GDN layers is per-row |
| G11 | PLE has no reader/prefetch/budget (host shards + pinned 26.8 GiB) | `ple_embedding.py:47-113` | high | host RAM pressure and per-token latency; §4.5 |
| G12 | no decode chain / one workspace | `decode_chain.hpp` | medium | ~1 host wait per op instead of 1–2 per layer |

### 3.3 Look-alikes that are *not* gaps

* **No preshuffle today.** Our `kernel_layout()` is the identity (`vllm-gaudi/vllm_gaudi/ops/modelopt_csf.py:692-717`), which is correct for a
  kernel whose 256-byte vector runs along K. Decision B's "preshuffle" is only needed if we adopt STRATA's row-tiled
  member (G2/G3); it is a performance option, not a correctness one.
* **`scale2` once after the block sum** is *better* than STRATA's fused `d`: STRATA has no per-expert F32 scale to
  apply because its IQ4 codec has no `weight_scale_2`
  (`docs/NVFP4.md:116-124` gives the checkpoint's law: `h = silu(s_gate·(Gx))·(s_up·(Ux))`, `y = s_down·(Dy)`).
  Our kernel already implements it (`native/…/kernels/…gemv_bf16_gaudi2.c:64`).
* **No `dx` plane.** Because our activation is BF16, there is no activation scale to multiply; the `dx` load
  disappears with the Q8 codec.
* **INT4 vs NVFP4 *codes*.** STRATA's int8 MAC needs an int8 activation; our contract is W4A16, so the F32 MAC is
  not a "slower copy" of theirs — it is a different arithmetic mode that the decision already made.

---

## 4. PLE host streaming

### 4.1 The artifact (verified from the safetensors headers, not assumed)

Our checkpoint ships the table as **128 shards**, each `weight U8 [2,500,012 × 80]` + `weight_scale F8_E4M3
[2,500,012 × 10]` = 225,001,080 B/shard → **28,800,138,240 B = 26.82 GiB** for 320,001,536 rows — exactly STRATA's
`PLE_TABLE_ROWS` (`include/strata/kernels/ngram.hpp:46`). One row = 160 values = 80 packed bytes (2 E2M1 codes per
byte) + 10 E4M3 scales (one per 16 values) = **90 bytes**. §7 has the command.

### 4.2 The row law

16 rows per token = 2 n-grams (size 2 and 3) × 8 heads; each row `mixed % vocab[h] + offset[h]` with
`mixed = XOR_j(token[j]·mult[j])` in `uint64`; predecessors **oldest first**; the EOS cut propagates forward
(`src/kernels/ngram.cpp:72-98`, constants `include/strata/kernels/ngram.hpp:31-42`). The 16 rows flatten
**head-slowest** into 2560 values.

### 4.3 How STRATA keeps 48 GiB of table out of 96 GiB of HBM

1. **The table is never a device tensor.** Only `w_key`, `w_value`, the two norms and the conv kernel are uploaded
   (`hpu/src/hc_ple.cpp:132-150`; ~66 MB BF16), and only layer 1 consumes the embedding.
2. **Three residency modes** (`hpu/include/strata/hpu/ple_io_mode.hpp:47-52`): `direct` = bounded pread reader, no
   mapping at all (`src/kernels/ngram.cpp:314-329`); `mmap`; `ram` = mmap + multi-threaded touch + `mlock`
   (`src/kernels/ngram.cpp:355-385`) — the production config uses `--ple-io ram --host-limit-mib 65536`
   (`hpu/deploy/strata-hpu-production.json`).
3. **A byte plan before the reader opens** (`hpu/src/ple_source.cpp:44-118`): table payload + row cache + worker
   stacks + inflight queue are priced against the host limit and refused if they do not fit.
4. **Row-granular reader with a cache and a queue** (`src/kernels/ngram.cpp:322`): `max_inflight` reads
   (≤ 1024) over a `cache_rows` row cache (default 1M rows ≈ 90 MB at 90 B/row).
5. **Prefetch by token, not by row**: 16 slots, each holding a whole token's 16 rows
   (`src/kernels/ngram.cpp:217-221`, `:398-407`), and one reader request for a whole window
   (`hpu/src/model.cpp:1577-1581`) instead of one per row.
6. **Parallel decode**: the 16×16 rows of a window decode on 16 threads (`hpu/src/ple_input.cpp:118-127`).

The per-token cost is tiny by construction: **16 × 90 B = 1.44 KiB read, one 5 KiB BF16 row uploaded**. At 100 tok/s
that is 0.5 MB/s of PCIe — the risk is *latency and jitter* of 16 scattered 90-byte reads, not bandwidth.

### 4.4 What our port does now

`PLEShardTable` (`vllm-gaudi/vllm_gaudi/models/qwen4exp/ple_embedding.py:47-113`): 128 host tensors, `pin_memory()`
by default, `index_select` per shard, dequantise on the host, `.to(device)`.
`PLENGramEmbedding.lookup` (`vllm-gaudi/vllm_gaudi/models/qwen4exp/ple_runtime.py:408-435`): per token, group the 16
ids by shard (`torch.unique(shard)`), gather each group, concatenate, one H2D. Correct, and already "streamed per
token" in the sense that nothing is resident on the device.

What it lacks, against §4.3: no prefetch (the gather is synchronous inside the step), no row cache or dedupe across
tokens of a batch, no byte plan, and `pin_memory()` on 26.8 GiB of host RAM — pinning a whole table that the OS is
supposed to page is a host-lock hazard, not a speed-up.

### 4.5 The offloader a 187 GiB host should run, per step

Assume a host budget of 187 GiB, of which the table's 26.8 GiB may be resident (it fits seven times over) — the
design below is written so that it still works if only a fraction is resident.

| decision | what to do | why |
|---|---|---|
| **residency** | mmap the 128 shard tensors read-only, `MADV_RANDOM` (no readahead); optionally touch (16 threads) + `mlock` after admission, exactly STRATA's recipe at `src/kernels/ngram.cpp:355-385` | the table is 26.8 GiB and never written; readahead would evict the useful hot set |
| **granularity** | one row = 90 B, but never issue per-row syscalls: batch all rows of the step's tokens, sort by `(shard,row)`, coalesce contiguous runs | 16 random rows/token × 90 B = 16 page-cache misses; sorting turns them into ≤16 large-ish reads |
| **dedupe** | de-duplicate `(shard,row)` over the whole step *and* against the previous step's rows | the same n-gram rows recur across adjacent tokens; a 16-token step has ≤256 candidates, often far fewer |
| **prefetch** | run the gather for step *n+1* on a host thread while step *n*'s graph executes; ring of 16 token slots (STRATA's `kMaxPrefetch`, `src/kernels/ngram.cpp:217`) | hides the random-read latency behind a full decode step |
| **decode** | host side, 16 threads, E2M1 LUT × E4M3(per 16) → BF16 (or F32) `[T,2560]` | 2560 values/token is 2 µs of ALU; the device must not see the table |
| **transfer** | one pinned staging buffer per step, one H2D of `[T,2560]` BF16 (5 KiB/token) | per-row H2D would cost more than the reads |
| **HBM** | zero table residency; the only PLE tensors on the device are `w_key/w_value/norm_key/norm_query/norm_conv/conv1d` | decision E |
| **budget** | refuse to start unless `table + reader cache + worker stacks` fits the host limit (STRATA's `plan_ple_ram`/`plan_ple_reader`, `hpu/src/ple_source.cpp:92-118`) | a partial mmap that faults on the token path is worse than a clean refusal |
| **state** | one 9-row history per sequence (`include/strata/kernels/ngram.hpp:40`), on the device or host, not per token | the conv's history is 9 × 10240 F32 = 360 KiB per sequence |

### 4.6 Budget arithmetic (so the numbers are checkable)

* table: 128 × 225,001,080 B = 26.82 GiB (verified)
* row cache 1M rows × 90 B = 86 MiB; 16M rows = 1.34 GiB
* per step of 16 tokens: ≤ 256 rows = 23 KiB read, 80 KiB H2D
* per token: 1.44 KiB read, 5 KiB H2D, ~2 µs host decode + 1 gather

---

## 5. Launch-chain and workspace patterns adoptable as-is

None of these touches the weight codec, so all are available today regardless of how the NVFP4 planes are tiled.

1. **One stream for the whole token.** `DecodeChain` (`hpu/include/strata/hpu/decode_chain.hpp:14-37`): every layer
   graph queues on one stream; the host waits only where it must read a value (router logits, QSA projections).
   *Buys:* removes one round trip per layer. *Lands in:* the vLLM forward loop's per-layer Python call.
2. **The residual lives on the device, double-buffered.** `residual{2}` with `in()/out()/advance()`
   (`hpu/include/strata/hpu/decode_chain.hpp:26-36`). *Buys:* no `[HC,H]` copy per layer. *Lands in:* the HFN
   wrapper's residual handling.
3. **One workspace per chain, not per recipe.** `hpu/src/layer.cpp:32-34` — every row count and key bucket compiles
   its own recipe but they share `verify_chain` workspace (the widest needed). *Buys:* bucket count multiplies code,
   not HBM. *Lands in:* a single vLLM graph pool / workspace name for all buckets.
4. **Internal tensors allocated once per *name*.** `hpu/include/strata/hpu/decode_chain.hpp:56-63`: the six buffers
   `attn_write.out`, `attn.mixed`, `ffn.mixed`, `o.out`, `gdn.out`, `router.logits` are shared by every recipe.
   *Buys:* the same as 3, at tensor granularity.
5. **Weight leases released after the next wait.** `hpu/include/strata/hpu/decode_chain.hpp:18-19`, `:29`.
   *Buys:* streams expert slabs in/out of HBM without stalling (the stream orders the release). *Lands in:* the
   expert pack / residency manager (`packed_moe.py` + `expert-pack.bin`).
6. **Split a layer where a workspace would grow with the window.** `hpu/src/layer.cpp:39-42`: a window bucket ends
   the layer graph at the post-attention residual and runs **one** shared window-FFN launch for every layer, keeping
   one MME workspace per device. *Buys:* bounds MME scratch. *Lands in:* prefill buckets.
7. **Fused tails instead of extra nodes.** `moe_act_packed` (38), `hc_write_moe` (42/88), `row_norm_dual` (39),
   `hc_mix_dual` (101): each replaces 2–4 SDK nodes with one TPC launch and one fewer HBM round-trip. *Lands in:*
   the MoE/HC boundary; this is where our port has the most launches to save (G7, G8).
8. **Chain the head too.** `STRATA_HPU_CHAIN_HEAD` (`hpu/include/strata/hpu/performance_modes.hpp:122-126`): the
   final HC read and the LM head queue behind the last layer on the same stream, so the token costs one wait.
   *Lands in:* the sampler/head boundary.

Not adopted from STRATA, deliberately: the prefill MME path expands 4-bit codes to BF16 per window
(`hpu/src/moe_mme.cpp:88-95`). That is exactly the materialisation decision C forbids on the decode path; if a
prefill MME path is built for us it should expand to **FP8 per-channel** (decision D) or read the packed plane in the
GEMV, and the expansion must be priced per window, not per token.

---

## 6. The five call contracts

A downstream implementer must be able to obey these without re-deriving anything. Each is stated as
signature + invariant + the symptom of violating it (all five fail *silently* — in-range indices, plausible logits).

### C1 — Resident weight plane (NVFP4, in HBM, 4-bit forever)

```
packed  U8  [E, N, K/2]     torch; TPC sees [K/2, N, E]      byte q holds column 2q (low nibble), 2q+1 (high)
scales  U8  [E, N, K/16]    torch; TPC sees [K/16, N, E]     E4M3, one per 16 consecutive k
scale2  F32 [E]                                             one per expert, per projection
invariants: scales.dim0*8 == packed.dim0;  packed.dim0*2 == K;  K % 128 == 0;  K % 512 == 0 for the
            current kernel's 256-byte row loads (over-reads by <1 vector otherwise)
```
Anchors: `native/qwen4exp-nvfp4-moe/host/qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.cpp:79` (`scales.maxSizes[0] * 8 ==
k_half`), `:73` (slot cap), `vllm-gaudi/vllm_gaudi/ops/modelopt_csf.py:692-717` (identity layout).
**No consumer may materialise an expert as BF16 in the resident store**, and `scale2` multiplies the finished
reduction, never the individual weights. *Violation symptom:* decoded weights off by a factor (2× if the GGML UE4M3
half is copied in), or a plausible-but-wrong result where one nibble belongs to the neighbouring column.

### C2 — Decode GEMV call

```
custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2(ids[S] I32, packed[E,N,K/2] U8, scales[E,N,K/16] U8,
                                           x[S,K] BF16, scale2[E] F32) -> out[S,N] BF16
invariants: 1 <= S <= 64 (today); ids < 0 means "all-zero output row" (never an OOB read);
            per (slot,row) the MAC order is K ascending, then an F32 reduction, then scale2, then one BF16
            store — so the result is bit-reproducible; duplicate ids are separate slots and must not be
            merged into one output row.
```
Anchors: kernel `native/…/kernels/qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.c:74-140` (store at `:63-72`), host caps at
`…/host/…gemv_bf16_gaudi2.cpp:67-89`, Python wrapper `packed_moe.py:491-512`. *Violation symptom:* a zero row that
should be the sum of two duplicates, or a 65-slot window that silently reads past `ids`.

### C3 — Bucketing triple (prefill)

```
count(ids[S] I32, E)   -> counts[blocks,E] I32,  blocks = ceil(S/64)      ids < 0 skipped
prefix(counts)         -> offsets[blocks,E] I32, totals[E] I32
scatter(ids, offsets, E, S, cap>=S) -> bucket[E,cap] I32 (expert-major), place[S] I32 (-1 = skipped)
invariants: every routed slot lands in exactly one bucket row or is reported (place=-1 / overflow counter);
            bucket rows are a *dense prefix* per expert (a consumer may stop at totals[e]); E % 64 == 0;
            the shared-expert slot (slot % per_row == top_k) is never bucketed.
```
Anchors: `native/…/host/…count…cpp`, `…prefix…cpp`, `…scatter…cpp`; STRATA's F32 twin at
`hpu/tpc/moe_mme_count.c:1-5`, `hpu/tpc/moe_mme_prefix.c:1-4` and the *stable* block-skipping variant
`hpu/tpc/u4_bucket_blocks.c:1-3`. *Violation symptom:* a dropped slot (the token's expert contribution vanishes
without an error) or a bucket row that is not where `place` says it is.

### C4 — PLE row contract (host only)

```
per step:  rows[T,16] = ngram_rows(tokens, predecessors)            # src/kernels/ngram.cpp:72-98
           gather rows from the 128 host shards: 90 B/row (80 packed + 10 E4M3)
           decode to BF16 [T,2560], one pinned H2D per step
invariants: the table never becomes a device tensor; predecessors are oldest-first; the EOS cut is forward
            propagating and the token's own EOS does not cut its own context; the 16 rows flatten
            head-slowest; a row id outside [0, 320001536) is a hard error, never a clamp.
```
Anchors: `src/kernels/ngram.cpp:72-98`, `include/strata/kernels/ngram.hpp:31-52`,
`vllm-gaudi/vllm_gaudi/models/qwen4exp/ple_runtime.py:408-435`, `ple_embedding.py:82-129`. *Violation symptom:*
plausible embeddings from the wrong n-gram (the hash is order-sensitive and the fix is invisible in the logits).

### C5 — Chain and workspace call

```
one stream per decode chain; residual in two device buffers, swapped per stage;
one workspace name for every recipe of a chain (the widest needed); internal tensors bound by name;
weight leases released only after the next wait; one host wait per host-visible scalar.
invariants: a graph may not be launched on the chain stream after the host has read a value that the same
            stream produced unless the read is synchronised; a recipe compiled for one bucket must not be
            rebound to a different buffer shape without a re-check.
```
Anchors: `hpu/include/strata/hpu/decode_chain.hpp:14-37`, `:56-63`, `hpu/src/layer.cpp:32-34`,
`hpu/include/strata/hpu/performance_modes.hpp:31-35`. *Violation symptom:* missing synchronisation produces the
*previous* token's logits intermittently — the classic silent chain bug.

---

## 7. Verification log

Everything below was run on this host with no device, no `torch.hpu` initialisation and no HPU allocation.
Commands are quoted as run from the repo root.

### 7.1 The document's own anchors and inventory

```
$ .venv/bin/python validation/qwen4exp/tools/check_strata_map.py
anchors: 102 distinct path:line citations in STRATA_QWEN4EXP_MAP.md
critical: 55 pinned lines checked
all anchors resolve and every pinned line matches

$ .venv/bin/python validation/qwen4exp/tools/strata_module_index.py --section summary
## Summary
- STRATA: 108 TPC kernel sources, 112 registered kernels, 35 host modules (11596 LOC named in this inventory)
- ours: 4 native kernels + 5 glue files, 4 custom GUIDs, 18 Python modules
```

The 112-entry count and the kernel ids are derived from `hpu/tpc/glue.cpp` by registration order; the
`id==22`/`id==35`/`id==47`/`id==54`/`id==67` checks inside the glue agree with the derived ids, which is what makes
the appendix trustworthy.

### 7.2 Tests (CPU only)

```
$ .venv/bin/python -m pytest -q validation/qwen4exp/test_strata_map.py
5 passed, 14 warnings in 0.03s

$ .venv/bin/python -m pytest -q validation/qwen4exp
134 passed, 1 skipped, 14 warnings in 6.7s        # 129 + 1 before this track's 5 (see the note below)

$ bash validation/run_hpu_verify.sh --offline --skip-build
(c) offline pytest suites
232 passed, 14 warnings in 12.8s                  # final run of this session, 21:16
=== done (offline): native build + host-only suites passed
```

Notes, stated plainly:

* The brief's "163" does not match this tree and did not before this track started: `run_hpu_verify.sh` runs four
  paths (`validation/qwen4exp` plus `test_qwen4_exp_model.py`, `test_modelopt_csf.py`, `test_hpu_modelopt.py`) and
  the suite grows as tracks land. Measured while this map was written, in one session: **202** before these 5 tests (207 with
  them, 21:07), **230** at 21:13, **232** at 21:16 — the growth is other tracks adding tests and code concurrently
  (mtimes in `vllm-gaudi/tests/unit_tests/ops/` and `validation/qwen4exp/` moved at 21:09–21:16). The 5 tests of this
  track are a constant +5 and changed nothing else; the numbers above are a timestamped snapshot, not a fixed
  suite size.
* One transient red was observed and is recorded rather than hidden:
  `test_qwen4_exp_model.py::test_quant_dispatch_is_logged_per_prefix` failed at 21:11 on
  `"GDN linear attention -> …HPUGatedDeltaNetAttention"` while the log line named that test module's own
  `FakeGDN`; the file had been modified at 21:10:46 by the track that owns it, and it was green again at 21:13 with
  no change from this track (none of the files it imports were touched here).
* The new tests are stdlib-only: they read the two trees and this markdown, import no torch and touch no device.

### 7.3 Artifact facts quoted in the map, verified by command

```
$ # PLE table: 128 shards, one row = 80 packed bytes + 10 E4M3 scales
$ python3 -c "...read safetensors headers of tensors/model-00015-of-00021.safetensors..."
shard weight U8 [2500012, 80] scale F8_E4M3 [2500012, 10]
bytes/shard 225001080 total GiB 26.822 rows total 320001536

$ # original NVIDIA CSF export (real bytes for tests, architecture decision A)
$ python3 -c "...index of the HF snapshot..."
nvfp4_csf_* tensors: 147456 of 375487

$ # our checkpoint's tensor count (ModelOpt plain export)
checkpoint tensors: 301759
```

`26.822 GiB` and `320001536` are the two numbers §4.1 and §4.5 rely on; both come from the file headers, not from
the brief.

### 7.4 What could not be checked here

* **No device.** Nothing was timed. The "≈13× activation loads" and "8× scale bytes per load" in §2.6 are load-count
  arithmetic from the two kernels' source, not measurements; L2 residency of a 5 KiB activation row is an
  assumption a device profile must confirm (or refute).
* **No `tpc-clang` build in this track.** The four GEMV/bucketing kernels were already compiled by Track C
  (`build/native/*.so` exists and the runner loads it); this track only read their sources and the host caps.
* **No CSF byte-level re-decode in this track.** The CSF formula and the 147,456-tensor count are cited from the
  earlier Track B round and re-verified only as a count here; `vllm-gaudi/vllm_gaudi/ops/modelopt_csf.py` and its
  tests are the authority for the decode itself.
* **STRATA's numbers are from its own docs**, e.g. the 51.2e9-value PLE table and the NVFP4 measurements in
  `docs/NVFP4.md`; they were read, not reproduced.

---

## Appendix A — selected kernel ids

`strata_module_index.py --section kernels` prints all 112. The ones this map cites:

| id | guid | file |
|---:|---|---|
| 0 | `strata_gdn_step_f32` | `hpu/tpc/gdn_step.c` |
| 1 | `strata_gdn_conv_f32` | `hpu/tpc/gdn_conv.c` |
| 22 | `strata_iq4q8_gemv_f32` | `hpu/tpc/iq4q8_gemv.c` |
| 34 | `strata_q8_quant_packed_f32` | `hpu/tpc/q8_quant_packed.c` |
| 35 | `strata_iq4q8_gemv_packed_f32` | `hpu/tpc/iq4q8_gemv_packed.c` |
| 38 | `strata_moe_act_packed_f32` | `hpu/tpc/moe_act_packed.c` |
| 42 | `strata_hc_write_moe_f32` | `hpu/tpc/hc_write_moe.c` |
| 47 | `strata_iq4q8_gemv_tile_f32` | `hpu/tpc/iq4q8_gemv_tile.c` |
| 53 | `strata_iq4q8_gemv_group_f32` | `hpu/tpc/iq4q8_gemv_group.c` |
| 54 | `strata_iq4q8_gemv_packed_pf_f32` | `hpu/tpc/iq4q8_gemv_packed_pf.c` |
| 56-60 | `strata_u4_moe_*` | `hpu/tpc/u4_moe_*.c` |
| 67 | `strata_iq4q8_gemv_pair_f32` | `hpu/tpc/iq4q8_gemv_pair.c` |
| 71 | `strata_moe_mme_join_perm_f32` | `hpu/tpc/moe_mme_join_perm.c` |
| 72 | `strata_moe_mme_count_f32` | `hpu/tpc/moe_mme_count.c` |
| 73 | `strata_moe_mme_prefix_f32` | `hpu/tpc/moe_mme_prefix.c` |
| 74 | `strata_moe_mme_scatter_f32` | `hpu/tpc/moe_mme_scatter.c` |
| 77 | `strata_iq4_expand_tiles_bf16` | `hpu/tpc/iq4_expand_tiles.c` |
| 91 | `strata_qsa_select_mask_f32` | `hpu/tpc/qsa_select_mask.c` |
| 95 | `strata_qsa_gather_fp8` | `hpu/tpc/qsa_gather_fp8.c` |
| 103/104 | `strata_ple_gate_rows_f32` / `strata_ple_conv_rows_f32` | `hpu/tpc/ple_*_rows.c` |

---

## Strata FP8 output path analysis (Track 1)

Read for Track 1's F8-E4M3 output variant of the NVFP4 indexed dequant (Task 1, 2026-10-30). Every claim below is
anchored to the STRATA tree (`hpu/…`) or to this repo (`vllm-gaudi/…`); the Track-1 measurements were made with the
host **TPC simulator** (`libtpcsim_shared`, no device), the same harness `native/qwen4exp-nvfp4-moe/tests/moe_sim.cpp`
uses.

### The format and the intrinsic

* The output dtype of every STRATA FP8 kernel is **`DATA_F8_143` = E4M3, bias 7, 1+4+3 bits** — the same format as
  torch's `float8_e4m3fn`. Declared at `hpu/tpc/glue.cpp:452`, `hpu/tpc/glue.cpp:465`, `hpu/tpc/glue.cpp:479`.
  The mysterious note "TPC f8 is 143 … h8 targets 152" at `hpu/tpc/fp8_encode.c:22` is just these Synapse dtype
  names (143 = E4M3, 152 = E5M2), not rounding thresholds.
* The encoder intrinsic is `v_convert_f32_to_f8_all_b(x, SW_LINEAR)` (`hpu/tpc/fp8_encode.c:23`, `:28`,
  `hpu/tpc/moe_mme_gather_fp8.c:44`, `hpu/tpc/moe_mme_act_fp8.c:53`, `hpu/tpc/fp8_encode_fixed.c:14`) or
  `v_convert_bf16_to_f8_all_b(x, 0)` from a BF16 intermediate (`hpu/tpc/iq4_expand_tiles_fp8.c:59`). STRATA passes
  no rounding flag on the F32 path; the Gaudi kernels in our own tree use the explicit
  `SW_RHNE | SW_FP8_BIAS7 | SW_LINEAR` (`vllm-gaudi/csrc/deepseek_v4/include/deepseek_v4_qnorm_rope_kv_pack_bf16.h:33`,
  `vllm-gaudi/csrc/deepseek_v4/include/deepseek_v4_mxfp4_dequant_fp8.h:100`).

### Rounding mode

STRATA never asks the hardware convert to resolve a tie. Every FP8 producer **pre-scales so the row maximum lands at
code 240**, mid-binade, where RNE never matters:

* weight expansion: `scale[n] = max_b |s_b[n]| * 127/240` — "the row's largest code lands at 240"
  (`hpu/tpc/iq4_expand_tiles_fp8.c:1`, `:3`, computed at `:31`);
* activation gather: `scale_r = max|x| / 240` (`hpu/tpc/moe_mme_gather_fp8.c:1`, `:36`);
* act encode: `scale_r = max|act| / 240` (`hpu/tpc/moe_mme_act_fp8.c:3`, `:44`), `fp8_encode.c`'s dynamic scale
  `peak*(1/240)` (`hpu/tpc/fp8_encode.c:13`).

Where exact rounding *is* needed (the INT8 quantizer), STRATA does it by hand: `q8_quant.c` states the law
"round-half-away-from-zero" (`hpu/tpc/q8_quant.c:2`), then implements it as "F32 -> I32 rounds half to even; move
exact halves away from zero" with an explicit `+sign` on the 0.5 remainder (`hpu/tpc/q8_quant.c:17`–`:20`).

### Saturation — deliberately at ±240, never ±448

* `fp8_encode_fixed.c` is the explicit statement: inputs are clamped
  `min(max(x*16, -240), 240)` **before** the convert (`hpu/tpc/fp8_encode_fixed.c:9`–`:12`) with the header
  "Saturation is deliberate and must be checked by end-to-end fixed-input quality measurements"
  (`hpu/tpc/fp8_encode_fixed.c:2`).
* The dynamic kernels reach the same 240 ceiling by construction (the `127/240` and `1/240` factors above); the
  `+1e-30f` added to every scale (`hpu/tpc/iq4_expand_tiles_fp8.c:31`, `hpu/tpc/moe_mme_gather_fp8.c:36`,
  `hpu/tpc/moe_mme_act_fp8.c:44`, `hpu/tpc/fp8_encode.c:13`) only guards the all-zero row against a division by
  zero.
* **Why 240 (and not 448):** Track 1 measured the convert in the simulator — `v_convert_f32_to_f8_all_b` with
  `SW_RHNE | SW_FP8_BIAS7 | SW_LINEAR` is a faithful RNE only up to |x| ≈ 272; above that it **saturates its output
  at code 0x78 (±256) and can never emit 0x79..0x7E at all** (26k-value battery: every failure lies in
  |x| ∈ [272.9, 464]). STRATA's 240 target keeps every code a full binade below that wall. NaN handling: there is
  none — no STRATA FP8 kernel can produce the 0x7F/0xFF NaN codes because nothing survives past 240 (no `sel`/NaN
  path exists in any of the eight kernels read).

### Scale folding — never into the FP8 payload

Scales always travel as F32 side tensors and are multiplied back after the GEMM, in F32/BF16:

* expansion: `scale {rows, count} F32 … in natural row order` written per 64-row chunk
  (`hpu/tpc/iq4_expand_tiles_fp8.c:5`, `:35`–`:36`);
* activations: `a_scale {64, rows, count} F32 (lane-replicated)` so readers take it in one vector load
  (`hpu/tpc/moe_mme_gather_fp8.c:3`, `:37`; act twin at `hpu/tpc/moe_mme_act_fp8.c:5`, `:45`);
* the join undoes both scalings plus the weight scale in one F32 multiply:
  `routed[n] * act_scale[row] * s_down[tile][n]` (`hpu/tpc/moe_mme_join_fp8.c:2`, the store at `:38`).

`fp8_encode.c` additionally splits each activation into a **hi + lo/16 residual pair** (encode x, restore, re-encode
`(x-restored)*16`) for accuracy (`hpu/tpc/fp8_encode.c:1`–`:2`, `:23`–`:28`) — orthogonal to weight quantization.

The BF16 expansion twin (`iq4_expand_tiles.c`) shows the shared skeleton: i8 codebook lookup → `v_convert_i8_to_bf16`
→ one BF16 scale multiply → store (`hpu/tpc/iq4_expand_tiles.c:36`–`:39`, scale convert at `:25`).

### What Track 1 took, and what it had to invent

**Taken (structure only, per §8.2):** the F8_143 output dtype and `v_f8_st_tnsr_partial` stores; the "decode → scale
in high precision → convert → store" skeleton; the F32-sidecar scale discipline for the prefill GEMM epilogue.

**Not taken (arithmetic):** the dynamic `max·(127/240)` rescaling. Our kernel's values are *defined* by the
checkpoint (`e2m1 · e4m3fn · scale2[e]`) — rechoosing a scale at runtime would break the port's rule that the F8
plane equals the torch reference cast of the dequantized weights.

**Invented — the hybrid RNE encode** (`vllm-gaudi/csrc/qwen4exp/include/qwen4exp_nvfp4_indexed_dequant_fp8.h:45`–`:62`):
clamp to ±448 (the explicit-clamp precedent is `vllm-gaudi/csrc/deepseek_v4/include/deepseek_v4_qnorm_rope_kv_pack_bf16.h:297`–`:301`),
then take the *direct* convert below 256 and the *halved* convert (encode x/2, add 8 to the exponent field — the
`csrc/deepseek_v4` KV-pack helper at `vllm-gaudi/csrc/deepseek_v4/include/deepseek_v4_qnorm_rope_kv_pack_bf16.h:26`–`:44`)
for |x| ≥ 248, selected on the halved code ≥ 0x70. Both measured failure modes are covered: the direct convert's
0x78 ceiling (above) and the halved convert's wrong tie window when x/2 crosses the E4M3 subnormal boundary
(x ∈ [0.029296875, 0.0302734375), 45 mismatches vs torch in the battery). Result: bit-equal to
`torch.float8_e4m3fn` wherever that cast is finite (|v| ≤ 464; above, torch emits NaN codes while this kernel
saturates to ±448 by contract).

**Invented — the −0 sign fix** (`vllm-gaudi/csrc/qwen4exp/include/qwen4exp_nvfp4_indexed_dequant_fp8.h:78`–`:85`):
a bare TPC vector·scalar multiply flushes zero products to +0.0, losing `-0.0` weights; `v_f32_mac_b(w, s2, -0.0f, 0)`
keeps the IEEE product sign (`+0 + -0 = +0`, `-0 + -0 = -0`). Verified in the simulator: `mac(+0, -2.5, -0)` →
−0.0, `mac(-0, -2.5, -0)` → +0.0.

**Wiring:** host glue `vllm-gaudi/csrc/qwen4exp/host/qwen4exp_nvfp4_indexed_dequant_fp8_gaudi2.cpp` (F8_143 output
enforced at `:58`–`:60`), third custom GUID in `vllm-gaudi/csrc/qwen4exp/host/entry_points.cpp:79`–`:91` with dispatch
at `:105`, PT2 op `custom_qwen4exp_nvfp4_indexed_dequant_fp8_gaudi2` (schema at
`vllm-gaudi/csrc/qwen4exp/pytorch/hpu_qwen4exp_nvfp4_dequant_pt2.cpp:200`, `Float8_e4m3fn` meta at `:111`).

**Offline verification (no device):** `tpc-clang -Wall -Werror -march=gaudi2` clean; the real ELF run through the TPC
simulator is bit-exact against the torch cast on synthetic ragged/edge cases (K ∈ {128, 256, 1280, 2560}, ids −1 and
out-of-range, scale2 from subnormal-inducing to saturating) and on **real layer-0 expert-0 gate_proj bytes with the
real `weight_scale_2` (2 slots × 640 × 2560 = 3,276,800 elements, zero mismatches vs the pure torch cast)**;
CPU twin: `tools/qwen4exp_flashnext/test_nvfp4_tpc_op.py` (compile gate + Python bit-sim of both output paths).
