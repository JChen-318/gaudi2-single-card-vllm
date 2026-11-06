# Qwen4Exp → Gaudi2: engineering map extracted from STRATA

**Status:** host-only extraction. No device was touched, and **no source file in either tree was modified** — the deliverables are this map plus the host-only checkers under `validation/qwen4exp/tools/`.
**Sources of truth**

| tree | root | revision |
|---|---|---|
| STRATA reference engine | `$STRATA_ROOT` (default `~/gaudi2-single-card-strata`) | `874502d` |
| our port (this repo) | this repository | working tree |

**How to read the anchors.** Every `path:line` anchor is relative to one of the two roots above and is
machine-checked by `validation/qwen4exp/tools/porting_anchors.py` (see §8). Paths starting with `hpu/`, `src/`,
`include/`, `docs/` live in the **STRATA** tree; paths starting with `vllm-gaudi/`, `validation/` live in **this**
repo. Where an anchor is followed by a quoted snippet on the same line, the snippet is the text at that anchor.

**This revision** (reviewed against the sources line by line) corrects three claims and adds the pieces a kernel
implementer needs next: Strata's resident slab is **row-tiled**, so its 256-byte addressing is *not* transferable to
our `[K/2, N]` plane (§4.1, and the GEMV signature in §5.2 now shows the required dimension order); the D2/D3 width
arithmetic is enumerated rather than hand-derived (§1.4, `tools/check_selection_width.py`); and the `.venv` **does**
have pytest (§8.1). Added: the two device prep kernels (§1.2), the own-block delta D9, the resident **U4** store as
the closest analogue to our NVFP4 path (§4.4), the glue/ABI conventions (§5.2), chain scratch sharing (§6), and a
cross-tree proof that both artifacts carry the **same PLE hash constants** (§3.2, `tools/check_ple_hash_parity.py`).

**What we keep and what we do not.** Our checkpoint is NVFP4 (E2M1 nibbles, group-16 E4M3 scales, F32 `scale2`),
Strata's is IQ4_XS (4-bit codebook, per-32 6-bit scale). Hardware is identical (one Gaudi2 96 GiB) and the model
geometry is identical (`hidden=2560`, `hc_count=4`, `hc_lowrank=320`, `head_dim=256`, `num_experts=512`,
`num_experts_per_tok=10`, `moe_intermediate_size=640`, `indexer_n_heads=4`, `indexer_head_dim=128`,
`indexer_compress_ratio=4`, `indexer_budget=2048`, `ngram_size=3`, `heads_per_ngram=8`,
`split_ngram_parts=128` — `config.json`). So Strata's *skeletons* transfer 1:1, and only the weight/scale
codec, the KV dtype and a few selection bookkeeping constants have to change.

---

## 1. QSA (full-attention) indexer and sparse attention

12 of the 48 layers are `full_attention` (`config.json` `layer_types`). Each owns an indexer that scores
*blocks* of `compress_ratio = 4` cells and keeps `indexer_budget = 2048` cells.

### 1.1 The selection law, as the host states it

Pooling. A block's pooled key is the **mean of the block's raw index keys**, then an RMS norm with the index-key
gamma, then NeoX RoPE at the **block's first cell**:

- `hpu/src/qsa_state.cpp:143-150` — `std::vector<float> pooled_key(std::span<const double> raw,std::span<const float> gamma,float epsilon) {` — the norm is `1/sqrt(mean(x^2)+eps)` and the gamma is multiplicative (`x*inverse*gamma`), accumulated in `double`.
- `hpu/src/qsa_state.cpp:365-372` — `if(slot==c.block-1) {` — pooling happens when the block *completes*, over `tail_` (the `block-1` earlier raw keys) plus the newest raw key, divided by `block`.
- `hpu/src/qsa_state.cpp:372` — `rotate_neox(pooled,c.index_width,rope_row_at(c.positions,c.rotary,c.position_base+uint64_t(completed)*c.block,c.rope));` — RoPE position is `position_base + block_index*block` (block start), not the cell.
- `hpu/src/qsa_state.cpp:403-404` — `std::span<const float> QsaState::pooled_rows() const {` — the pooled rows kept for the device are exactly `(size/block + 1)` rows: every complete block plus the "spare" row of the block in progress.

Scoring. `relu(q·k)` summed over the indexer heads, F32 (host sums in double), no `1/sqrt(d)` on the host:

- `hpu/src/qsa_state.cpp:429-430` — `score+=std::max(dot(query.data()+uint64_t(head)*c.index_width,pooled,c.index_width),0.0);`
- `hpu/src/qsa_state.cpp:432` — `float value=float(score);if(block+1==blocks && size_%c.block) value+=1e9f;` — **the row's own unfinished block is forced first by +1e9**, and only when `size % block != 0`.

Top-k and the cut. Selection width is `min(cells, top_k + block - 1)`:

- `hpu/src/qsa_state.cpp:164-167` — `uint32_t QsaConfig::selection_width(uint32_t cells) const {` … `return uint32_t(std::min(uint64_t(cells),uint64_t(top_k)+block-1));`
- `hpu/src/qsa_state.cpp:458-472` — `const auto better=[&](int32_t a,int32_t b){return scores[a]>scores[b] || (scores[a]==scores[b] && a<b);};` — the total order is **(score desc, block id asc)**, and `nth_element`+`sort` is exact on it because it is a strict total order.
- `hpu/src/qsa_state.cpp:461` — `const uint32_t keep=std::min<uint32_t>(blocks,uint32_t((uint64_t(width)-1)/c.block+2));` — `ceil(width/block)+1` ranked blocks are enough, because only the last block can be short.

The device restates the same law at **cell** granularity (this is the one a TPC port must reproduce):

- `hpu/tpc/qsa_select_mask.c:50` — `const int forced=size%block?size-size%block:size;` — first cell of the row's own block (`forced == size` when the row sits exactly on a block boundary, i.e. no partial block).
- `hpu/tpc/qsa_select_mask.c:52` — `const int64 missing=size%block?(int64)(block-size%block):(int64)0;` — the forced block is **counted as `block` cells** in the budget but holds only `size-forced` of the row's cells.
- `hpu/tpc/qsa_select_mask.c:57-58` — `s=v_f32_sel_less_f32_b(first,(float)forced,s,s+1.0e9f);` then `s=v_f32_sel_less_f32_b(first,(float)size,s,-1.0f);` — blocks whose first cell is before `forced` get `+1e9`; blocks starting at or past `size` get `-1`.
- `hpu/tpc/qsa_select_mask.c:62-63` — `#define CELLS(N,PROBE) (v_f32_reduce_add(v_convert_i32_to_f32_b(N,0))*(float)block-` — cells above a threshold, minus `missing` when the forced block is above it.
- `hpu/tpc/qsa_select_mask.c:66-92` — the cut is an **8-way search on the F32 bit patterns** (all candidate scores are `>= 0`, so bit order == value order), 11 fixed passes over `[0, 0x7F800000]`, no vector→scalar transfer; a second 8-way search (`qsa_select_mask.c:103-137`) resolves ties by block id; the last block keeps only `rest` cells (`qsa_select_mask.c:145-153`).

### 1.2 TPC kernel contracts

Registration table (guid, #inputs, #outputs) — `hpu/tpc/glue.cpp:249-259`, e.g.
`hpu/tpc/glue.cpp:249` — `{"strata_qsa_index_cells_f32",strata_qsa_index_cells_elf_start,strata_qsa_index_cells_elf_end,6,1},`
and `hpu/tpc/glue.cpp:257` — `{"strata_qsa_gather_blocks_fp8",strata_qsa_gather_blocks_fp8_elf_start,strata_qsa_gather_blocks_fp8_elf_end,7,5},`
Index spaces are declared in `configure()` (`hpu/tpc/glue.cpp:323`); the table is append-only
(`hpu/tpc/glue.cpp:282` — `static_assert(std::size(kernels)==112,"kernel ids are append-only: configure() and base_kernel() index this table");`).

#### `qsa_index_cells` — pooled key of every block (`hpu/tpc/qsa_index_cells.c`)

| item | contract |
|---|---|
| guid | `strata_qsa_index_cells_f32`, 6 in / 1 out |
| inputs | `start` `{1}` F32 (runtime first cell); `pooled` F32 `{width, >= blocks}` (the host's rows); `tail` F32 `{width, block}` (raw keys before `start`); `window` F32 `{width, rows}`; `gamma` F32 `{width}`; `rope` F32 `{rotary, >= cells}`, per cell |
| output | `out` F32 `{width, keys/block}` |
| params | `QsaIndexCells{width, rotary, epsilon, keys, rows, block}` (glue `hpu/tpc/glue.cpp:1021-1033`) |
| index space | rank 1, `indexSpaceGeometry[0]=keys/block` — **one member per block** |
| guards | `width%64==0`, `128<=width<=65536`, `rotary%2==0`, `keys%block==0`, `keys>rows` |

Body: blocks before `start` are copied from `pooled` (`hpu/tpc/qsa_index_cells.c:20-25`); later blocks are pooled
from `tail`/`window` with a 64-lane mean (`hpu/tpc/qsa_index_cells.c:27-33`), summed squares → `inverse`
(`hpu/tpc/qsa_index_cells.c:36`), gamma on the non-rotary head (`:39-42`) and NeoX on the first `rotary` columns
(`hpu/tpc/qsa_index_cells.c:43-49`). Note the mean divides by `block` even for a short tail block
(`hpu/tpc/qsa_index_cells.c:33` — `OUT=sum_*(1.0f/(float)block);`).

#### `qsa_decode_blocks` — decode-time block scores (`hpu/tpc/qsa_decode_blocks.c`)

- `hpu/tpc/qsa_decode_blocks.c:15` — `for(int h=0;h<heads;++h) {int5 p={v*64,h,0,0,0};s+=v_f32_max_b(v_f32_ld_tnsr_b(p,scores),0.0f);}`
- `hpu/tpc/qsa_decode_blocks.c:17` — `s=v_f32_sel_less_f32_b(id,complete,s,-3.0e38f);` with `complete = cell/block` (`:11`).
- Contract: `scores` F32 `{blocks, heads}` + `cell` `{1}` F32 → `out` F32 `{blocks}`; `blocks%64==0`; rank 1 with
  `blocks/64` members of 64 blocks each (`hpu/tpc/glue.cpp:650-658`).
- `qsa_window_blocks` is the same kernel with a row axis: `scores {blocks, heads, rows}` → `{blocks, rows}`,
  rank 2 `(blocks/64, rows)` (`hpu/tpc/qsa_window_blocks.c:13-19`, glue `hpu/tpc/glue.cpp:543-551`).

#### `qsa_select_expand` / `qsa_select_bits` / `qsa_select_expand_fold`

- `qsa_select_expand` (`hpu/tpc/qsa_select_expand.c:8`) — `start` `{2}` F32 (window first cell, chunk first row),
  `kept` `{keys, rows}` F32 → `mask` `{keys, 1, rows}` F32, values `0` selected / `-3e38` masked
  (`hpu/tpc/qsa_select_expand.c:23`). `keys%64==0`, `block` a power of two ≤ 64, one member per row.
- `qsa_select_bits` is the same translation unit compiled with `STRATA_SELECT_BITS`
  (`hpu/tpc/qsa_select_bits.c:2-3`), emitting `bits {keys/block/16, rows}` F32 (16 blocks per word, exact in F32)
  plus `info {2, rows}` (cut block and its kept cells) — glue `hpu/tpc/glue.cpp:1044` (inside the id-91 branch, `:1034-1046`).
- `qsa_select_expand_fold` writes the SDPA's BF16 mask `{keys, group, rows}` directly, replicating each row's mask
  over the query-head group (`hpu/tpc/qsa_select_expand_fold.c:26-27`), so no F32 mask, cast and broadcast follow.

#### `qsa_gather_blocks_fp8` — decode gather (`hpu/tpc/qsa_gather_blocks_fp8.c`)

- Contract (glue `hpu/tpc/glue.cpp:659-670`): `keys, values` F8_143 `{256, heads, cells}`; `key_scale,
  value_scale` F32 `{32, cells}`; `top_ids` I32 `{top}`; `top_scores` F32 `{top}`; `cell` `{1}` F32 →
  `key, value` F8_143 `{256, n, heads}`, `key/value_out_scale` F32 `{n, heads}`, `mask` F32 `{n}`, with
  `n = (top+1)*block`. Rank 2, members are `(rank in [0,top], head)`.
- Index rules: `own=cell/block`, `before=cell-own*block`, `ranked=width_sel-1-before`
  (`hpu/tpc/qsa_gather_blocks_fp8.c:17-18`); rank `top` is the token's own block; a dropped cell reads the tensor's
  last valid cell so it decodes to zero with no branch (`hpu/tpc/qsa_gather_blocks_fp8.c:35`), and gets a `-3e38`
  mask (`hpu/tpc/qsa_gather_blocks_fp8.c:51`).
- `qsa_gather_window_bf16` is the verify-window twin: 16 ranks per member, block==4, FP8 read widened to BF16 in
  **even/odd channel order** (`hpu/tpc/qsa_gather_window_bf16.c:56-58`), output `{256, n, heads, rows}` BF16
  (glue `hpu/tpc/glue.cpp:625-635`). The query is permuted to match and the context back
  (`hpu/src/layer.cpp:790-794`).

#### `qsa_prep` / `qsa_index_prep` — norm + partial RoPE + the FP16 grid

These two are the device halves of the host steps our torch path does in `qsa.py`, and they pin the *arithmetic
mode* of the norms, which is where a port drifts without crashing.

- `hpu/tpc/qsa_prep.c:21-23` — `void main(tensor q,tensor k,tensor v,tensor q_gamma,tensor k_gamma,tensor rope, tensor query,tensor k_slot,tensor v_slot,tensor k_now,tensor v_now, int width,int query_heads,int kv_heads,int rotary,float epsilon,int tokens)`
  — one member owns **four heads of one kind of one token** (query heads in fours, then the key heads in fours,
  `hpu/tpc/qsa_prep.c:28-32`), so the four independent load/norm/RoPE chains overlap. Params are
  `abi::QsaPrep{width, query_heads, kv_heads, rotary, epsilon, tokens}` (`hpu/include/strata/hpu/model_abi.hpp:75`). The gammas are
  `{width}`, **shared by every head**, and read at row 0 (`hpu/tpc/qsa_prep.c:53-58`); the RoPE table is
  `{rotary}` per cell — `rotary/2` cosines then `rotary/2` sines (`hpu/tpc/qsa_prep.c:9-10`), so no lane shuffle
  is needed for the NeoX pair.
- Two numerics a port must reproduce: the RMS sums run in **F32 trees** while the host sums in `double`
  (`hpu/tpc/qsa_prep.c:11`), and K/V are rounded to the **FP16 grid on the device** by a Veltkamp split —
  `hpu/tpc/qsa_prep.c:12-20` — instead of through an FP16 register. Our cache stores "FP16-rounded F32"
  (`vllm-gaudi/vllm_gaudi/models/qwen4exp/qsa.py:107-113`); the two agree only if we round to the same grid.
- `hpu/tpc/qsa_index_prep.c:6` — `void main(tensor raw,tensor gamma,tensor rope,tensor out,int width,int heads,int rotary,float epsilon,int rows,int gamma_heads)`
  — the indexer query rows: per (row, head) an RMS norm with a gamma that is **shared or one row per head**
  (`hpu/tpc/qsa_index_prep.c:13` — `const int g0=gamma_heads==1?0:head*width;`), then NeoX on the first `rotary`
  columns at the row's position. `{width, heads, rows}` → same shape, members are rows; `gamma_heads` must be
  `1` or `heads` (`hpu/tpc/glue.cpp:1015-1020`).

#### `qsa SDPA` usage

- `hpu/src/layer.cpp:392-397` — `ns_Sdpa::ParamsV2 a{};a.scale=1.0f;a.is_causal=false;` builds the fused SDPA and calls the guid `sdpa_recomp_fwd_bf16`; **the scale is applied to the query, not the scores** (`hpu/src/layer.cpp:483` — `g.multiply(query,scale,scaled_query);`).
- Shapes: query `{t, hq, d}` F32, key/value `{1, hk, K, d}` BF16, mask `{t, 1, K}` F32 → `{t, hq, d}` F32
  (`hpu/src/layer.cpp:389-391`). `d` must be 256 for the QSA paths (`hpu/src/layer.cpp:239`).
- Two layout tricks, both worth copying: (a) *GQA fold* — the query rows become `(row, head-of-group)` so the
  SDPA's head axis is the **KV head** and each group member becomes an extra row
  (`hpu/src/layer.cpp:410-427`); (b) *batch fold* — `q {t,hk,group,d} -> {group,hk,t,d}`
  (`hpu/src/layer.cpp:398-408`).
- Chunking of the selection prefill is bounded by the indexer planes, not the attention planes
  (`hpu/src/layer.cpp:521-531`): `const uint64_t per_row=sdpa?keys*4*8:...` then halve until ≤ 256 MiB.

### 1.3 What our torch QSA already does

- Indexer projections and the two layernorms live in `vllm-gaudi/vllm_gaudi/models/qwen4exp/qsa.py:115-163`
  (`class QSAIndexer`, `project_qk`), the selection math in `qsa_math.py`:
  - `vllm-gaudi/vllm_gaudi/models/qwen4exp/qsa_math.py:199-218` — `pool_indexer_blocks`: float mean over the block, `gemma_rmsnorm` (i.e. `*(1+w)`), partial RoPE at the block start. **Same law as `pooled_key` + `rotate_neox`**, computed in torch.
  - `vllm-gaudi/vllm_gaudi/models/qwen4exp/qsa_math.py:240` — `scores = torch.relu(dots).sum(dim=1) / math.sqrt(head_dim)` — relu-sum over heads, **divided by sqrt(128)**.
  - `vllm-gaudi/vllm_gaudi/models/qwen4exp/qsa_math.py:253-255` — `order = torch.argsort(-scores, dim=-1, stable=True)` + `picked = order[:, :k]` — (score desc, id asc), k = `token_budget // ratio` = **512 complete blocks**.
  - `vllm-gaudi/vllm_gaudi/models/qwen4exp/qsa_math.py:189-195` — the row's own partial block is **appended after** the selected blocks: `out[rows[tail_ok], tail_cols[tail_ok]] = tail_pos[tail_ok]`.
  - `vllm-gaudi/vllm_gaudi/models/qwen4exp/qsa_math.py:150` — `width = token_budget + ratio - 1` (2051), the packed width the attention consumes.
- Attention: `qsa_math.py:361-411` (`sparse_attention`) gathers the selected cells per row and runs BF16/fp32
  torch matmuls, with an exact fp32 path below a 256 MiB plane.
- The K/V and raw index keys live in a per-slot software cache in torch (`qsa.py:107-113`, `_QSASlot`), padded
  FP16-rounded, **not** an FP8 cache.

### 1.4 Delta (torch → TPC), precisely

| # | item | Strata | ours | action |
|---|---|---|---|---|
| D1 | score scale | none (`qsa_state.cpp:429-430`) | `/sqrt(128)` (`qsa_math.py:240`) | **drop it**; a positive constant cannot change the ranking, and ties keep the same order because both use (score desc, id asc). Keep it only if we also port `block_scores` verbatim. |
| D2 | budget accounting | the forced block is counted **inside** `width_sel` (`qsa_select_mask.c:50-52`) | 512 complete blocks **plus** the tail cells | the two differ by up to one block. See below. |
| D3 | forced-first | `+1e9` folds the row's own block into the ranking (`qsa_state.cpp:432`) | the tail is appended after the ranked blocks (`qsa_math.py:189-195`) | for bit-parity, port `qsa_select_mask.c` **with the `+1e9` removed** and keep the tail inside `size<=width_sel`'s branch; see below. |
| D4 | KV dtype | FP8 E4M3, per-(cell,head) scale `max\|x\|/240`, 32-slot rows for 128-byte alignment (`hpu/include/strata/hpu/qsa.hpp:19-28`, `hpu/src/qsa.cpp:29-57`) | FP16-rounded F32 | stage 1 keeps FP16/BF16 (gather kernels must then be BF16 variants, as `qsa_gather_window_bf16.c` already is); FP8 is a separate, measurable step. |
| D5 | gamma | multiplicative, "not raw (1+w)" (`hpu/include/strata/hpu/qsa_block.hpp:13-15`) | `gemma_rmsnorm` multiplies by `1+w` (`qsa_math.py:43-47`) | fold `1+w` into the gamma tables on the host before binding; the kernels then need no change. |
| D6 | RoPE | NeoX, pair `i` with `i+rotary/2`, position = row's cell (`hpu/tpc/qsa_index_prep.c:21-28`) | same convention, M-RoPE sections `[11,11,10]` (`config.json`) | identical for text (`t==h==w`); only vision needs the sector table (`qsa_state.hpp:18-32`). |
| D7 | selection kernel count | 1 launch for scores+cut, 1 for mask | ~6 torch ops | the cut search is the piece worth porting: it is branch-free, 11 fixed passes, no scalar readback. |
| D8 | where selection runs | device, `start`/`cell` are **runtime scalars in a table** (`hpu/src/layer.cpp:232` — `dsel.cell {1} is the token's cell (bound into a table of cell numbers)`) | host Python + `.item()` calls | removing the `.item()` syncs is the decode-latency win. |
| D9 | *which* blocks may be ranked | device path ranks **only blocks strictly before the query's own** — `const float complete=(float)(cell/block);` (`hpu/tpc/qsa_decode_blocks.c:11`) — then forces the own block at rank `top` with `count=before` cells (`hpu/tpc/qsa_gather_blocks_fp8.c:22-28`); the **host** path instead ranks `ceil(size/block)` blocks and forces the last only when it is incomplete (`hpu/src/qsa_state.cpp:414,432`) | ranks every *complete* block (own included) and appends the incomplete tail; when `size % block == 0` nothing is forced (`vllm-gaudi/vllm_gaudi/models/qwen4exp/qsa_math.py:161-195`) | **Strata's own two paths disagree here**, once every `size % block == 0` (= one position in four). Ours matches the *host* path. A kernel port must take the `ceil` cut from `qsa_state.cpp:414`, not the `floor` cut from `qsa_decode_blocks.c`, or it will force a block the reference does not. |

**D2/D3 algebra (corrected — the earlier form of this paragraph had an arithmetic slip; these numbers are
enumerated, see §8.1).** With `block = 4`, `top_k = 2048`, `r = size mod block` and `size` the row's causal length:

| law | cells kept, long rows | what the cut counts |
|---|---|---|
| Strata host `QsaState::select`, `selection_width = min(cells, top_k+block-1)` = 2051 | **2051** | *cells*; the forced/partial own block is inside the budget |
| Strata device `qsa_select_mask.c` with `width_sel = 2051` (`ranked = width_sel-1-before`, `+ before` own cells, `+1` self) | **2051** | *cells* |
| ours (`k = token_budget // ratio = 512` complete blocks, tail appended after) | **2048 + r** | *blocks* |

So ours is **never wider** than Strata's, and it coincides only at `r == 3`; the gap is at most **3 cells**
(=`block-1`), not a block. The two are *different functions of the scores*, not two settings of one knob: ours
cuts on a **block count** (512 whole complete blocks — `qsa_math.py:151` `block_topk = token_budget // ratio`,
`qsa_math.py:253-255`), Strata's on a **cell count** (2051). Reproducing ours therefore means passing
`block_topk = 512` and counting *blocks* in the cut, then appending the row's incomplete block unconditionally;
`width_sel` + the `+1e9` term are the wrong parameterisation for it. The mask kernel can keep Strata's overall
shape (`qsa_select_mask.c:50-63`) if the `CELLS()` macro is replaced by a block counter — at which point the
`+1e9` / `missing` machinery (`qsa_select_mask.c:33-34,50-52`) disappears with it.

---

## 2. Hyper-connections (`hc_count = 4`)

### 2.1 The math

Four residual streams of width `hidden = 2560`; `rank = hc_lowrank = 320`.

**Read** (per token): normalize the `HC*H = 10240`-wide stacked vector, project to `rank + HC`, activate,
project back to `HC*H`, then mix over the streams:

- `hpu/tpc/hc_down_bf16.c:2-5` — `d[j] = sum_c x[c] * w[j][c]`, `lo[j] = BF16(s * sigmoid(s)), s = d[j] * inverse_streams` for `j < rank`, `inject[j-rank] = d[j]`.
- `hpu/tpc/hc_mix.c:5-11` — `sum+=v_f32_ld_tnsr_b(p,normalized)*strata_sigmoid(v_f32_ld_tnsr_b(p,gate));` … `v_f32_st_tnsr(p,output,sum*(1.0f/streams));` — the **mix is a σ-gated mean over the four streams**.
- `hpu/src/hc_ple.cpp:53-55` — `g.multiply(down,scale,scaled);g.sigmoid(scaled,sig);g.multiply(scaled,sig,lo);` — the graph form: one merged GEMM `w_down_inject {rank+HC, HC*H}` split into `down` and `inject` (`hc_ple.cpp:53`), i.e. `lo = s·σ(s)`, then `w_up` GEMM to `gate {HC,H}`, then the mix.

**Write** (per token): `out[s] = residual[s] + block · 2σ(inject[s]/HC)`

- `hpu/tpc/hc_write.c:6-11` — `float64 scalar=s_f32_ld_g((__global__ float*)gen_addr(ic,inject))*(1.0f/streams);` … `float64 weight=strata_sigmoid(scalar)*2.0f;` … `v_f32_st_tnsr(p,output,v_f32_ld_tnsr_b(p,residual)+v_f32_ld_tnsr_b(b,block)*weight);`

Our torch already implements exactly this:

- `vllm-gaudi/vllm_gaudi/models/qwen4exp/hyperconnection_amd.py:69-76` — `hc_gate_mix` (σ-gated mean);
- `vllm-gaudi/vllm_gaudi/models/qwen4exp/hyperconnection_amd.py:78-90` — `out = residual + block_output * (2 * sigmoid(inj / hc)) per stream.` — computed as `inj = 2.0 * torch.sigmoid(injection_logits.float() / hc_count)` then `residual + block_output * inj`.

### 2.2 Kernel contracts and layouts

| kernel | contract (from `configure()`) | notes |
|---|---|---|
| `hc_mix` (id 5) | `normalized, gate` F32 `{width, streams}` → `out` F32 `{width}`; params `Hc{width,streams}`; 64-column members | `hpu/tpc/hc_mix.c:2` |
| `hc_write` (id 7) | `residual {width,streams}`, `block {width}`, `inject {streams}` → `out {width,streams}` | `hpu/tpc/hc_write.c:2` |
| `hc_mix_window` (id 10) | `{width,streams,rows}` → `{width,rows}` | `hpu/tpc/hc_mix_window.c:2` |
| `hc_mix_window_dual` (id 40) | adds the BF16 copy `rounded {width,rows}`; **256 columns per member**, `streams==4` fast path with all sixteen loads in straight-line code | `hpu/tpc/hc_mix_window_dual.c:5-34` |
| `hc_write_window` (id 11) | 8 column tiles per member, gate scalar loaded once per member | `hpu/tpc/hc_write_window.c:2-4` |
| `hc_write_moe` (id 42) | folds the MoE route weighting and per-row expert sum into the write; `hpu/tpc/hc_write_moe.c:2-4` — `block[t] = sum_j experts[t*per_row+j] * route[t*per_row+j]` | `hpu/tpc/hc_write_moe.c:2-5` |
| `hc_write_moe_routed` (id 88) | same, reading BF16 grouped-GEMM output **in place** via `place`, shared expert as the last slot | `hpu/tpc/hc_write_moe_routed.c:3-7`, glue `hpu/tpc/glue.cpp:997-1009` |
| `hc_down_bf16` (id 100) | `x {columns,1}` BF16, `w {columns, rank+streams}` BF16, `inv {1}` → `lo {rank,1}` BF16 + `inject {streams,1}` F32; **eight output rows per member**, `columns%128==0` | `hpu/tpc/hc_down_bf16.c:9`, glue `hpu/tpc/glue.cpp:552-561` |
| `hc_down_rows_bf16` (id 108) | same over `rows<=64`, two output rows per member | glue `hpu/tpc/glue.cpp:562-571` |

**Layouts / stream count.** `streams = 4` always (`HcConfig{hidden=2560,streams=4,rank=320}`,
`hpu/include/strata/hpu/hc_ple.hpp:4-8`). Three layouts appear and must not be confused:

- decode read: `residual {HC,H}` F32, `norm {HC,H}`, `xn {HC,H}`, `xf {1, HC*H}` BF16 (`hpu/src/hc_ple.cpp:17,22-23`);
- window: `residual {t,HC,H}`, flattened to `{t*HC, H}` for the norm and `{t, HC*H}` for the GEMM
  (`hpu/src/hc_ple.cpp:83-86`);
- the MoE-write kernels see `{width, streams, rows}` because TPC is column-major — i.e. `[rows, streams, width]`
  in torch terms (`hpu/tpc/hc_write_moe.c:5`).

Two footguns worth naming, because neither is a compile error:

- **`1/HC` is a bound tensor, not a constant.** `inverse_streams` is a `{1}` F32 tensor
  (`hpu/src/hc_ple.cpp:21` — `const auto scale=g.tensor({"inverse_streams",DType::F32,{1}});`) that the kernels
  *multiply* by (`hpu/src/hc_ple.cpp:55`, then `hpu/tpc/hc_mix.c:10` divides the mix by `streams` again). Nothing
  validates it: `HcConfig::validate` only checks `0 < streams <= 32` (`hpu/src/hc_ple.cpp:10-13`). Binding
  `4.0` instead of `0.25` is silently wrong everywhere. Our port computes `x / hc_count` inline
  (`vllm-gaudi/vllm_gaudi/models/qwen4exp/hyperconnection_amd.py:65-66`), so the port must bind `1/HC` explicitly.
- **The window TPC path has a narrow shape gate**: `rows<=64 && n%256==0 && lr%2==0` with `n = HC*H`
  (`hpu/src/hc_ple.cpp:98`, `:111`). Outside it the graph silently falls back to the GEMM + three elementwise
  nodes, so a benchmark that "used the TPC path" for one bucket may not use it for the next.

### 2.3 Delta for our port

Our port has the same arithmetic in three separate torch ops per read (RMS norm, `hc_gate_mix`, `hc_combine`)
and no fused write. Porting gives three wins, none of which touches the weight format:
`hc_down_bf16` merges norm→down→inject→`s·σ(s)` into one node (removes two HBM round trips of `[1,10240]`
BF16 and one `[1,320]` F32); `hc_mix_window_dual` removes the BF16 cast node; `hc_write_moe_routed` removes the
route-weight multiply, the `reduce_sum` and the write (three nodes → one). The `lo` BF16 rounding at
`hpu/tpc/hc_down_bf16.c:34` and the `2σ` at `hc_write.c:7` are the two numerics to reproduce exactly.

---

## 3. PLE n-gram table

### 3.1 The host hash law (verbatim constants)

- `include/strata/kernels/ngram.hpp:31-42` — `inline constexpr int NG_N_EMBD = 2560;` … `inline constexpr int PLE_HEAD_DIM = 160;` … `NGRAM_SIZE = 3`, `HEADS_PER_NGRAM = 8`,
  `PLE_N_HEADS = (NGRAM_SIZE-1)*HEADS_PER_NGRAM` = **16**, `PLE_CONV_KERNEL = 4`, `NG_HC = 4`, `NG_HIST = 9`.
- `src/kernels/ngram.cpp:47-49` — `c.mult[0] = 23703573157769ull;` `c.mult[1] = 20109073645365ull;` `c.mult[2] = 8052911324071ull;`
  and the 16 head vocab sizes `20000003, 20000023, 20000033, …` (not an arithmetic sequence, `src/kernels/ngram.cpp:50-52`).
- `src/kernels/ngram.cpp:63-70` — `uint64_t mixed = (uint64_t) ctx[0] * mult[0]; for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * mult[j];` — **multiply then XOR**, first term an assignment.
- `src/kernels/ngram.cpp:82-95` — `const int32_t t = cut ? TOKEN_NULL : prev[i * n_prev + (n_prev - s)];` — the row law; predecessors are **oldest first**, `cut = cut || t < 0 || t == PLE_EOS_TOKEN_ID;` (the cut propagates **forward**), then for `n = 2..3` and `g = 0..7`:
  `out[i*PLE_N_HEADS + h] = (uint32_t) (mixed % c.vocab[h] + c.offset[h]);` with `h = (n-2)*8 + g`.
  So heads `0..7` are **bigram** rows and `8..15` are **trigram** rows, all `% head_vocab + head_offset`,
  never a mask (`vocab` is not a power of two).
- `include/strata/kernels/ngram.hpp:15-21` lists the four rival readings that silently produce in-range rows
  (sum instead of XOR, mask instead of `%`, cut backwards, `-1` versus token 0). Any port must assert them.

### 3.2 Table row layout and how the rows are added

- One row is 160 values = one head's contribution (`include/strata/kernels/ngram.hpp:44-52`); the 16 gathered rows
  are flattened **head-slowest** into 2560 values — `include/strata/kernels/ngram.hpp:210-213` and
  `src/kernels/ngram.cpp:665-668` — so the gathered vector is directly `n_embd`-wide.
- **The two artifacts carry the same row space, verified.** Our checkpoint ships `layer_multipliers`,
  `ngram_heads_vocab_sizes` and `ngram_heads_offsets` as tensors
  (`model.language_model.layers.1.ple.ple_embedding.{layer_multipliers,ngram_heads_vocab_sizes,ngram_heads_offsets}`,
  I64 `[3]`/`[16]`/`[16]`), and all three are **identical** to the constants Strata hard-codes
  (`src/kernels/ngram.cpp:47-55`): multipliers `23703573157769 / 20109073645365 / 8052911324071`, vocab sizes
  `20000003, 20000023, 20000033, …`, offsets the running sum. Our 128 shards × 2 500 012 rows = 320 001 536 =
  `PLE_TABLE_ROWS` (`include/strata/kernels/ngram.hpp:46`), and the 16 head ranges end 90 rows short of the table —
  the same 90 alignment rows Strata's validation reports (`hpu/src/ple_input.cpp:17-20`). So the PLE hash needs no
  re-derivation from `ngram_vocab_size_base`: read the three tensors. Checked by
  `validation/qwen4exp/tools/check_ple_hash_parity.py` (§8.1).
- Our checkpoint stores the same table as **128 shards** of NVFP4 — `text_config.split_ngram_parts = 128` in
  `config.json`, which is exactly the shard count; the flat row space is unchanged, so the hash is unaffected:
  `vllm-gaudi/vllm_gaudi/models/qwen4exp/ple_embedding.py:4-10` — `ngram_embedding.shard_<i>.weight` U8 `[rows, 80]` (packed E2M1) +
  `ngram_embedding.shard_<i>.weight_scale` F8_E4M3 `[rows, 10]` (per-16) = 160 values a row. `shard = row_id // shard_rows`
  (`vllm-gaudi/vllm_gaudi/models/qwen4exp/ple_runtime.py:416-421`), and the 16 head ranges are sub-spans of one flat row
  space that deliberately do not align with the shard boundaries (`ple_runtime.py:374-386`).
- Addition: the gathered 2560 vector is the **only** PLE input to the device; the PLE block's output is
  `hidden + gated + silu(conv)` — an add onto the multi-stream residual:
  `hpu/tpc/ple_conv.c:16` — `const float64 result=(v_f32_ld_tnsr_b(p,hidden)+v_f32_ld_tnsr_b(p,gated))+sum*strata_sigmoid(sum);`
  and our equivalent `vllm-gaudi/vllm_gaudi/models/qwen4exp/ple_runtime.py:359-360`.
- The gate that turns the 2560 vector into 4 gated views is identical in both trees
  (`hpu/tpc/ple_gate.c:12-15` vs `vllm-gaudi/vllm_gaudi/models/qwen4exp/ple_runtime.py:246-253`): `d = (k·q)/sqrt(160)`,
  `g = σ(sign(d)·sqrt(max(|d|,1e-6)))` — note the clamp is on the **magnitude**, and `1/sqrt(width)` is
  precomputed (`inverse_sqrt_width`, `hpu/src/hc_ple.cpp:182`).

### 3.3 Why the hash stays on the host

- `include/strata/kernels/ngram.hpp:5-6` — `the HOST hash.  Sixteen row indices per token from the last three token ids, computed`
- Our port states the hardware reason: `vllm-gaudi/vllm_gaudi/models/qwen4exp/ple_runtime.py:416-418` —
  `Gaudi2 int64 multiply is not exact`. TPC has no int64 multiply (32-bit product) and no cheap 64-bit modulo;
  the 16 rows are a few dozen host instructions and would be a scalar-pipeline stall on device. Both the checkpoint's
  `head_vocab_sizes` and `layer_multipliers` are 64-bit quantities.
- Consequence: the *only* device-side PLE work is the 2560-wide block (two `[10240,2560]`/`[2560,2560]` GEMMs
  plus the gate and conv), i.e. **`w_key` 52.4 MB + `w_value` 13.1 MB BF16**, plus `norm_*` `{4,2560}` and
  `conv_weights {4, 10240}` — about 66 MB. The table itself is 320 001 536 rows × 90 B = **28.8 GB**
  (the 162 GB figure is the whole checkpoint) and none of it enters HBM.

### 3.4 Host-offload design (what to build)

1. **Storage.** `mmap(MAP_PRIVATE|MAP_POPULATE off)` each of the 128 shard files, or `pread` per row. Strata's
   `mmap` path is the A/B arm that measured 2.10–2.61 ms/token of *serial page faults* for 16 rows
   (`src/kernels/ngram.cpp:638-647`), and its fix is one `PrefetchVirtualMemory`/`madvise(MADV_WILLNEED)` for all
   sixteen rows (`src/kernels/ngram.cpp:648-663`). On Linux we already have `madvise`
   (`src/kernels/ngram.cpp:566-572`).
2. **Pinned staging.** One `[window_rows, 16, 160]` BF16 (or F32) pinned buffer + one H2D copy per PLE layer per
   window; Strata's `gather_rows` decodes a window's rows on 16 threads
   (`hpu/src/ple_input.cpp:118-127`).
3. **Issue/collect split** so the gather overlaps the previous layer's graph:
   `hpu/src/ple_input.cpp:112-117` (`issue` then `collect`), and the batch form
   `hpu/src/ple_input.cpp:61-82` (`ple_batch_rows`, per-row predecessors threaded through the batch).
4. **Per-token cost.** 16 rows × (80 B + 10 B) = 1 440 B read + 5 120 B written as BF16 to the device; that is
   latency, not bandwidth — hide it by issuing the gather for window `w+1` behind window `w`'s graph.
5. **Our port already does 1–4 in spirit** (`ple_embedding.py:16-22`, `ple_runtime.py:408-435`) but per-lookup and
   per-shard in torch, with the gather on the caller's thread. The port work is: replace the torch shard loop
   with a `pread`/`mmap` + decode loop (no torch op per shard), add the issue/collect prefetch, and keep the
   decoded rows in BF16 to halve the H2D bytes.
6. **Numerics to preserve.** Row decode = `E2M1 code × E4M3 scale` (per-16), then `× global_scale`
   (`ple_runtime.py:456`) — the `weight_scale_2` analogue; the 16-row flatten is head-slowest, and the hash's EOS
   cut is forward-propagating. That global scale is no longer an open question: the checkpoint carries
   `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale_2` as F32 `[1]` =
   **3.324236240587197e-05**, and `load_ple_ngram` already reads it when present (`ple_runtime.py:486-491`).
   It must be applied **once per row, after** the per-16 E4M3 scale, never folded into the tables.

---

## 4. 4-bit MoE kernels (decode GEMV, prefill bucketing, packed matmul)

### 4.1 The weight layout Strata assumes

**Resident IQ4** store, one `U8` slab: codes first, then F32 scales.

- `hpu/tpc/glue.cpp:938` — `const uint64_t c=a.columns,r=a.rows,s=a.slots,xrows=s/a.x_group,bytes=r/2*c+r*(c/32)*4;`
  i.e. `bytes = rows/2*columns` (nibbles, 2 codes a byte along **K**) `+ rows*(columns/32)*4` (one F32 a 32-value
  group). Codes and scales are one tensor, scales **after** the codes.
- Addressing in the GEMV: `hpu/tpc/iq4q8_gemv_packed.c:27` — `const int base=((group*blocks+b)*16*5)*256;`
  and `hpu/tpc/iq4q8_gemv_packed.c:33` — `int5 at={base+(c/2)*5*256,erow,0,0,0};` — a **256-byte vector along the
  output-row axis**, one vector per (column pair, 256-row tile). Nibble order inside a byte:
  `hpu/tpc/iq4q8_gemv_packed.c:43-44` — `const char256 w0=(char256)v_u8_shuffle_b(lut,V|(unsigned char)0x80,0,lut);` — `w0` is the low nibble and feeds column `c`, `w1` the high nibble and column `c+1`, so **the low nibble is the even column**.
- Scales: `hpu/tpc/iq4q8_gemv_packed.c:79` — `int5 sp={codes_bytes+((group*blocks+b)*5)*1024,erow,0,0,0};`
  then four 256-byte loads = 256 F32 = one scale per output row, per 32-column block.
- The dequant-to-BF16 prefill twin reads the same slab with a different vector width
  (`hpu/tpc/iq4_expand_tiles.c:19` — `int5 sp={codes_bytes+((group*blocks+b)*5+t)*1024,e,0,0,0};` and
  `hpu/tpc/iq4_expand_tiles.c:26` — `int5 at={((group*blocks+b)*80+t)*256,e,0,0,0};`).

**Our NVFP4** store is the same idea with three differences that matter:

| | Strata IQ4_XS | ours NVFP4 |
|---|---|---|
| code | 4-bit **codebook index** (`packed_iq4nl_values`, `hpu/tpc/packed_matmul.c:30`) | 4-bit **E2M1** (sign in bit 3, magnitude in bits 0-2) |
| group | 32 values, one F32 scale | **16** values, one **E4M3 byte** |
| per-expert | none (scale lives in the slab) | separate F32 `scale2 [E]`, applied once after the block sum |

Our planes are `packed [K/2, N, E] U8`, `scales [K/16, N, E] U8`,
`vllm-gaudi/csrc/qwen4exp/include/qwen4exp_nvfp4_indexed_dequant_bf16.h:50-53` and
`vllm-gaudi/csrc/qwen4exp/host/qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2.cpp:4-11`.

**The pattern transfers; the address arithmetic does not.** Strata's resident slab per expert is *not* a plain
`[row][K/2]` matrix: it is re-tiled so that the innermost 256 bytes are 256 consecutive **output rows** of one
code byte. Flattened, with `G = rows/1280` super-tiles of 5 × 256 rows and `B = columns/32` blocks:

```
codes  : ((  ((g*B + b)*16 + c) * 5 + t) * 256 + r    c in [0,16) code byte, t in [0,5) sub-tile, r in [0,256) row
scales : codes_bytes + ( ((g*B + b)*5 + t) * 1024 + r*4 )
```

which is exactly what `hpu/tpc/iq4q8_gemv_packed.c:27` (`const int base=((group*blocks+b)*16*5)*256;`), `:33`
(`int5 at={base+(c/2)*5*256,erow,0,0,0};`) and `:79` (`int5 sp={codes_bytes+((group*blocks+b)*5)*1024,erow,0,0,0};`)
encode, and the same tiling the dequant twin walks with `80 = 16*5` tile strides per block
(`hpu/tpc/iq4_expand_tiles.c:19` — `int5 sp={codes_bytes+((group*blocks+b)*5+t)*1024,e,0,0,0};` and
`hpu/tpc/iq4_expand_tiles.c:26` — `int5 at={((group*blocks+b)*80+t)*256,e,0,0,0};`). A `[K/2, N]` plane keeps its
256-byte vector along **K**, as our kernel does, so those three lines must be re-derived for our layout, not copied.

What genuinely transfers: one 256-byte load per (column pair, 256-row sub-tile); five independent sub-tile
accumulators; a 256-byte LUT vector shuffled by the code instead of a table load; one F32 multiply per block by
`scale * dx`; and codes-then-scales in one slab.

### 4.2 `iq4q8_gemv_packed.c` — the decode GEMV

- Signature: `hpu/tpc/iq4q8_gemv_packed.c:6-7` — `void main(tensor x,tensor dx,tensor table,tensor ids,tensor w,tensor shared,tensor out, int columns,int rows,int x_group,int parts,int slots,int period)`.
- One **member = (slot, part, group of five 256-row tiles)**:
  `hpu/tpc/iq4q8_gemv_packed.c:14` — `const int slot=member/(parts*groups),rest=member-slot*parts*groups,part=rest/groups,group=rest-part*groups;`
- Latency hiding: (a) `parts` splits the K axis so several members stream different column ranges of the same
  expert; (b) the shared expert is a separate `W=5` weight row read at the same address (`erow = is_shared?0:id`,
  `hpu/tpc/iq4q8_gemv_packed.c:18`) so routed+shared come out of one launch; (c) the software-pipelined variant
  issues the next column pair's five loads before the current pair's MACs
  (`hpu/tpc/iq4q8_gemv_packed.c:49-77`), bit-identical to the plain loop (`:28-30` documents the A/B toggle);
  (d) the LUT is a **256-byte vector shuffled by the code** (`hpu/tpc/iq4q8_gemv_packed.c:43-45`), so the 4-bit
  code is decoded without a table load; (e) five independent accumulators `partial[5][4]` hide MAC latency
  (`hpu/tpc/iq4q8_gemv_packed.c:5`, `:19`).
- Output: `out {rows, slots, parts}` F32 (`hpu/tpc/glue.cpp:942`) — the parts are summed by the caller/join.

### 4.3 Activation packing — `q8_quant_packed.c`

- `hpu/tpc/q8_quant_packed.c:5` — `void main(tensor x,tensor xq,tensor dx,int columns,int rows)`; input `x {columns, rows}` F32, outputs `xq {256, columns/16, rows}` U8 and `dx {64, columns/32, rows}` F32 (glue `hpu/tpc/glue.cpp:944-951`).
- Per **32 columns**: one symmetric scale `d = peak/127` with round-half-away-from-zero
  (`hpu/tpc/q8_quant_packed.c:20-25`), and **two 256-byte code vectors**, each holding 16 columns' codes
  replicated in every 16-lane group (`hpu/tpc/q8_quant_packed.c:28-36`), so the GEMV broadcasts a column with one
  in-group shuffle instead of a 256-byte load (`hpu/tpc/iq4q8_gemv_packed.c:34-36`).
- Four members run side by side because each is one dependent reduction chain
  (`hpu/tpc/q8_quant_packed.c:10-15`).

### 4.4 Prefill bucketing — `moe_mme_count` / `prefix` / `scatter` / `place` / `tilemap`

Purpose: turn `rows*per_row` routed slots into **per-expert contiguous row runs** so one grouped MME GEMM per tile
can run, with every slot placed and none dropped.

1. `hpu/tpc/moe_mme_count.c:6` — `void main(tensor ids,tensor counts,int experts,int slots,int per_row,int top_k,int rows) {` — `ids {slots} I32 -> counts {experts, blocks} F32`; members are **(64-slot block, 64 experts)** (glue `hpu/tpc/glue.cpp:380-393` … `out.indexSpaceGeometry[0]=blocks;out.indexSpaceGeometry[1]=e/64;`), the block's ids are loaded and masked once (`moe_mme_count.c:12-18`), the shared slot of each row (`slot%per_row==top_k`) is excluded.
2. `hpu/tpc/moe_mme_prefix.c:5` — `void main(tensor counts,tensor offsets,tensor totals,tensor tiles,int experts,int slots,int per_row,int top_k,int rows) {` — `counts {experts, blocks} -> offsets {experts, blocks} F32, totals {experts} I32, tiles {experts} F32`; exclusive running count over blocks (`:11-16`), `ceil(total/rows)` as `floor((total + rows - 0.5)/rows)` (`:20-21`).
3. `hpu/tpc/moe_mme_scatter.c:12` / `moe_mme_place.c:15` — every routed slot's tile row: `const int position=(int)base[i]+earlier;` (`moe_mme_place.c:47`), tile `start[e] + position/rows`, row `position%rows` (`moe_mme_scatter.c:42-49`). The "earlier count" is an **open-addressed 64-entry hash** over the block's experts (`moe_mme_scatter.c:38-46`), eight slots' scalar loads issued together (`:20-34`), and a full block's places go out as one vector (`:54-58`). `place` is the inverse map, `-1` for the shared slot.
   `moe_mme_place.c` is the newer form: it drops the scattered scalar stores and lets the SDK do
   `bucket[target[s]] = s` (`hpu/tpc/moe_mme_place.c:1-9`).
4. `hpu/tpc/moe_mme_tilemap.c:5` — `void main(tensor end_,tensor start,tensor totals,tensor tile_expert,tensor tile_rows,int experts,int tiles,int rows) {` — `end/start {experts} F32, totals {experts} I32 -> tile_expert {tiles} I32, tile_rows {tiles} I32`, a binary search of tile → expert (`:10-22`).
5. Then `iq4_expand_tiles` dequantizes the bucket's experts to BF16 tiles (see below), the grouped MME runs, and
   `moe_mme_act` / `u4_moe_gu_epi` / `u4_moe_down_epi` / `moe_mme_join_perm` fuse the epilogues
   (`hpu/tpc/moe_mme_act.c:2-3`, `hpu/tpc/u4_moe_gu_epi.c:1-5`).

The decode-side alternative (`u4_moe_bucket.c:9-37`) is the same law without tiling: `bucket[e*rows+m] = slot` and
`place[slot] = e*rows+m`, with an `overflow[e]` count instead of silent drops (`hpu/tpc/u4_moe_bucket.c:26-29`).

**The resident U4 store is the closest existing analogue to our NVFP4 experts, and it is the one to copy.** It keeps
the weights 4-bit in HBM with a *separate* scale plane — `w(e,n,k) = scale[e,g,n]*code(e,n,k) + offset[e,g,n]` — and
runs the whole layer as one graph of ~a dozen nodes: routing, bucketing, two grouped batch GEMMs (each preceded by a
U4→BF16 cast) with a fused epilogue after each, then the join (`hpu/include/strata/hpu/u4_resident.hpp:2-22`,
`hpu/src/u4_moe.cpp:24-103`):

| step | node | contract |
|---|---|---|
| route | `strata_route_topk_f32` | logits + shared gate → `ids {slots}` I32, `route_weights {slots}` F32, shared expert as slot `top_k` (`hpu/src/u4_moe.cpp:36-39`) |
| bucket | `strata_u4_route_blocks_f32` + `strata_u4_bucket_blocks_f32` | 64-slot count per expert, then a **fixed 64-row bucket per expert** with an `overflow[e]` spill (`hpu/src/u4_moe.cpp:12-18` — `u4_bucket_rows` returns `64`, and `hpu/include/strata/hpu/u4_moe.hpp:16-18` requires an overflow replay) |
| gather | `strata_u4_moe_gather_f32` | activation → the MME's BF16 `a {batch, rows, group}` **plus `sumx`**, the per-(batch,row) input sum the affine epilogue needs (`hpu/tpc/u4_moe_gather.c:1-5`) |
| GEMM | `cast_u4_to_bf16` then `batch_gemm(a, unpacked, out, transpose_b)` | **Gaudi2 rejects a mixed BF16/low-bit GEMM**, so the codes are widened to a BF16 tile per batch member first (`hpu/src/u4_moe.cpp:51-58`) |
| epilogue | `strata_u4_moe_gu_epi_f32`, `strata_u4_moe_down_epi_f32` | `y = (s·code + o·sumx)` per K group, `silu(gate)*up`, route weight, straight into `expert_rows[slot]` (`hpu/tpc/u4_moe_gu_epi.c:1-5`, `hpu/tpc/u4_moe_down_epi.c:1-4`) |
| join | `strata_u4_moe_join_f32` | routed rows by `place`, shared rows by token, zeros where the bucket overflowed (`hpu/tpc/u4_moe_join.c:1-5`) |

Three consequences for us, all of which hold for NVFP4 unchanged: (i) the per-expert scale must be a **separate
plane indexed in the epilogue** — our `scale2` plays exactly the role of `u4`'s per-row `(scale, offset)` pair, and
applying it in the epilogue instead of folding it into the weights is what keeps one codec for all 512 experts;
(ii) a BF16 tile per bucket member is the *only* legal way into the MME on this SDK, so the port's intermediate is
BF16 while the resident store stays 4-bit; (iii) the epilogue needs the activation's **own sum** per (row, K group)
whenever the codec has an offset term — NVFP4's E2M1 has none, so our epilogue is a pure scale multiply and does
not need `sumx` at all (a simplification, not a gap).

### 4.5 `packed_matmul.c` — scalar IQ4 reference dot products

587 lines, one function per ggml type, all with the same shape:
`float packed_<type>_dot(tensor weight,tensor activation,tensor metadata,int columns,int row,int token)` where the
activation is Q8 and `metadata` is the per-token activation scale. Two are directly relevant:

- **IQ4_NL** (ours is not this): `hpu/tpc/packed_matmul.c:30` — `static const int packed_iq4nl_values[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};`, used at `hpu/tpc/packed_matmul.c:211-212` with the **low nibble = element `32*sub+j`, high nibble = `32*sub+16+j`** (split-half, not interleaved).
- **NVFP4 (GGML type 40)**, the closest existing thing to our format: `hpu/tpc/packed_matmul.c:483-492` — `const float scale=packed_ue4m3((unsigned int)packed_byte(weight,at+sub,row));` then an **integer** dot with the
  doubled E2M1 table and `total+=(scale*d8)*(float)dot;` (`:492`). `packed_ue4m3` is the exact E4M3 decode
  (`hpu/tpc/packed_matmul.c:469-477`), and the codebook is
  `hpu/tpc/fp4_values.inc:6-8` — `0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12` (E2M1 **doubled**; codes 0..7 positive, 8..15 negative, sign in bit 3).
- The IQ4_XS path (`hpu/tpc/packed_matmul.c:195-221`) is the reference for the "integer dot per sub-block, then
  one F32 multiply by `d8*(scale)`" structure, including the `(raw-32)` 6-bit scale decode we do not need.

### 4.6 What must change for NVFP4 — item by item

1. **Code → value.** The E2M1 code is **4 bits** (sign in bit 3, two exponent bits, one mantissa bit = 16 codes), not
   2 bits. Strata's LUT is an int8 array indexed by that 4-bit code, holding *integers*
   (`fp4_values.inc`). E2M1's magnitudes are `{0,0.5,1,1.5,2,3,4,6}`; **doubled** they are exactly the integers in
   `fp4_values.inc`, and our own LUT is the same set (`vllm-gaudi/vllm_gaudi/models/qwen4exp/csf_math.py:12-19`).
   So the LUT path is available, with one caveat: because the E4M3 group scale is **not a power of two**, the
   integer dot must be multiplied by `(scale/2)·dx` — a *different rounding* than our current bit-exact
   `E2M1 × E4M3 → BF16` product (`vllm-gaudi/csrc/qwen4exp/include/qwen4exp_nvfp4_decode.h:12-55`). Both are
   defensible; the integer form is faster (int8 MAC at 2 codes/byte), the bit form is bit-exact against the fp32
   golden. **Decision for the port: keep the bit-exact form for the decode GEMV (it is already hardware-verified),
   and use the doubled-int LUT only if a profile shows the multiply is the bottleneck.**
2. **Group size 32 → 16.** Two consequences: (a) the scale plane is `K/16` bytes a row, not `K/32` F32, i.e. the
   per-32-block scale load count doubles while the bytes halve; (b) an E4M3 scale must be decoded — cheap as a
   `v_u8_shuffle` through a 256-byte table, which is what `packed_ue4m3` does scalar-wise and what our
   `nvfp4_mul_bits` does by bit construction.
   **Two traps in `packed_ue4m3`, both measured (`tools/check_fp4_lut.py`):** it ignores bit 7, so it decodes
   only magnitudes (`hpu/tpc/packed_matmul.c:470` — `if(x==0u || x==0x7fu) return 0.0f;` then `exp=(x>>3)&15u`),
   and its exponent is `2^(exp-8)`, i.e. **half** the standard E4M3FN value — `hpu/tpc/packed_matmul.c:472` —
   `const int e=exp==0?-10:exp-8;   // the helper's closing half applies to both branches` — because GGML's
   NVFP4 block folds that half into its own `d`. Our planes use signed E4M3FN at `2^(exp-7)`
   (`vllm-gaudi/vllm_gaudi/models/qwen4exp/csf_math.py:24`), so `packed_ue4m3` must **not** be copied verbatim: it
   would halve every group scale and drop the sign of the 126 negative codes.
3. **Low-nibble-first, consecutive-K pairs.** Our packing is `packed [K/2, N, E]` with byte `k/2` holding column
   `k` (low) and `k+1` (high) — `vllm-gaudi/csrc/qwen4exp/include/qwen4exp_nvfp4_decode.h:30-32` (u16 lane `j`
   reads byte `j>>1`, shift `(j&1)*4`). Strata's IQ4 nibble order is the same (`iq4q8_gemv_packed.c:43-44`), but
   its **NVFP4** reference uses GGML's split-half order (`packed_matmul.c:487-488`). **Do not copy
   `packed_nvfp4_dot`'s index arithmetic** — only its structure.
4. **F32 accumulate.** Strata's GEMV accumulates *int32 per 32-column block* and widens per block
   (`iq4q8_gemv_packed.c:83-86`). We need F32 because the LUT is scaled: keep our
   `v_bf16_mac_acc32_b(weight, xv, acc)` (`qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2.c:31`) or do int32 per 16-group
   and one F32 multiply — both end in F32.
5. **`scale2` once, after the block sum.** Our kernel already does this at
   `vllm-gaudi/csrc/qwen4exp/kernels/qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2.c:73` —
   `const float64 total = v_f32_reduce_add(acc.v1 + acc.v2) * s2;` — one multiply per (row, slot). Strata has no
   analogue (its per-expert scale is fused into `d`); keeping it here is a *feature*: it is what lets the routed
   experts share one codec.
6. **The vector axis moves from K to rows.** Our 256-byte loads run along K (one row at a time, 512 K per load);
   Strata's run along the output-row axis (256 rows at a time, one column pair per load). The bytes are identical,
   but the *load count per useful byte* is not: see §5.4 for the measured scale-plane over-read and the repeated
   activation loads. This is the one structural change the port should take, and it needs no format change.
7. **Nothing else.** The nibble planes stay `uint8` in HBM, group scales stay 1 byte per 16 values, `scale2` stays
   one F32 per expert. No BF16 materialization of the resident store (~225 GiB for all 48 layers' experts) and no requantization.

---

## 5. Port plan for our TPC-C kernels

### 5.1 What we already have (do not break)

- `qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2.c` — decode GEMV, index space **rank 2** `[ceil(N/4), slots]`
  (`indexSpaceGeometry[0]=(rows+3)/4`, `[1]=slots`, row axis stride 4, `vllm-gaudi/csrc/qwen4exp/host/qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2.cpp:81-106`; H8 gives one member
  four output rows, like the native MoE decode GEMV); the K tile loop is *inside* the member
  (`vllm-gaudi/csrc/qwen4exp/kernels/qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2.c:185-204`), unlike Strata which puts `parts` in the index space,
  `x [K,S] BF16`, `packed [K/2,N,E] U8`, `scales [K/16,N,E] U8`, `scale2 [E] F32`, `out [N,S] BF16`
  (`vllm-gaudi/csrc/qwen4exp/host/qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2.cpp:4-11,81-106`).
- `qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2.c` — per-expert dequant to BF16 tiles (≤ 64 ids/call), index space
  **rank 3** `[tiles, rows, slots]` (`vllm-gaudi/csrc/qwen4exp/host/qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2.cpp:78-81`).
- Host glue with hard geometry checks: `slots<=64`, `k_half%64==0`, `scales.maxSizes[0]*8==k_half`
  (`vllm-gaudi/csrc/qwen4exp/host/qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2.cpp:56-63`), and explicit access patterns (`:81-92`).

### 5.2 Function-level signatures to add (TPC-C `main()` argument lists)

```c
/* 1. decode GEMV, Strata-shaped: one member per (slot, part, 256-row tile), the 256-byte vector
 *    along ROWS. This is the structural change from our current kernel, whose vector is along K.
 *    NOTE the dimension order: TPC dim 0 is the fastest axis, so "the vector along rows" REQUIRES
 *    dim 0 = rows. That is a different resident packing from today's, not a re-indexing of it:
 *        today    torch [E, N, K/2]  -> TPC (K/2, N, E)   256-byte vector = 512 k-values of one row
 *        proposed torch [E, K/2, N]  -> TPC (N, K/2, E)   256-byte vector = 256 rows of one k-byte
 *    The loader pays one transpose (and a 16-B-per-64-value tiling if Strata's sub-tile order is
 *    kept); both planes stay U8 with the same byte count.                                    */
void main(tensor ids, tensor packed, tensor scales, tensor x, tensor scale2,
          tensor shared_packed, tensor shared_scales, tensor out,
          int columns, int rows, int parts, int slots, int period);
/*    packed  {rows, columns/2, experts} U8   dim0 = rows; byte j = columns (2j, 2j+1), low nibble first
 *    scales  {rows, columns/16, experts} U8  dim0 = rows; E4M3 byte, one per 16 columns
 *    x       {columns, slots} BF16           the slot's activation row (broadcast over the 256 rows)
 *    out     {rows, slots, parts} F32        parts summed by the caller, then scale2
 *    members: slots*parts*(rows/256)         per member: 16 packed loads x 5 sub-tiles + scales
 *    each member's 5 sub-tiles are 5 independent accumulator chains, exactly as Strata's partial[5][4]
 *    the shared expert is its own pair of planes (our dim 2 is the expert axis, so Strata's
 *    "same address, W=5" trick does not apply) and is selected by slot % period == period-1  */

/* 2. bucketing: count -> prefix -> place (one launch each, 64-slot members) */
void main(tensor ids, tensor counts, int experts, int slots, int per_row, int top_k, int rows);
void main(tensor counts, tensor offsets, tensor totals, tensor tiles,
          int experts, int slots, int per_row, int top_k, int rows);
void main(tensor ids, tensor offsets, tensor start, tensor place, tensor target, tensor slot_ids,
          tensor blank, int experts, int slots, int per_row, int top_k, int rows, int trash);

/* 3. per-expert dequant-to-BF16 tiles for the grouped MME, from OUR planes */
void main(tensor scale_lut, tensor w, tensor tile_expert, tensor out,
          int rows, int columns, int first);   /* out {rows, columns, count} BF16 */

/* 4. QSA selection (our law: no +1e9, tail outside the budget) */
void main(tensor start_cell, tensor scores, tensor scratch, tensor kept,
          int keys, int rows, int window, int width_sel, int block, int heads);
```

Index-space geometry to declare in the glue (mirroring `hpu/tpc/glue.cpp` conventions):

| kernel | rank | geometry | guard rails copied from Strata |
|---|---|---|---|
| GEMV | 1 (Strata) / 2 (ours today) | `slots*parts*(rows/1280)` (Strata `glue.cpp:943`); ours `[rows, slots]` | `columns%32==0`, `rows%1280==0` (`glue.cpp:935-937`); ours uses 128-row members, so `rows%128==0` |
| count | 2 | `(blocks, experts/64)`, `blocks=ceil(slots/64)` | `experts%64==0`, `slots<=kMaxMoeSlots=90112` (`model_abi.hpp:10`) |
| prefix | 1 | `experts/64` | outputs `{e,blocks}`, `{e}` I32, `{e}` F32 |
| place | 1 | `blocks` | `slots%per_row==0`, `per_row>1`, `top_k<per_row` |
| expand-tiles | 2 | `(rows/256, count)` | tile rows 256, `columns%32==0` |
| select | 1 | `rows` | `keys%(256*block)==0`, `block` a power of two ≤ 64 (`glue.cpp:1034-1040`) |

Four conventions the glue imposes on every one of those kernels, and which a new kernel must follow exactly
(`hpu/tpc/glue.cpp`):

1. **Parameters are a POD struct of `int32_t`/`float`**, never a long argument list at the ABI level: each kernel
   gets one `abi::*` struct (`hpu/include/strata/hpu/model_abi.hpp:11-113`) with a `static_assert` on its size, and
   the graph node is built as `g.op("strata_<kernel>_<suffix>", in, out, &a, sizeof(a))`. `main()` then declares
   those fields as loose scalars in the same order — the struct never appears in the kernel source.
2. **The kernel is registered in an append-only table** of
   `{guid, elf_start, elf_end, #inputs, #outputs}` (`hpu/tpc/glue.cpp:150-281`), and the table length is asserted —
   `hpu/tpc/glue.cpp:282` — `static_assert(std::size(kernels)==112,"kernel ids are append-only: configure() and base_kernel() index this table");`.
   Ids are indices into it, so a new kernel goes on the **end**; variants alias back with `base_kernel()`
   (`hpu/tpc/glue.cpp:286`, e.g. `kernel_id==107?91:` for the bits/`kept` twin).
3. **`configure()` re-checks every shape against the declared `indexSpaceGeometry`** and returns
   `GLUE_UNSUPPORTED_LAYER_CONFIGURATION` / `GLUE_INCOMPATIBLE_INPUT_SIZE` before anything runs — e.g. the GEMV's
   `a.rows%1280` and `out.indexSpaceGeometry[0]=s*a.parts*(r/(kernel_id==47?256:1280));`
   (`hpu/tpc/glue.cpp:935-943`). A port that skips this gets a wrong answer, not an error.
4. **Scalars that change per token travel as 1-element tensors**, not as parameters: `start {1}` in
   `qsa_index_cells`, `cell {1}` in `qsa_decode_blocks`, `start {2}` in `qsa_select_expand`. That is what lets one
   compiled recipe serve every position and every bucket (`hpu/src/layer.cpp:232`).

### 5.3 Prefill bucketing, Strata-style, with our planes

```
router ids (rows*per_row, shared slot = id 0 at slot%per_row==top_k)
  -> count      {experts, blocks} F32                (moe_mme_count.c)
  -> prefix     offsets {experts, blocks}, totals {experts} I32, tiles {experts} F32
  -> place/target/slot_ids  {slots} I32              (moe_mme_place.c)
  -> SDK scatter: bucket[target[s]] = s
  -> tilemap    tile_expert {tiles} I32, tile_rows {tiles} I32
  -> expand     out {rows, columns, count} BF16      (our NVFP4 planes -> BF16 tiles, ALLOWED as an
                                                      intermediate: the resident store stays 4-bit)
  -> grouped MME gate/up  -> u4_moe_gu_epi (SiLU * up, BF16, per-row sum)
  -> grouped MME down     -> u4_moe_down_epi
  -> hc_write_moe_routed (route weight + per-row sum + HC write, one node)
```

Two invariants from Strata that must be preserved: (i) **no slot is ever dropped** — an expert gets
`ceil(total/rows)` tiles so every slot has a place (`moe_mme_scatter.c:2-6`); (ii) the routed and shared experts
land in the **same** slot space, shared last (`hc_write_moe_routed.c:6` — "The shared slot is the last of each
token's per_row slots"), which is what makes the single-node tail possible.

The dequant-to-BF16 tile path is `rows/2*columns + rows*(columns/16)` bytes read per expert tile and writes
`rows*columns*2` BF16, so at `rows=1280, columns=640` one expert costs 409 600 B in and 1 638 400 B out — the
`STRATA_HPU_MOE_MME_TILE=128` deployment value (`hpu/deploy/strata-hpu-production.json`) is the knob that trades
this against MME efficiency. Never write BF16 tiles back over the resident store.

### 5.4 Decode path, keeping 4-bit residency

**What our GEMV does today** (measured from the source, not guessed). One launch covers every slot of the token —
`grouped_gemv` flattens `topk_ids` and calls the op once per projection
(`vllm-gaudi/vllm_gaudi/models/qwen4exp/packed_moe.py:446-449`), so `S = tokens*top_k ≤ 64` is the `slots` axis; the
index space is `[rows, slots]` and the tile loop is inside the member
(`vllm-gaudi/csrc/qwen4exp/kernels/qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2.c:51-72`). Per (row, slot) and per
512-K tile it issues **1 packed load of 256 B (fully used: 512 K = 256 B), 1 scale load of 256 B of which only 32 B
are used, and 4 activation loads of 256 B** (128 BF16 each).

**The two measurable wastes**, both fixed by Strata's vector axis:

| | today | with the 256-byte vector along rows |
|---|---|---|
| scale plane | `scales` is `[K/16, N, E]`, and a 256-byte load starting at `tile*32` uses 32 bytes (`vllm-gaudi/csrc/qwen4exp/kernels/qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2.c:63-64`) | one load covers 256 rows x 1 group-byte, fully used |
| activation | re-loaded per row: 4 loads x 5 tiles = 20 loads of 256 B per row, all the same `x` row | one broadcast per column pair, shared by all 256 rows of the tile |

The scale over-read is the bigger one, and it is simple arithmetic on the shapes:

| projection | rows N | K | packed bytes a row | scale bytes needed a row | scale bytes loaded a row | over-read |
|---|---|---|---|---|---|---|
| `gate_proj`, `up_proj` | 640 | 2560 | 1280 | 160 | 5 tiles x 256 B = 1280 | 8x |
| `down_proj` | 2560 | 640 | 320 | 40 | 2 tiles x 256 B = 512 | 12.8x |

Summed over one slot of one expert: **1.64 MB of packed weights against 2.13 MB of scale reads** (of which only
205 KB are useful). Fixing the access pattern alone therefore cuts the per-slot expert traffic from ~3.8 MB to
~1.85 MB — a factor of two, before any change to the arithmetic — and the row-axis layout is what makes it possible.
These are computed from the shapes and the load offsets, not measured on device.

Shapes to use (from the checkpoint header, `tensors/model-00001-of-00021.safetensors`):
`gate_proj`/`up_proj` weight `U8 [640, 1280]`, scale `F8_E4M3 [640, 160]` (K=2560); `down_proj` weight
`U8 [2560, 320]`, scale `F8_E4M3 [2560, 40]` (K=640); `weight_scale_2` is a **scalar F32 per expert**, which is the
`[E]` vector our kernel already takes. With `top_k=10` + the shared expert, `per_row=11` and `slots=11`.

---

## 6. Launch chaining and workspace sharing worth adopting

1. **One workspace for the whole chain.** Every recipe of the prefill/verify chain compiles against the same
   workspace name, so only the widest recipe's scratch is allocated:
   `hpu/src/layer.cpp:32-33` — `Every recipe of the prefill/verify chain runs on the chain stream, one at a time, so they share one workspace` and `hpu/src/layer.cpp:34` — `const std::string kChainWorkspace="verify_chain";`. Our port allocates per-op; the equivalent is one `torch.hpu` workspace / one vLLM graph pool for the whole layer.
2. **One stream, residual on device, one wait per host-visible scalar.** `hpu/include/strata/hpu/decode_chain.hpp:14-19`
   — `one decode token through every fused layer with the residual kept on the device`; the host waits only for
   the router logits and the QSA projections. Concretely for us: no `.item()`/`.cpu()` between the HC read and the
   next layer's projections.
3. **A decode step is one compiled recipe, not a dozen ops.** `hpu/src/layer.cpp:235-299` builds HC read → QSA
   projections → `qsa_prep` → indexer prep → block GEMM → `qsa_decode_blocks` → topk → `qsa_gather_blocks_fp8` →
   attention → output → HC write → FFN read → router → MoE → tail as **one** compiled graph
   (`strata_fused_qsa_decode_select_<blocks>`), with `dsel.cell` a *persistent 1-element tensor* so the token
   position is a device value, not a recompile key.
4. **Split by workspace, not by logic, when a graph would grow with the context.**
   `hpu/src/layer.cpp:508-519` splits the long-context QSA window into `pre / kv / chunk / post` "so no workspace
   grows with the keys", with `sel.*` buffers bound by the caller and all four sharing one workspace. Directly
   reusable for our prefill buckets.
5. **Double-buffered residual.** `DecodeChain::residual` is two buffers with `in()/out()/advance()`
   (`hpu/include/strata/hpu/decode_chain.hpp:26-36`), so no copy of `[HC,H]` F32 per layer.
6. **Weight leases instead of copies.** `std::vector<ExpertLease> leases` released after the *next* wait, because
   the stream orders them (`hpu/include/strata/hpu/decode_chain.hpp:18-19,29`). For us this is the recipe for
   streaming expert slabs in and out of HBM without stalling: enqueue, then release after the following wait.
7. **Internal tensors are allocated once per *name*, not once per recipe.** Every row count and key bucket compiles
   its own recipes, but they all bind the same six internal buffers — `attn_write.out`, `attn.mixed`, `ffn.mixed`,
   `o.out`, `gdn.out`, `router.logits` — because the recipes are launched one at a time on one stream
   (`hpu/include/strata/hpu/decode_chain.hpp:56-63`). This is the concrete answer to "one graph per bucket": the
   bucket count multiplies *compiled code*, not *HBM*.
8. **Split a layer at the attention/FFN boundary to bound the MME workspace.**
   `hpu/src/layer.cpp:39-42` — with a window bucket the layer graph ends at the post-attention residual and the FFN
   half runs as its own launch of **one** window FFN graph shared by every layer, which "keeps one MME workspace per
   device instead of one per layer graph shape". Directly applicable to our 48 layers × 2 FFN halves.

---

## 7. Interfaces

Five contracts a downstream implementer must obey. Violating any of them is silent (in-range indices, plausible
logits) rather than a crash.

1. **Weight plane contract (NVFP4, resident).** `packed` is `U8 [K/2, N, E]` with byte `k/2` holding column `k` in
   the **low** nibble and `k+1` in the high nibble, low nibble first; `scales` is `U8 [K/16, N, E]` E4M3 bytes;
   `scale2` is `F32 [E]`. Association is **Synapse outermost-first** (`E` is dim 2). Must hold:
   `scales.dim0 * 8 == packed.dim0` and `packed.dim0*2 == K`, K a multiple of 128
   (`vllm-gaudi/csrc/qwen4exp/host/qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2.cpp:56-63`).
   No consumer may materialize an expert as BF16 in the resident store, and `scale2` is applied **once**, after
   the whole K reduction.
   The *dimension order* is the one degree of freedom this contract leaves: today's planes are K-contiguous
   (`dim0 = K/2`, so a 256-byte load walks 512 k-values of one row), while the row-tiled GEMV of §5.2 needs
   `dim0 = N`. Either satisfies "4-bit resident, group-16 scale byte, one F32 scale2"; what must not happen is two
   kernels of one layer disagreeing about which it is. Declare it once per layer and derive both planes from it.
2. **PLE hash contract.** 16 row ids per token = `mixed % vocab[h] + offset[h]` for `h=(n-2)*8+g`,
   `n=2,3`, `mixed = XOR_j(token[j]*mult[j])` in `uint64`, predecessors **oldest first**, EOS cut
   **forward-propagating**, missing predecessor `-1`; the 16 rows flatten **head-slowest** into 2560 values.
   The table never enters HBM; only `w_key/w_value/norm_*/conv_weights` (~66 MB BF16) do.
3. **QSA selection contract.** Block score = `Σ_heads max(q·k, 0)`; order = (score desc, block id asc); the row's
   own incomplete block is either forced first (`+1e9`) or appended after the budget — pick one and keep
   `width_sel` consistent with it (ours: budget 2048 = 512 blocks, tail outside). Block pooled key = mean over 4
   raw keys, RMS norm with multiplicative `(1+w)` gamma, NeoX RoPE at the **block's first cell**. Cell indices are
   logical positions in the row's own history, and `-1` (or `-3e38`) means "not selected", never an index.
4. **HC contract.** Four streams. Read: `xn = RMSNorm_{HC·H}(residual, gamma)` (one norm over the stacked
   vector), `lo = BF16(s·σ(s))` with `s = down/HC` on the first `rank` outputs, `inject = down[rank:]`,
   `gate = w_up·lo`, `mixed = (1/HC)Σ_s xn[s]·σ(gate[s])`. Write: `out[s] = residual[s] + block·2σ(inject[s]/HC)`.
   The `2` and the `/HC` are part of the contract, not a detail.
5. **Batching contract.** Slots are `rows*per_row`, `per_row = top_k + 1`, the shared expert is the **last** slot
   of each row (`slot % per_row == top_k`), and the bucketing must place **every** routed slot (`ceil(total/rows)`
   tiles per expert) or report overflow — never drop one. Ours does not do this yet: the shared expert is a
   separate module whose output is added after the routed GEMV (`vllm-gaudi/vllm_gaudi/models/qwen4exp/model.py:210-211`),
   which is one extra launch and one extra HBM read of `[10240,2560]` per layer.

---

## 8. Verification log

All commands are host-only. Reproduce with:

```
.venv/bin/python validation/qwen4exp/tools/porting_anchors.py       # every anchor + quote in this file
.venv/bin/python validation/qwen4exp/tools/check_fp4_lut.py         # our E2M1 LUT == Strata's doubled table
.venv/bin/python validation/qwen4exp/tools/check_ple_hash_parity.py # PLE constants: checkpoint == Strata
.venv/bin/python validation/qwen4exp/tools/check_selection_width.py # the three selection widths, enumerated
.venv/bin/python -m pytest validation/qwen4exp -q                   # the whole host-only suite
```

### 8.1 What was checked by host tools, with the actual output

| claim | how it was checked | result |
|---|---|---|
| every `path:line` anchor in this file exists, is in bounds, and the quoted snippet is at that anchor | `tools/porting_anchors.py -q` | `251 anchors, 67 quoted snippets, 0 failures` |
| our E2M1 magnitude table equals STRATA's `packed_kvalues_fp4 / 2` | `tools/check_fp4_lut.py` | `16/16 codes agree after the x2 scaling` |
| STRATA's `packed_ue4m3` is half-scaled and unsigned relative to our E4M3FN scales | `tools/check_fp4_lut.py` | `254/254 non-NaN codes satisfy strata == \|ours\|/2`; `126` codes negative |
| both trees' model geometry agrees | `config.json` vs `include/strata/kernels/ngram.hpp:31-42` and `hpu/include/strata/hpu/hc_ple.hpp:4-8` | 2560 / 4 / 320 / 160 / 3 / 8 / 16 all agree |
| selection widths (D2/D3) — **corrected** | `tools/check_selection_width.py`: `strata_cells()` (`min(size,2051)`) vs `ours_cells()` (512 blocks + tail), swept over `size` 1..399999 | Strata keeps **2051**, ours **2048+r**; ours is never wider, the widest gap is **3 cells**, and the two coincide only at `size % 4 == 3` |
| the PLE hash constants are the *same object* in both trees | `tools/check_ple_hash_parity.py`: parses `src/kernels/ngram.cpp` and reads `tensors/*.safetensors` | `layer_multipliers`, `ngram_heads_vocab_sizes`, `ngram_heads_offsets` all identical; `128 × 2 500 012 = 320 001 536`; head ranges end **90** rows short, as Strata documents |
| the XOR row law differs from the sum rival, and EOS cuts forward | `tools/check_ple_hash_parity.py` | `XOR` vs `SUM` disagree on the same window; an EOS predecessor changes the trigram rows |
| the PLE table's global scale exists in the checkpoint | `tools/check_ple_hash_parity.py` | `weight_scale_2 = 3.32423624e-05` (F32 `[1]`) |
| the host-only test suite still passes on this host | `.venv/bin/python -m pytest validation/qwen4exp -q` | `63 passed, 14 warnings in 3.13s`. **A live number**: sibling tracks add cases to these four files while this map is being written (57, then 62, then 63 within one session), so treat the pass/fail as the claim, not the count |
| our decode GEMV is index-space rank 2 `[rows, slots]`, the dequant is rank 3 `[tiles, rows, slots]` | read of both host wrappers | confirmed; Strata's equivalent is rank 1 with `parts` in the index space |

Commands, verbatim (both finish in < 3 s):

```
$ .venv/bin/python validation/qwen4exp/tools/porting_anchors.py -q
251 anchors, 67 quoted snippets, 0 failures

$ .venv/bin/python validation/qwen4exp/tools/check_fp4_lut.py
E2M1  code : ours*2  strata
  16/16 codes agree after the x2 scaling

E4M3  254/254 non-NaN codes satisfy strata == |ours|/2
      findings: STRATA's rule is UNSIGNED (we decode 126 negative codes with the sign bit)
      and HALF-SCALED (the extra factor belongs to GGML's NVFP4 block, not to our planes).

$ .venv/bin/python validation/qwen4exp/tools/check_ple_hash_parity.py
ok   layer_multipliers  checkpoint [23703573157769, 20109073645365, 8052911324071] == STRATA [23703573157769, 20109073645365, 8052911324071]
ok   head_vocab_sizes   16 entries identical (dtype I64 [16])
ok   head_offsets       16 entries identical (dtype I64 [16])
ok   head_offsets == running sum of head_vocab_sizes (last end 320001446)
ok   128 shards x 2500012 rows = 320001536 == STRATA PLE_TABLE_ROWS 320001536
ok   head partition 320001446 + 90 alignment rows == 320001536 (Strata: 90)
ok   XOR and SUM readings disagree on one window (16759787 vs 16549217)
ok   all 16 rows inside their head's [offset, offset+vocab)
ok   an EOS predecessor changes the trigram rows (cut propagates)
ok   PLE table global scale present in the checkpoint: weight_scale_2 = 3.32423624e-05

0 failure(s)

$ .venv/bin/python validation/qwen4exp/tools/check_selection_width.py
block=4 top_k=2048 width_sel=2051

   size  r  strata   ours  delta
   2050  2    2050   2050      0
   2051  3    2051   2051      0
   2052  0    2051   2048      3
   2053  1    2051   2049      2
   4096  0    2051   2048      3
  65536  0    2051   2048      3
  65537  1    2051   2049      2
  65538  2    2051   2050      1
  65539  3    2051   2051      0

ok   ours is never wider than Strata's (sweep 1..399999)
ok   the widest gap is block-1 = 3 cells
ok   for rows longer than width_sel they coincide exactly when size % block == block-1
ok   a row of exactly top_k cells keeps all of them
ok   and so does one of width_sel cells (identity selection)

0 failure(s)

$ .venv/bin/python -m pytest validation/qwen4exp -q
63 passed, 14 warnings in 3.13s
```

The anchor checker resolves the two roots itself and covers both trees; adding a claim without an anchor, or
letting an anchor drift, makes it fail (the map cannot rot silently).

The two original tools are plain scripts rather than pytest files, and that is fine — but the reason given
earlier in this document ("this checkout's `.venv` has no pytest") is **wrong and has been corrected**: the venv
carries **pytest 9.2.0** on Python 3.12.3, and `.venv/bin/python -m pytest validation/qwen4exp -q` runs the whole
host-only suite green (`63 passed ... in 3.13s` at the last run). What was probably observed was `python3 -m pytest`, i.e. the
system interpreter rather than the venv's. New checks added by this revision are pytest-free scripts so they run
under either interpreter, but the suite itself is available and must stay green.

### 8.2 What could **not** be checked on this host

- Anything requiring the card: whether `sdpa_recomp_fwd_bf16` accepts our BF16 mask on this SynapseAI version,
  TPC compile time of the new kernels, achieved GEMV bandwidth, SDPA behaviour with `group`-folded rows, and every
  throughput number Strata's README quotes.
- Numerical equality between our bit-exact `E2M1×E4M3` product and the `doubled-int × (scale/2)` form: the two
  differ by one F32 rounding per 16-group **by construction**. Which one the checkpoint's reference used was not
  established here, so the recommendation in §4.6 keeps the form our hardware test already pins.
- Whether our `ple_embedding` shards carry a per-shard `weight_scale_2` beyond the `global_scale` the loader
  already takes (`vllm-gaudi/vllm_gaudi/models/qwen4exp/ple_runtime.py:456`): the index lists only `weight` +
  `weight_scale`, so the global scale has to come from the quantizer's config — **open item**.
- The GDN layers (36 of 48). They are out of scope for this map; Strata's `hpu/tpc/gdn_*.c` (36 files) and
  `hpu/src/gdn*.cpp` are where a follow-up track starts.

### 8.3 Open items handed to the implementer

1. **E4M3 half-scale.** Decide once whether our planes' `weight_scale_encoding: csf` is standard E4M3FN (as our
   kernel and `csf_math.e4m3_value_lut` assume, `2^(exp-7)`); if a reference logits comparison ever disagrees by a
   factor of two per group, this is the first place to look.
2. **`width_sel` choice** for the QSA select kernel (D2/D3): the recommended form drops `+1e9` and passes
   `width_sel = 2048`; the alternative keeps Strata's file untouched and costs up to one block per row.
3. **FP8 KV cache** (`STRATA_HPU_QSA_KV_FP8`): adopt only after the FP16/BF16 gather path is at parity, because it
   changes every gathered value.
