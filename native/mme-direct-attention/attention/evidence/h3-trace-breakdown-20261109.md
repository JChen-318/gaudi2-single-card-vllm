# H3 DiT / VAE trace breakdown (2026-11-09 evening)

Run `video-trace-base-20261109c`: baseline mode (FusedSDPA-family attention, no direct
path, so the profiler never runs beside the private SCAL stream), `H3_TE_PIN=1`,
`H3_PHASE_OFFLOAD=0` (VAE tile batch 4), DiT call 5 (request 1, forward 2) and the request 1
video decode traced with torch.profiler CPU+HPU. Analysis scripts: `trace_summary.py`
(lanes), `trace_kernels.py` (arg grouping), `trace_recipes.py` (wall-clock union per compiled
graph). Outputs in `video-trace-base-20261109c/` (`dit-recipes.txt` is the main one).
Untraced request 2 of the same run: 95.88 s (old baseline 100.45 s), DiT 4 x 20.09 s.

## DiT forward, baseline (device span 20.10 s, any-engine busy 19.91 s, idle 0.19 s)

| Group (wall, union over all engines) | ms / forward |
|---|---:|
| Max-free attention graphs (15 recipes x 996 ms + 1 x 740 ms, bmm + fused exp TPC) | 15,680 |
| FusedSDPA `sdpa_recomp` (layers 45/48/49) | 1,160 |
| FP8 linear graphs (fp8_gemm + per-token abs/amax/scale TPC) | about 2,210 |
| TPC-only graphs (tile/mult modulation, gather, casts, norms) | about 920 |
| Many small graphs (add/cast, about 2 ms each) | about 930 |
| Idle | 190 |

- Non-attention DiT work is about 4.06 s busy + 0.19 s idle per forward, about 17 s per
  request (not the earlier estimate of about 20 s including attention).
- The three outlier layers cost about 1.16 s per forward, 4.6 s per request, in both modes.
- In direct mode (16.05 s per forward): 16.05 - 1.16 - 4.25 = about 10.6 s per forward for
  direct attention including gates/staging.
- FP8 linears: MME union about 1.93 s per forward. The TPC side of those graphs includes
  per-token dynamic quantisation (abs, amax, scale).
- Largest pure-TPC graph (543 ms): tile_fwd_bf16 (broadcast materialisation), mult, fused
  f32 kernels; consistent with adaLN modulation. Also add_bf16 (262 ms per lane),
  cast_bf16_to_f32 (177 ms), memcpy (166 ms), reduce_L2 (154 ms) across the forward.

## VAE decode (partial)

The VAE trace holds only about 1.6 s of device events for an 11.7 s decode (about 190k
events per TPC lane, device trace buffer full). In the captured part: TPC dominated by
softmax_fwd_hf8 (76,160 calls, about 4.7 us each) and MME by BatchGemm (attention), i.e. VAE
attention launches very many tiny kernels. CPU side: 11.1 s in aten::to/copy_ (frame D2H
copies, which also absorb the wait for device work). This trace used VAE tile batch 4, not
the production 28. A full VAE breakdown needs a shorter capture window (for example one
decoder call) or a larger device trace buffer.

## Follow-up (same evening)

### VAE tile batch (run `video-direct-pin-tile4-20261109d`)
Phase offload kept, only `H3_VAE_TILE_BATCH=4`: video MD5 of out1/out2 identical to run B
(`video-direct-pinres-20261109b`), audio identical to all runs. The run-B video difference
is caused only by the tile batch; DiT residency is bit-exact. Single-call decoder traces
(`vae-dec-tr20261109b`, `vae_decoder_trace.sh`): batch 4 first tile is bit-identical to
batch 1 (sequential reference); batch 14 is not (4.5M differing values). So tile 4 matches
the unbatched decoder and the production 28-tile path is the numerically deviating one.
Decoder time per tile: 77.8 ms (b1), 60.2 ms (b4), 55.2 ms (b14).

### VAE decoder composition (one call, batch 4, 238.7 ms, idle 17.3 ms)
Whole decoder is one compiled graph (transformer decoder; convolutions negligible).
MME lane D0: FFN w1 49.1 + w2 29.9 ms, attention (FusedSDPA) 48.2 ms, to_qkv 17.3, to_out
6.8 ms. TPC lane: softmax_fwd_hf8 52.8 ms (3,168 calls, about 17 us each), then tile,
reduce_mean (norms), fused SwiGLU/RoPE kernels about 10-12 ms each. Attention is about 40%
of the decoder, FFN about 33%. Batch 14 has the same mix.

### DiT small graphs (`dit-small-graphs.txt`, `dit-gemm-names.txt`, `dit-cpu-shapes.txt`)
1,893 small-graph launches, 1.59 s wall per forward; inter-graph idle only 142 ms. Most are
eager single ops (`n_aa/n_ab/n_ac`). The largest identifiable item is the dynamic PEFT LoRA
(LightX2V, rank 128) applied eagerly in BF16 after each FP8 base linear
(`vllm_omni/diffusion/lora/layers/base_linear.py::apply`): per forward 250 x
`mm [37760,5376]@[5376,128]` and 250 x `addmm` into the base output
(`[37760,128]@[128,7168 or 14336]`, out= slice). The addmm runs as MME gemm + a separate TPC
add over the full output slice (about 1-1.5 ms each): roughly 0.45 s per forward, 1.8 s per
request. Other eager items in baseline mode are attention-side (norms, casts, cat padding
37750 -> 37760, copies of [1,56,37750,128]).
