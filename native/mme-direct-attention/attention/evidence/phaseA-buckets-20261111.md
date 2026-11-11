# Phase A: automatic key-capacity buckets (2026-11-11)

Goal: every prompt length at 1344x768 / 124 frames runs the direct path, in one service, with results that do
not depend on earlier requests.

## Changes (not yet pushed)
- Native: `context_quad_release()` frees both slots' ECBs and all head banks (scal_free_buffer via
  `release_ecb`), clears MME sets and the wide K/V binding; option setters stay. `load_runtime_isa` caches
  kernel binaries by path and content. Layout cap 41472 (largest SRAM fit at 1024 queries).
- Python (`h3_direct_attention.py`): `BUCKETS = 37760, 38272, ..., 41344` (one every 512 keys; text <= 50, 562,
  1074, 1586, 2098, 2610, 3122, 3634 tokens). `H3DIRECT_KEYS=auto` (default) picks the smallest bucket holding L
  and rebuilds the plan when the bucket changes; a number pins one bucket. The private SCAL stream and LUT are
  initialised once per process (calling them again fails / leaks). Every bucket keeps the last key quarter
  narrower than the first; with equal quarters (41472) the exp setup deduplicated equal column counts at
  L = 41472, the TPC program size changed and the in-place rewrite was refused (-E2BIG). If a rewrite is ever
  refused, the plan is rebuilt at that length.
- New compiler captures: widths 9856, 9984, 10112, 10240, 10368 (QK and PV); 9728 reused from an earlier
  capture (verified by the tests below).

## Layer 25 (`issue3-layer25-auto8.log`, H3DIRECT_KEYS=auto)
t1 37750 SHA 0b66cf40 (pre-fix value). t5 switches 37859 -> 38317 -> 39000 -> 39500 -> 40000 -> 40500 ->
41000 -> 41344 -> 41472 -> 37750: buckets 38272, 38784, 39296, 39808, 40320, 40832, 41344, 41344, none, 37760;
all in-range lengths 56/56 heads direct with 0.23-0.24% worst error vs CPU FP32; plan rebuild 0.82-1.34 s;
each release frees 1796 ECBs; 41472 falls back to the guarded max-free path; back at 37750 the SHA equals t1.
t3/t4 unchanged (counterexamples exactly 1; max-free guard bit-identical).

## Videos (outputs on /mnt/sas/tmp/mme-direct-run)
`video-phaseA-auto-20261111a` (4 prompts in one service, before the 512-key buckets):

| Request | Prompt | L / bucket | Total | Steady DiT forward |
|---|---|---|---:|---:|
| 0 warmup | forest 40 tokens | 37750 / 37760 | 98.3 s | 16.0 s |
| 1 | harbor 607 tokens | 38317 / 41344 | 96.3 s | 18.1 s |
| 2 | fox 149 tokens | 37859 / 38272 | 88.9 s | 16.1 s |
| 3 | forest 40 tokens | 37750 / 37760 | 80.9 s | 16.0 s |

752/752 calls direct. Request 3 video/audio MD5 4cb6c8ad... / 1ee99358... = `video-direct-pinres-20261109b/out1`
(same seed, fixed 37760 plan): bucket switching leaves results bit-identical. First forwards after a new length
include the framework's graph compiles for that length (once per process per length).

`video-phaseA2-auto-20261111a` (512-key buckets; forest warmup, harbor prompt twice): the 607-token prompt now
uses the 38784 bucket; steady request 81.46 s, DiT 16.45-16.49 s per forward (was 18.1 s in the 41344 bucket),
564/564 calls direct; frames match the prompt.
