# Issue #4: atomic valid-length update (2026-11-12)

GitHub issue #4 (gaudi2-single-card-vllm): a failed `set_valid` could leave the two prepared slots on different
valid lengths, while Python kept its old length and later accepted that length without asking the native side.

## Change

- `runtime_pipeline.hpp`: `reupload_ecb` split into `stage_ecb_image` (build the code image and check its layout
  against the prepared job, no device access) and `write_ecb_image` (write and verify by readback; does not touch
  `job.code_image`).
- `runtime_quad.hpp` `context_quad_set_valid`, all-or-nothing:
  1. build exp + normalize programs of both slots and stage all images; any error here returns before a write;
  2. write the images that differ;
  3. on a failed write, rewrite the previous image into every job written so far and into the failing one;
     plan lengths unchanged, the error is returned;
  4. if that restore fails, the plan is poisoned: `set_valid` and `submit2` return `-ENOTRECOVERABLE` until
     `context_quad_release`;
  5. only after all writes succeed: swap the images into the jobs and set both slot lengths.
- Test-only hook `context_quad_test_fail_write(nth, restore)`: fail the nth write of the next update (and every
  restore write). Off by default, cleared after one update and by release.
- `h3_direct_attention.py` `_Plan.set_valid`: on any failure `self.valid = None`, so the next request of any
  length calls the native side (a no-op when the programs already match). Message no longer says "disabled".
  The call site already prepared a new plan at the requested length when `set_valid` fails (d4432a8).

## Results

CPU (`issue4_mock_test.py`, runs the real `_Plan.set_valid` source against mocked native updates, as in the
issue): fixed code 10/10 pass; the d4432a8 code fails 6, including the issue's scenario (old length accepted
from the stale Python value without a native call).

Device (`issue3_run.sh 25` with `H3DIRECT_KEYS=auto`, log `issue4-layer25-auto.log`; module 7 idle, no other
container on the card). The rebuilt bridge; all 52 TPC kernel binaries byte-identical to
`backup-pre-issue3-20261111/bins.sha256`.

| test | result |
|---|---|
| t1 L=37750 | SHA `0b66cf40…` = expected; 56/56 heads direct |
| t2 37760 / 37700 / 28800 / back to 37750 | all direct, FP32 reference error 0.23–0.25 %; 37750 SHA back to baseline |
| t3 counterexamples | 2 heads rejected, output exactly 1, other heads bit-identical |
| t6a 2nd of 4 writes fails | native: `write=2/4: restored valid=37750/37750`; Python valid → None; next 37750 call: 0 writes, no rebuild, SHA = t1 |
| t6b write and restore fail | `-ENOTRECOVERABLE` from set_valid and submit2 (nothing queued); next call released and rebuilt the plan (1 rebuild), SHA = t1 |
| t6c 37700 afterwards | 4 writes, SHA = the t2 37700 result |
| t6d back to 37750 | SHA = t1 |
| t5 bucket switches 37859 … 41344, 41472, 37750 | all direct up to 41344 (FP32 error 0.23–0.24 %); 41472 falls back as before; 37750 after the switches SHA = t1 |
| t4 MAXFREE guard | counterexample 1.0 → 0.0 error; real layer bit-identical (323.4 vs 323.8 ms) |

The success path writes the same images as before (only all builds now precede all writes), so the output
SHAs are unchanged; no new video run was made.

Not measured: a negative control showing that a missing restore changes the 37750 output (slot 0 would keep
the 37700 exp program; expected to differ, not run).
