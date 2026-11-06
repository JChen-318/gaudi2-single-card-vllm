# SPDX-License-Identifier: Apache-2.0
"""Cross-check the two 4-bit codecs the port has to reconcile, on the host.

1. **E2M1.** STRATA's doubled integer table ``hpu/tpc/fp4_values.inc`` must equal
   ``2 *`` our float table ``csf_math.e2m1_value_lut()``. If it does, the
   integer-LUT form of the NVFP4 GEMV (int8 MAC, then one F32 multiply by
   ``scale/2`` per 16-group) needs no second codebook.
2. **E4M3.** STRATA's scalar ``packed_ue4m3`` rule (``hpu/tpc/packed_matmul.c``,
   transcribed verbatim below) is checked against our
   ``csf_math.e4m3_value_lut()`` for the *relationship* it must have, not for
   equality: STRATA's NVFP4 path is GGML's **UE4M3, unsigned and scaled by one
   extra half** (its own comment: "the helper's closing half"), so
   ``strata_ue4m3(code & 0x7f) == abs(ours(code)) / 2`` must hold for every one
   of the 254 non-NaN codes. Both deviations are reported as findings: copying
   ``packed_ue4m3`` verbatim into our kernels would halve every group scale and
   drop the sign of the 126 negative codes.

Loads ``csf_math.py`` directly by path (``importlib``) so the ``vllm_gaudi``
package ``__init__`` — and with it the HPU platform plugin — never runs.
Host-only, ~0.2 s, no device.

Usage::

    .venv/bin/python validation/qwen4exp/tools/check_fp4_lut.py
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

STRATA = Path(os.environ.get("STRATA_ROOT", "~/gaudi2-single-card-strata")).expanduser()
PORT = Path(__file__).resolve().parents[3]
CSF = PORT / "vllm-gaudi/vllm_gaudi/models/qwen4exp/csf_math.py"


def load_csf():
    spec = importlib.util.spec_from_file_location("qwen4exp_csf_math", CSF)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def strata_doubled_e2m1() -> list[int]:
    text = (STRATA / "hpu/tpc/fp4_values.inc").read_text(encoding="utf-8")
    body = re.search(r"packed_kvalues_fp4\[16\]\s*=\s*\{(.*?)\}", text, re.S)
    if body is None:
        raise SystemExit("could not parse packed_kvalues_fp4 from fp4_values.inc")
    values = [int(v) for v in re.findall(r"-?\d+", body.group(1))]
    if len(values) != 16:
        raise SystemExit(f"expected 16 codes, parsed {len(values)}")
    return values


def strata_ue4m3(x: int) -> float:
    """hpu/tpc/packed_matmul.c:469-477, transcribed (note: bit 7 unused)."""
    if x == 0 or x == 0x7F:
        return 0.0
    exp = (x >> 3) & 15
    man = x & 7
    e = -10 if exp == 0 else exp - 8
    scale = 1.0
    for _ in range(e):
        scale += scale
    for _ in range(-e):
        scale *= 0.5
    return (man if exp == 0 else 1.0 + man * 0.125) * scale


def main() -> int:
    csf = load_csf()
    e2m1_bad = 0
    sign_delta = 0
    e4m3_bad = 0

    ours = csf.e2m1_value_lut()
    doubled = [int(round(float(v) * 2)) for v in ours.tolist()]
    theirs = strata_doubled_e2m1()
    print("E2M1  code : ours*2  strata")
    for code, (a, b) in enumerate(zip(doubled, theirs)):
        if a != b:
            e2m1_bad += 1
            print(f"  BAD {code:2d} : {a:6d}  {b:6d}")
    print(f"  {16 - e2m1_bad}/16 codes agree after the x2 scaling\n")

    table = csf.e4m3_value_lut()
    checked = 0
    for code in range(256):
        if (code & 0x7F) == 0x7F:  # E4M3FN NaN
            continue
        ours_value = float(table[code])
        strata_magnitude = strata_ue4m3(code & 0x7F)
        checked += 1
        if abs(abs(ours_value) / 2.0 - strata_magnitude) > 1e-12 * max(1.0, strata_magnitude):
            e4m3_bad += 1
            print(f"  BAD e4m3 0x{code:02x}: |ours|/2 {abs(ours_value) / 2!r} strata {strata_magnitude!r}")
        if ours_value < 0.0:
            sign_delta += 1
    print(f"E4M3  {checked - e4m3_bad}/{checked} non-NaN codes satisfy strata == |ours|/2")
    print(f"      findings: STRATA's rule is UNSIGNED (we decode {sign_delta} negative codes with the sign bit)")
    print("      and HALF-SCALED (the extra factor belongs to GGML's NVFP4 block, not to our planes).")
    return 1 if (e2m1_bad or e4m3_bad) else 0


if __name__ == "__main__":
    sys.exit(main())
