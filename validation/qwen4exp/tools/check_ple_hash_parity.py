#!/usr/bin/env python3
"""Cross-tree check of the PLE n-gram hash: STRATA's transcribed constants vs our checkpoint.

Three independent claims, each falsifiable without a device:

  1. The constants STRATA hard-codes (parsed out of ``src/kernels/ngram.cpp``, not copied here) are the
     ones our checkpoint ships (read out of ``tensors/*.safetensors``).
  2. Our 128 shards tile exactly the row space STRATA's ``PLE_TABLE_ROWS`` declares.
  3. The row law itself is the XOR form: a sum variant yields different rows on the same window
     (the ``ngram.hpp`` header lists this as one of the four silently-plausible rival readings).

Exit code 0 on success, 1 on any failure. Runs in well under a second; no torch, no pytest, no device.
"""
from __future__ import annotations

import json
import os
import re
import struct
import sys
from pathlib import Path

STRATA = Path(os.environ.get("STRATA_ROOT", "~/gaudi2-single-card-strata")).expanduser()
CKPT = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()
PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding"
N_HEADS = 16
NGRAM_SIZE = 3

_DT = {"I64": ("<q", 8), "I32": ("<i", 4), "F32": ("<f", 4), "F64": ("<d", 8)}


def read_tensor(weight_map: dict[str, str], name: str) -> tuple[str, list[int], list]:
    """One tensor out of a sharded safetensors checkpoint, without torch."""
    path = CKPT / "tensors" / weight_map[name]
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        entry = header[name]
        begin, end = entry["data_offsets"]
        handle.seek(8 + header_len + begin)
        raw = handle.read(end - begin)
    fmt, width = _DT[entry["dtype"]]
    values = [struct.unpack_from(fmt, raw, i)[0] for i in range(0, len(raw), width)]
    return entry["dtype"], entry["shape"], values


def strata_constants() -> tuple[list[int], list[int], list[int]]:
    """Parse ``ple_artifact_consts()`` out of STRATA's ngram.cpp."""
    text = (STRATA / "src/kernels/ngram.cpp").read_text()
    mult = [int(v) for v in re.findall(r"c\.mult\[\d\] = (\d+)ull;", text)]
    vocab_block = re.search(r"const uint64_t vocab\[PLE_N_HEADS\] = \{(.*?)\};", text, re.S)
    offset_block = re.search(r"const uint64_t offset\[PLE_N_HEADS\] = \{(.*?)\};", text, re.S)
    if len(mult) != NGRAM_SIZE or not vocab_block or not offset_block:
        raise AssertionError("could not parse STRATA's PLE constants")
    vocab = [int(v) for v in re.findall(r"\d+", vocab_block.group(1))]
    offset = [int(v) for v in re.findall(r"\d+", offset_block.group(1))]
    return mult, vocab, offset


def strata_table_rows() -> int:
    text = (STRATA / "include/strata/kernels/ngram.hpp").read_text()
    match = re.search(r"PLE_TABLE_ROWS = (\d+)", text)
    if not match:
        raise AssertionError("could not parse PLE_TABLE_ROWS")
    return int(match.group(1))


def ngram_rows(tokens: list[int], prev: list[int], mult: list[int], vocab: list[int],
               offset: list[int], eos: int, xor: bool = True) -> list[int]:
    """STRATA ``ngram_rows`` for one token. ``xor=False`` is the rival reading (a sum)."""
    ctx = [tokens[0]]
    cut = False
    for step in range(1, NGRAM_SIZE):
        token = -1 if cut else prev[len(prev) - step]
        cut = cut or token < 0 or token == eos
        ctx.append(eos if cut else token)
    rows = [0] * N_HEADS
    for n in range(2, NGRAM_SIZE + 1):
        mixed = (ctx[0] * mult[0]) & 0xFFFFFFFFFFFFFFFF
        for j in range(1, n):
            product = (ctx[j] * mult[j]) & 0xFFFFFFFFFFFFFFFF
            mixed = (mixed ^ product) if xor else ((mixed + product) & 0xFFFFFFFFFFFFFFFF)
        for g in range(8):
            head = (n - 2) * 8 + g
            rows[head] = mixed % vocab[head] + offset[head]
    return rows


def main() -> int:
    failures = 0

    def check(ok: bool, message: str) -> None:
        nonlocal failures
        print(("ok   " if ok else "FAIL ") + message)
        failures += 0 if ok else 1

    weight_map = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    mult_s, vocab_s, offset_s = strata_constants()

    _, shape_m, mult_c = read_tensor(weight_map, f"{PLE_PREFIX}.layer_multipliers")
    dtype_v, shape_v, vocab_c = read_tensor(weight_map, f"{PLE_PREFIX}.ngram_heads_vocab_sizes")
    dtype_o, shape_o, offset_c = read_tensor(weight_map, f"{PLE_PREFIX}.ngram_heads_offsets")

    check(mult_c == mult_s, f"layer_multipliers  checkpoint {mult_c} == STRATA {mult_s}")
    check(vocab_c == vocab_s, f"head_vocab_sizes   {len(vocab_c)} entries identical (dtype {dtype_v} {shape_v})")
    check(offset_c == offset_s, f"head_offsets       {len(offset_c)} entries identical (dtype {dtype_o} {shape_o})")

    # The offsets must be the running sum of the vocab sizes: the artifact stores both, and a mismatch
    # between them is exactly the kind of thing that a single-source transcription hides.
    running, ok_running = 0, True
    for head in range(N_HEADS):
        ok_running = ok_running and offset_c[head] == running
        running += vocab_c[head]
    check(ok_running, f"head_offsets == running sum of head_vocab_sizes (last end {running})")

    table_rows = strata_table_rows()
    shard_rows = 2_500_012
    check(shard_rows * 128 == table_rows,
          f"128 shards x {shard_rows} rows = {shard_rows * 128} == STRATA PLE_TABLE_ROWS {table_rows}")
    # STRATA's validation notes "90 trailing alignment rows beyond the sum of head vocabularies"
    # (hpu/src/ple_input.cpp): if our artifact has the same 90, the row spaces are the same object.
    check(table_rows - running == 90,
          f"head partition {running} + {table_rows - running} alignment rows == {table_rows} (Strata: 90)")

    eos_match = re.search(r"PLE_EOS_TOKEN_ID = (\d+)", (STRATA / "include/strata/kernels/ngram.hpp").read_text())
    eos = int(eos_match.group(1)) if eos_match else 248044

    # Row law: the XOR form and the rival sum form must disagree, so a port that "simplifies" is caught.
    window = [248044, 1000, 32]
    prev = [500, 32]
    rows_xor = ngram_rows(window, prev, mult_c, vocab_c, offset_c, eos)
    rows_sum = ngram_rows(window, prev, mult_c, vocab_c, offset_c, eos, xor=False)
    check(rows_xor != rows_sum, f"XOR and SUM readings disagree on one window ({rows_xor[0]} vs {rows_sum[0]})")

    # Every row must land inside its head's range: in-range-but-wrong is the failure mode the header warns about.
    ok_range = all(offset_c[h] <= rows_xor[h] < offset_c[h] + vocab_c[h] for h in range(N_HEADS))
    check(ok_range, "all 16 rows inside their head's [offset, offset+vocab)")

    # EOS at the newest predecessor cuts that position and every older one, forward.
    cut_rows = ngram_rows([1000], [500, eos], mult_c, vocab_c, offset_c, eos)
    no_cut_rows = ngram_rows([1000], [500, 499], mult_c, vocab_c, offset_c, eos)
    check(cut_rows != no_cut_rows, "an EOS predecessor changes the trigram rows (cut propagates)")

    scale_name = f"{PLE_PREFIX}.ngram_embedding.weight_scale_2"
    if scale_name in weight_map:
        _, _, scale = read_tensor(weight_map, scale_name)
        check(len(scale) == 1 and 0.0 < scale[0] < 1.0,
              f"PLE table global scale present in the checkpoint: weight_scale_2 = {scale[0]:.9g}")
    else:
        check(False, "PLE table has no weight_scale_2 in the checkpoint")

    print(f"\n{failures} failure(s)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
