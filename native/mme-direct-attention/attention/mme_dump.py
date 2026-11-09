# SPDX-License-Identifier: MIT
"""Convert a compiler MME descriptor dump (PRINT_MME_DESCRIPTORS) to the 182
register words written at MME QMAN offsets 0xb008..0xb2dc.

Generalizes direct/generate_desc.py (same field placement). Differences:
  * any shape; the GEMM128-only assertion is gone;
  * keep_slave=True keeps slave brains and the master<->slave handshake as the
    compiler emitted them (generate_desc.py cleared them for a one-master path);
  * base addresses (A, B, Cout0, Cout1) and so0Addr are left 0: the runtime fills them.
RATE_LIMITER (0x29c) is not in the text dump; the observed 0x040404 is used.
Perf/bias/user-data/padding words stay 0 (profiling and unused features).
"""
import re
import sys
from pathlib import Path

RATE_LIMITER = 0x040404
OFF_COUT1, OFF_COUT0, OFF_A, OFF_B, OFF_SO0 = 0x08, 0x10, 0x18, 0x20, 0x14C


def convert(text: str, keep_slave: bool = True) -> list:
    regs = dict.fromkeys(range(8, 0x2E0, 4), 0)
    sections = [
        ("Brains:", "Header:", 0x28, 2),
        ("Header:", "Ctrl:", 0x30, 2),
        ("Ctrl:", "Tensor: tensorA", 0x38, 2),
        ("Tensor: tensorA", "Tensor: tensorB", 0x40, 22),
        ("Tensor: tensorB", "Tensor: tensorCout", 0x98, 22),
        ("Tensor: tensorCout", "SyncObject:", 0xF0, 22),
        ("SyncObject:", "AguIn:", 0x148, 5),
    ]
    for start, end, offset, count in sections:
        block = text.split(start, 1)[1].split(end, 1)[0]
        values = re.findall(r"dw\[(\d+)\]\s*:\s*(0x[0-9a-f]+)", block)
        if [int(i) for i, _ in values] != list(range(count)):
            raise ValueError(f"Incomplete or unexpected {start} block")
        for i, v in values:
            regs[offset + int(i) * 4] = int(v, 16)
    for tag, base, count in [("aguIn", 0x15C, 5), ("aguOut", 0x22C, 2)]:
        entries = re.findall(tag + r"\[(\d+)\]\[(MASTER|SLAVE)\]: \[([^\]]+)\]", text)
        if {(int(i), side) for i, side, _ in entries} != {(i, s) for i in range(count) for s in ("MASTER", "SLAVE")} \
                or len(entries) != count * 2:
            raise ValueError(f"Incomplete or duplicate {tag} entries")
        for i, side, array in entries:
            values = [int(x.strip()) for x in array.split(",")]
            if len(values) != 5:
                raise ValueError("An AGU entry must have five offsets")
            for j, v in enumerate(values):
                regs[base + int(i) * 40 + (20 if side == "SLAVE" else 0) + j * 4] = v & 0xFFFFFFFF
    for field, offset in [("A", 0x224), ("B", 0x228), ("Cout", 0x27C)]:
        regs[offset] = int(re.search("spatialSizeMinus1" + field + r" : (0x\w+)", text)[1], 16)
    conv = text.split("Conv:", 1)[1].split("OuterLoop:", 1)[0]
    regs[0x280] = int(re.search(r"dw\[0\]: (0x\w+)", conv)[1], 16)
    halves = [int(v, 16) for _, v in re.findall(r"w\[([1-4])\]: (0x\w+)", conv)]
    if len(halves) != 4 or any(v > 0xFFFF for v in halves):
        raise ValueError("Invalid convolution association words")
    regs[0x284] = halves[0] | (halves[1] << 16)
    regs[0x288] = halves[2] | (halves[3] << 16)
    outer = text.split("OuterLoop:", 1)[1].split("numIterationsMinus1", 1)[0]
    regs[0x28C] = int(re.search(r"dw: (0x\w+)", outer)[1], 16)
    regs[0x290] = int(re.search(r"numIterationsMinus1 : (0x\w+)", text)[1], 16)
    repeat = text.split("SBRepeat:", 1)[1].split("Fp8Bias:", 1)[0]
    regs[0x294] = int(re.search(r"dw: (0x\w+)", repeat)[1], 16)
    regs[0x29C] = RATE_LIMITER
    if not keep_slave:
        regs[0x28] &= ~0x80808080
        regs[0x2C] &= ~0x808080
        regs[0x148] &= ~(3 << 15)
    for off in (OFF_COUT1, OFF_COUT1 + 4, OFF_COUT0, OFF_COUT0 + 4, OFF_A, OFF_A + 4, OFF_B, OFF_B + 4, OFF_SO0):
        regs[off] = 0
    return [regs[i] for i in range(8, 0x2E0, 4)]


def load_set(directory: Path, keep_slave: bool = True):
    """All act_<a>_mme_descriptor_dump_<d>.txt in a capture: [(activation, engine, words)]."""
    out = []
    for path in sorted(directory.glob("act_*_mme_descriptor_dump_*.txt")):
        a, d = map(int, re.findall(r"act_(\d+)_mme_descriptor_dump_(\d+)", path.name)[0])
        out.append((a, d, convert(path.read_text(encoding="utf-8-sig"), keep_slave)))
    return sorted(out)


def coalesce_qk(descs):
    """Coalesce consecutive width activations with identical non-offset state.

    Pinned 128/256/512 x N x 128 BF16 QK geometry only (512: 2026-11-09). Preserve master/slave,
    tensor bounds, strides, partial ROI and all control bits. Tail ROI changes
    start a new group. Only low-byte Conv loop / SB A repeat and the completion
    count change. No register reads or runtime field guessing.
    """
    if not descs:
        raise ValueError("Empty QK descriptor set")
    ix = lambda off: (off-8)//4
    acts = sorted(set(a for a, _, _ in descs))
    if acts != list(range(len(acts))) or len(descs) != 2*len(acts):
        raise ValueError("Expected consecutive paired QK activations")
    offsets = (0x184,0x198,0x1ac,0x1c0,0x1d4,0x1e8,0x1fc,0x210,0x22c,0x240,0x254,0x268)
    shifted = (0x22c,0x240,0x254,0x268)
    groups, previous, group = [], None, []
    prefix = 0
    for act in acts:
        pair = sorted([(e,list(w)) for a,e,w in descs if a == act])
        if [e for e,_ in pair] != [0,1]:
            raise ValueError("Unexpected MME engine set")
        signatures = []
        loop = pair[0][1][ix(0x280)]+1
        step = pair[0][1][ix(0x104)]
        if not 1 <= loop <= 256 or step not in (256,512,1024):
            raise ValueError("Unsupported QK loop geometry")
        for e,w in pair:
            ref = next(ws for aa,ee,ws in descs if aa==0 and ee==e)
            if w[ix(0x40)] != 128 or w[ix(0x44)] not in (128*128,256*128,512*128) or w[ix(0x28)] != ref[ix(0x28)] or w[ix(0x104)] != step or w[ix(0x280)]+1 != loop or w[ix(0x294)] != loop-1:
                raise ValueError("Unsupported QK reuse/geometry")
            sig = list(w)
            for off in offsets:
                delta = w[ix(off)] - ref[ix(off)]
                if delta not in (0,prefix) or (off in shifted and delta != prefix):
                    raise ValueError("QK AGU does not match width prefix")
                sig[ix(off)] -= delta
            sig[ix(0x280)] = sig[ix(0x294)] = 0
            signatures.append(sig)
        if signatures != previous or sum(x[1] for x in group)+loop > 256:
            if group: groups.append(group)
            group = []
        group.append((pair,loop))
        previous = signatures
        prefix += loop*step
    groups.append(group)
    width = descs[0][2][ix(0xf0)]
    if prefix != ((width+step-1)//step)*step:
        raise ValueError("QK activations do not cover exactly the logical width")
    result = []
    for act,group in enumerate(groups):
        total = sum(x[1] for x in group)
        for e,w in group[0][0]:
            w=list(w);w[ix(0x280)]=w[ix(0x294)]=total-1
            result.append((act,e,w))
    return result


def repeat_rows_512(descs):
    """Two M512 row tiles using the existing Conv[1] A1/B5/Out1 loop.

    No output/input-base guessing: retain spatial/ROI M512 geometry, let the
    row loop advance the pinned 512-row A/C strides. B does not advance.
    Must be checked with a complete M1024 GEMM before use in attention.
    """
    ix=lambda off:(off-8)//4
    result=[]
    for act,e,orig in descs:
        w=list(orig)
        assert w[ix(0x280)]<256 and w[ix(0x290)]==0
        assert w[ix(0x284)]>>16==0x69, "Expected A1/B5/Out1 Conv[1] association"
        assert w[ix(0x98+6*4)]==0, "B must not advance in row loop"
        for ten in (0x40,0xf0):
            cols=w[ix(ten)]
            assert w[ix(ten+4)]==512*cols
            assert w[ix(ten+6*4)]==512*cols
            assert w[ix(ten+11*4)]==512*cols
            assert all(w[ix(ten+4*j)]==512*cols for j in range(1,5))
            for j in range(1,5):w[ix(ten+4*j)]*=2
        w[ix(0x280)]|=1<<8
        result.append((act,e,w))
    return result


if __name__ == "__main__":
    # Self-check: GEMM128 fixture with keep_slave=False must reproduce desc128.h (except base/SO words).
    fixture, header = Path(sys.argv[1]), Path(sys.argv[2])
    words = convert(fixture.read_text(encoding="utf-8-sig"), keep_slave=False)
    ref = [int(x, 16) for x in re.findall(r"0x([0-9a-f]{8}),", header.read_text())]
    skip = {(o - 8) // 4 for o in (OFF_COUT1, OFF_COUT1 + 4, OFF_COUT0, OFF_COUT0 + 4, OFF_A, OFF_A + 4, OFF_B,
                                   OFF_B + 4, OFF_SO0)}
    diff = [(hex(8 + 4 * i), hex(a), hex(b)) for i, (a, b) in enumerate(zip(words, ref)) if a != b and i not in skip]
    print("words", len(words), "ref", len(ref), "differences outside base/SO:", diff)
