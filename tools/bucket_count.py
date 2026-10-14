"""Replicates vllm_gaudi decode bucket generation (linear strategy, non-contiguous PA, spec decode T=8)."""
import itertools, math, operator, sys
BLOCK, MAXLEN, T = 128, 262144, 8
def warmup_range(bmin, bstep, bmax):
    ramp = itertools.takewhile(lambda x: x < bstep and x <= bmax, itertools.accumulate(itertools.repeat(2), operator.mul, initial=bmin))
    return sorted([b for b in list(ramp) + list(range(bstep, bmax + 1, bstep)) if b >= bmin])
def gen(bs_cfg, blk_cfg):
    corr = lambda bs, q, ctx: (bs, q, min(ctx, bs * math.ceil(MAXLEN / BLOCK)))
    seed = sorted({corr(bs, 1, c) for bs in warmup_range(*bs_cfg) for c in warmup_range(*blk_cfg) if bs <= c})
    out = set(seed)
    for bs, q, ctx in seed:
        sb = bs * T
        if sb <= ctx: out.add(corr(sb, q, ctx))
        out.add(corr(sb, q, ctx * T))
    return seed, sorted(out)
for name, blk in (("now  (max 512 pages)", (1, 128, 512)), ("256K single (max 2048)", (1, 128, 2048)), ("256K + any 2-way mix (max 4096)", (1, 128, 4096)),
                  ("256K single, step 256", (1, 256, 2048)), ("256K single, step 512", (1, 512, 2048))):
    seed, allb = gen((1, 3, 2), blk)
    spec = [b for b in allb if b not in set(seed)]
    by = {bs: [c for b, _, c in allb if b == bs] for bs in sorted({b for b, _, _ in allb})}
    print(f"{name:32} total {len(allb):3d}  seed {len(seed):3d}  spec {len(spec):3d}  " +
          "  ".join(f"bs{bs}: {len(v)} (max {max(v)})" for bs, v in by.items()))
    if "256K single (max 2048)" == name:
        print("   bs8 ctx:", by[8])
