"""bundles2.py FILE.s: per basic block, true VLIW bundle count (brace groups + lone instruction lines) and VPU ops."""
import re, sys
VPU = re.compile(r'^(mac|mul|add|sub|and|or|xor|shr|shl|lookup_1c|lookup_2c|shuffle|sel_|mov_dg|convert|madd|cmp|msac|nearbyint|mov_dual|calc_fp_special|form_fp_num|get_lut_entry|fclass|extract_exp|pack|unpack|abs|max|min|not|popcnt|find_first|ash|asl|mov\.)')
def parse(path):
    blocks, name, cur, inb = [], 'entry', [], False
    for raw in open(path):
        line = raw.split('//')[0].rstrip()
        m = re.match(r'^(\.LBB\S+|main):', line)
        if m:
            blocks.append((name, cur)); name, cur = m.group(1), []; continue
        s = line.strip()
        if not s or s.startswith('.') or s.endswith(':'): continue
        if s == '{': inb = True; cur.append([]); continue
        if s == '}': inb = False; continue
        if inb: cur[-1].append(s)
        else: cur.append([s])
    blocks.append((name, cur))
    return blocks
for name, bundles in parse(sys.argv[1]):
    if len(bundles) < int(sys.argv[2] if len(sys.argv) > 2 else 8): continue
    flat = [i for b in bundles for i in b]
    vpu = sum(1 for i in flat if VPU.match(i) and '%V' in i or i.startswith('lookup') or i.startswith('mac.'))
    ld = sum(1 for i in flat if i.startswith('ld_tnsr')); spill = sum(1 for i in flat if i.startswith(('ld_l_v', 'st_l_v')))
    nop = sum(1 for b in bundles if all(x.lower() == 'nop' for x in b))
    print(f"{name:10s} bundles {len(bundles):4d}  vpu~{vpu:4d}  ld_tnsr {ld:3d}  spill {spill:3d}  nop-bundles {nop:3d}")
