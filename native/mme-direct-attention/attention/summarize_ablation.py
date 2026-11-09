# SPDX-License-Identifier: MIT
# Compact one-line summary of RESULT_REAL_BATCH / RESULT_PROFILE_ONLY logs.
import json,sys,glob,os
for f in sys.argv[1:]:
    for line in open(f,encoding='utf-8',errors='replace'):
        for tag in ("RESULT_REAL_BATCH ","RESULT_PROFILE_ONLY "):
            if line.startswith(tag):
                d=json.loads(line[len(tag):])
                m=d["median_us"]
                print(f"{os.path.basename(f):28s} {tag.strip()[7:11]} g={d['group_blocks']:2d} compute={m['compute_us']:8.1f} total={m['total_us']:8.1f} in={m['input_us']:6.1f} out={m['output_us']:6.1f} per_blk={d['median_compute_us_per_block']:.2f} rms={d.get('max_rms','-')} sha={d.get('output_sha256','-')[:12]}")
