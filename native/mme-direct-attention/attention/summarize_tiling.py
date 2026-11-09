# SPDX-License-Identifier: MIT
"""Summarize retained tiling experiments without losing their timing boundaries."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
records={}
for p in sorted((root/'evidence').glob('*.log')):
 if not p.name.startswith(('raw-m256-','raw-m512-','quad512-','tiling-abba-','coalesce','head-h0-g20-','parallel-pv-','tile256-','pv-owner')):continue
 for line in p.read_text(errors='replace').splitlines():
  for marker in ('RESULT_REAL_BATCH ','RESULT_MME_RAW '):
   if line.startswith(marker):
    d=json.loads(line[len(marker):]);keys=('blocks','valid_queries','group_blocks','groups','tile_rows','coalesce_qk','parallel_pv','kind','size','rows','descriptors','pv_owner','median_us','max_rms','exact','sum_rel_error','output_sha256','guard_ok','sram_inputs_unchanged','special_ops_unchanged','reload_head_kv')
    records[p.name]={k:d[k] for k in keys if k in d}
p=root/'evidence/tiling-summary-20261109.json';p.write_text(json.dumps(records,indent=2)+'\n')
for name,d in records.items():
 if name.startswith(('head-h0-g20-','tiling-abba-')): print(name,json.dumps(d))
print('Saved',len(records),'records to',p)
