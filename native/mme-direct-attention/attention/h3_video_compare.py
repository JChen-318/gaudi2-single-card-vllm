# SPDX-License-Identifier: MIT
"""Compare matched completed H3 runs, excluding request0 warmup."""
import argparse,hashlib,json,re,statistics,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('direct');a=p.parse_args()
root=Path(__file__).resolve().parent
runs=[]
for tag in (a.baseline,a.direct):
 d=root/'evidence'/f'video-{tag}'
 assert (d/'status').read_text().strip()=='finished rc=0',f'Incomplete run: {tag}'
 records=[json.loads(x) for x in (d/'records.jsonl').read_text().splitlines() if x.strip()]
 meta=[json.loads(x.read_text()) for x in sorted(d.glob('out[0-9]*.json'))]
 assert len(records)==len(meta)>=3,(tag,len(records),len(meta))
 assert all(x['status']=='pass' for x in meta)
 steady=records[1:]
 runs.append(dict(tag=tag,metadata=meta,records=records,steady_pipeline_s=statistics.median(x['total_s'] for x in steady),
                  steady_request_wall_s=statistics.median(x['wall_seconds'] for x in meta[1:]),
                  steady_dit_s=statistics.median(sum(x['stages_s']['dit_forward']) for x in steady),
                  steady_video_decode_s=statistics.median(sum(x['stages_s']['video_decode_latent']) for x in steady),
                  steady_peak_memory_mib=max(float(x['response_headers']['x-peak-memory-mb']) for x in meta[1:])))
for x,y in zip(runs[0]['metadata'],runs[1]['metadata']):
 assert x['request']==y['request'],'Requests are not matched'
 assert x['sampling_contract']==y['sampling_contract'],'Sampling contract changed'
quality=[]
for i in range(len(runs[0]['metadata'])):
 paths=[root/'evidence'/f'video-{tag}'/f'out{i}.mp4' for tag in (a.baseline,a.direct)]
 cmd=['/home/a/h3-deploy/media-bin/ffmpeg','-hide_banner','-nostats','-i',str(paths[1]),'-i',str(paths[0]),'-lavfi','psnr','-f','null','-']
 r=subprocess.run(cmd,capture_output=True,text=True,check=True)
 lines=[x for x in r.stderr.splitlines() if 'PSNR' in x]
 assert lines,'No PSNR result'
 quality.append(dict(request=i,psnr=lines[-1],sha256=[hashlib.sha256(x.read_bytes()).hexdigest() for x in paths]))
summary=dict(runs=runs,matched_requests=True,warmup_excluded=True,
             pipeline_time_reduction=1-runs[1]['steady_pipeline_s']/runs[0]['steady_pipeline_s'],
             request_time_reduction=1-runs[1]['steady_request_wall_s']/runs[0]['steady_request_wall_s'],
             dit_time_reduction=1-runs[1]['steady_dit_s']/runs[0]['steady_dit_s'],quality=quality)
out=root/'evidence'/f'video-comparison-{a.baseline}-vs-{a.direct}.json'
out.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='runs'},indent=2))
for r in runs:print(json.dumps({k:v for k,v in r.items() if k not in ('metadata','records')}))
