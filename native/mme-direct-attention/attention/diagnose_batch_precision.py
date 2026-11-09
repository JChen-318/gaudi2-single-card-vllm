# SPDX-License-Identifier: MIT
# CPU-only diagnostic: expected BF16 QK/PV error per distinct query tile.
import torch,struct,json,argparse
from pathlib import Path
torch.set_num_threads(8)
ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--limit',type=int,default=16);args=ap.parse_args();p=args.input
b=bytearray(p.read_bytes());_,keys,valid,dim,n,rows,_=struct.unpack_from('<8s6I',b);offset=32;ts=[]
for shape in [(n*128,128),(keys,128),(keys,128)]:
 count=shape[0]*shape[1];ts.append(torch.frombuffer(b,dtype=torch.bfloat16,count=count,offset=offset).clone().reshape(shape));offset+=2*count
q,k,v=ts;ref=torch.frombuffer(b,dtype=torch.float32,count=n*16384,offset=offset).clone().reshape(n,128,128)
first=(keys//128+1)//2*128
rows=[]
for i in range(min(n,args.limit)):
 raw=q[i*128:(i+1)*128].float()@k.float().T
 s=raw.bfloat16().float();s16=raw.half().float()
 record={'block':i}
 for mode in ['fp32_prescaled','bf16_exact','bf16_sampled','logq_sampled','fp16_exact','fp16_logq_sampled']:
  num=torch.zeros(128,128);den=torch.zeros(128,1);true_max=s[:,:valid].max(-1,keepdim=True).values;gap=0
  if mode=='fp32_prescaled':
   out=torch.softmax(q[i*128:(i+1)*128].float()@k[:valid].float().T,-1)@v[:valid].float()
  else:
   for lo,hi in [(0,first),(first,valid)]:
    sh=(s16 if mode.startswith('fp16_') else s)[:,lo:hi];m=sh.max(-1,keepdim=True).values
    if 'sampled' in mode:
     picks=(torch.arange(sh.shape[1])//128)%8==0;m=sh[:,picks].max(-1,keepdim=True).values
    gap=max(gap,float((sh.max(-1,keepdim=True).values-m).max()))
    if 'logq' in mode:
     t=torch.round(sh*184.6649652337873 + (16256-m*184.6649652337873)).clamp(128,32639).int();j=t&127
     mant=torch.round((2**(j.float()/128)-1)*128).int()
     prob=(t-j+mant).to(torch.int16).view(torch.bfloat16)
    else: prob=torch.exp((sh-m) if mode.startswith('fp16_') else (sh-m).bfloat16().float()).bfloat16()
    weight=torch.exp(m-true_max)
    num+=(prob.float()@v[lo:hi].float()).bfloat16().float()*weight
    den+=prob.float().sum(-1,keepdim=True)*weight
   out=(num/den).bfloat16().float()
  record[mode]=float(((out-ref[i]).square().mean()/ref[i].square().mean()).sqrt())
 record['sample_gap']=gap;rows.append(record)
 print(json.dumps(record),flush=True)
if args.output: args.output.write_text(json.dumps(rows,indent=2)+'\n')
