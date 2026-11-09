# SPDX-License-Identifier: MIT
"""CPU-only compare failed tile256 diagnostics with FP16-score/logq stages."""
import argparse,struct,json,torch
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('dump',type=Path);a=p.parse_args()
torch.set_num_threads(8)
b=bytearray(a.input.read_bytes());_,keys,valid,dim,n,rows,_=struct.unpack_from('<8s6I',b);off=32;ts=[]
for shape in ((rows,128),(keys,128),(keys,128)):
 count=shape[0]*shape[1];ts.append(torch.frombuffer(b,dtype=torch.bfloat16,count=count,offset=off).clone().reshape(shape));off+=count*2
q,k,v=ts;d=torch.load(a.dump,map_location='cpu',weights_only=True);st=d['stats'].reshape(-1,512,2)
first=(keys//128+1)//2*128
for j in (0,1,n//2-1):
 for h,(lo,hi) in enumerate(((0,first),(first,keys))):
  s=(q[j*256:(j+1)*256].float()@k[lo:hi].float().T).half().float();nv=min(hi,valid)-lo
  picks=(torch.arange(nv)//128)%8==0;m=s[:,:nv][:,picks].max(-1,keepdim=True).values
  t=torch.round(s*184.6649652337873+(16256-m*184.6649652337873)).clamp(128,32639).int();idx=t&127
  mant=torch.round((2**(idx.float()/128)-1)*128).int();pp=(t-idx+mant).to(torch.int16).view(torch.bfloat16);pp[:,nv:]=0
  actual=st[j,h*256:(h+1)*256];sm=pp.float().sum(-1)
  record={'pair':j,'half':h,'max_abs_diff':float((actual[:,0]-m[:,0]).abs().max()),'sum_relative_max':float(((actual[:,1]-sm)/sm).abs().max()),'sum_ratio_first8':(actual[:8,1]/sm[:8]).tolist()}
  if j==n//2-1:
   got=d['prob'+str(h)].float();ref=pp.float();record.update(prob_mismatch=float((got!=ref).float().mean()),prob_relative=float((got-ref).norm()/ref.norm()),prob_max_abs=float((got-ref).abs().max()),sum_from_actual_p_relative=float(((got.sum(-1)-actual[:,1])/actual[:,1]).abs().max()))
  print(json.dumps(record),flush=True)
