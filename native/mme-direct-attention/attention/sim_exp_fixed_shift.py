# SPDX-License-Identifier: MIT
"""CPU model for bounded fixed-shift exp, replacing sampled per-row maxima.
Shift 4 is softmax-invariant. Input gates prevent FP32/PV overflow and BF16 FTZ.
No tolerance relaxation; compare every selected 128-query tile with FP32 ref.
"""
import argparse,json,struct,math
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument("--shift",type=int,choices=(0,4),default=0);p.add_argument('--limit',type=int,default=16);p.add_argument('--output',type=Path);a=p.parse_args()
torch.set_num_threads(8)
b=bytearray(a.input.read_bytes());magic,keys,valid,dim,n,rows,validq=struct.unpack_from('<8s6I',b);off=32;ts=[]
assert magic==b'G2BATCH1' and dim==128
for shape in ((rows,128),(keys,128),(keys,128)):
 count=shape[0]*shape[1];ts.append(torch.frombuffer(b,dtype=torch.bfloat16,count=count,offset=off).clone().reshape(shape));off+=count*2
q,k,v=ts;ref=torch.frombuffer(b,dtype=torch.float32,count=n*16384,offset=off).clone().reshape(n,128,128)
u=float((q.float().norm(dim=-1)*k[:valid].float().norm(dim=-1).max()).max());vmax=float(v[:valid].float().abs().max());logbound=u+0.25-a.shift+math.log(valid)+math.log(max(vmax,1.0))
assert u<=80 and logbound<86 and -u-a.shift-.25>-86,(u,vmax,logbound)
first=(keys//128+3)//4*128
j=torch.arange(128);correction=(torch.round((2**(j.double()/128)-1)*128).int()-j+16256)
result={'input':str(a.input),'bound':u,'vmax':vmax,'pv_log_upper':logbound,'shift':a.shift,'blocks':[]}
for i in range(n if a.limit==0 else min(n,a.limit)):
 num=torch.zeros(128,128);den=torch.zeros(128,1)
 for h in range(4):
  lo=h*first;hi=min((h+1)*first,keys);nv=min(hi,valid)-lo
  s=(q[i*128:(i+1)*128].float()@k[lo:hi].float().T).half()
  aff=((s-a.shift).half()*184.625).half();quant=aff.float().round().int()
  assert quant.min()>=-16128 and quant.max()<16384
  pp=(quant+correction[quant&127]).to(torch.int16).view(torch.bfloat16);pp[:,nv:]=0
  assert torch.isfinite(pp).all() and (pp[:,:nv]>0).all()
  num+=(pp.float()@v[lo:hi].float()).bfloat16().float();den+=pp.float().sum(-1,keepdim=True)
 got=(num/den).bfloat16().float();err=float(((got-ref[i]).square().mean()/ref[i].square().mean().clamp_min(1e-30)).sqrt())
 result['blocks'].append({'block':i,'rms':err});print(json.dumps(result['blocks'][-1]),flush=True)
 assert err<.01, result['blocks'][-1]
result['max_rms']=max(d['rms'] for d in result['blocks']);print('RESULT_FIXED_MODEL '+json.dumps(result),flush=True)
if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
