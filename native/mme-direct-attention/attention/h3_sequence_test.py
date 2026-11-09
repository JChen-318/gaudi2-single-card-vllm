# SPDX-License-Identifier: MIT
"""Persistent-plan stress: changing layers, FP8 GEMMs, gate fallback and output ownership."""
import argparse,hashlib,json,time,os
from pathlib import Path
import torch
import habana_frameworks.torch.core
from vllm_gaudi.extension.utils import FP8Matmul
from vllm_gaudi.omni.attention import max_free_attention
import h3_direct_attention as hd
p=argparse.ArgumentParser();p.add_argument('dump',type=Path);p.add_argument('--repeats',type=int,default=1);a=p.parse_args()
torch.set_num_threads(16)
root=Path(__file__).resolve().parent
expected={}
names=[('h3-pow2-layer25',25),('h3-pow2-layer0',0),('h3-pow2-layer38',38)] if hd.POW2_Q else [('h3-banks-full25-safe',25),('h3-pack-direct-layer0',0),('h3-pack-direct-layer38',38)]
for name,layer in names:
 for l in (root/'evidence'/f'{name}.log').read_text().splitlines():
  if l.startswith('H3_LAYER '):expected[layer]=json.loads(l[9:])['direct_output_sha256'][0]
def digest(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
x=torch.full((128,128),.5,dtype=torch.bfloat16,device='hpu')
y=torch.full_like(x,.25);fp8=FP8Matmul()
def check_fp8():
 z=fp8(x,y).cpu()
 assert bool(torch.all(z==16)), 'FP8 GEMM state changed'
def framework_signature(q,k,v,scale):
    # Exercise the real fallback graph and the fused attention used by skipped
    # model layers, in addition to FP8 projections. Compare before/after raw.
    from vllm_gaudi.omni.attention import HPUSDPAImpl
    qq=q[:,:1].contiguous();kk=k[:,:1].contiguous();vv=v[:,:1].contiguous()
    fallback=max_free_attention(qq,kk,vv,scale)
    # Use the actual production 56-head shape and aligned-KV split, rather
    # than a synthetic 1-head/128Q recompute shape rejected by HEAD_FACTOR=2.
    impl=HPUSDPAImpl(num_heads=q.shape[1],head_size=128,softmax_scale=scale,causal=False)
    fused=impl.forward(*(t.permute(0,2,1,3) for t in (q,k,v)))
    special=torch.nn.functional.gelu(x)+torch.exp(y)+torch.rsqrt(x+1)
    torch.hpu.synchronize()
    return dict(maxfree=digest(fallback),fused=digest(fused),special=digest(special))
records=[]
for layer in (25,0,38,25):
 check_fp8()
 d=torch.load(a.dump.parent/f'qkv-call1-layer{layer}.pt',map_location='cpu',weights_only=True,mmap=True)
 q,k,v=(d[n].permute(0,2,1,3).contiguous().to('hpu') for n in ('q','k','v'));scale=float(d['scale'])
 before_framework=framework_signature(q,k,v,scale)
 torch.hpu.synchronize();t=time.perf_counter()
 out=hd.direct_max_free_attention(q,k,v,scale,max_free_attention)
 torch.hpu.synchronize();ms=(time.perf_counter()-t)*1000
 actual=digest(out);assert actual==expected[layer], (layer,actual,expected[layer])
 check_fp8()
 after_framework=framework_signature(q,k,v,scale)
 assert before_framework==after_framework, 'Framework fallback/fused/special state changed'
 records.append(dict(layer=layer,ms=ms,sha256=actual,framework_before_after=before_framework))
 print('H3_LAYER_SEQUENCE_STEP '+json.dumps(records[-1]),flush=True)
# A new result may not overwrite a previous returned tensor.
saved=out;bad=v.clone();bad[:,0]=float('nan');fallback=[]
def route_only(q,k,v,scale):
 assert q.shape[1]==1 and bool(torch.isnan(v).any())
 fallback.append(q.shape[1]);return torch.zeros_like(q)
route=hd.direct_max_free_attention(q,k,bad,scale,route_only)
assert fallback==[1] and bool(torch.all(route[:,0]==0))
assert digest(route[:,1:])==digest(saved[:,1:])
assert digest(saved)==expected[25], 'Previous output was overwritten'
check_fp8()
again=hd.direct_max_free_attention(q,k,v,scale,max_free_attention)
assert digest(again)==expected[25], 'Staging did not recover after fallback'
check_fp8()
print('H3_LAYER_SEQUENCE '+json.dumps(dict(records=records,fp8_before_after=True,production_fallback_fused_special_interop=True,fallback_route_checked=True,returned_output_owned=True,recovery_checked=True,stats=hd.STATS)),flush=True)
