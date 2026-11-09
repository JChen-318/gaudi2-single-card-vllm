# SPDX-License-Identifier: MIT
import argparse,json,time,statistics
import torch
import habana_frameworks.torch.core
import h3_compiled_gates_attention as hd
p=argparse.ArgumentParser();p.add_argument('dump');p.add_argument('--repeats',type=int,default=1);a=p.parse_args()
torch.set_num_threads(16)
d=torch.load(a.dump,map_location='cpu',weights_only=True,mmap=True)
q,k,v=(d[n].permute(0,2,1,3).contiguous().to('hpu')[0] for n in ('q','k','v'));qs=q*.0625;factor=hd.SCORE_FACTOR
records=[]
for inference in (False,True):
 with torch.inference_mode(inference):
  values=[]
  for compiled in (False,True):
   fn=hd._compiled_bounds if compiled else hd._device_bounds;times=[]
   for i in range(11):
    torch.hpu.synchronize();t=time.perf_counter();u,vm=fn(qs,k,v,factor);uc,vc=u.cpu(),vm.cpu();times.append((time.perf_counter()-t)*1000)
   values.append((uc,vc));records.append(dict(inference=inference,compiled=compiled,steady_ms=statistics.median(times[1:]),times_ms=times))
  assert torch.equal(values[0][1],values[1][1])
  assert torch.all(values[1][0]*1.00002>=values[0][0]), 'Compiled bound is not conservative'
# Check that compiler fusion retains nonfinite rejection used by fallback.
for name in ('value_nan','value_inf','query_nan','key_inf'):
 qq=qs.clone();kk=k.clone();vv=v.clone()
 if name=='value_nan':vv[0,100,0]=float('nan')
 if name=='value_inf':vv[0,100,0]=float('inf')
 if name=='query_nan':qq[0,100,0]=float('nan')
 if name=='key_inf':kk[0,100,0]=float('inf')
 gates=hd.head_gates(qq,kk,vv,factor);assert not gates[0],name
 assert all(gates[1:]),name
print('H3_LAYER_BOUNDS_COMPILE '+json.dumps(dict(records=records,conservative_bounds_checked=True,nonfinite_rejection_checked=True)),flush=True)
