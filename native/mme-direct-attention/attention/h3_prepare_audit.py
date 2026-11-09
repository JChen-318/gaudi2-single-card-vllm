# SPDX-License-Identifier: MIT
"""Prepare-only integration audit: no raw MME/TPC ECB dispatch."""
import argparse,ctypes as C,json,torch
import habana_frameworks.torch.core
import vllm_gaudi.omni.attention
import h3_direct_attention as hd
p=argparse.ArgumentParser();p.add_argument('dump');p.add_argument('--repeats',type=int,default=1);a=p.parse_args()
d=torch.load(a.dump,map_location='cpu',weights_only=True,mmap=True)
q,k,v=(d[n].permute(0,2,1,3).contiguous().to('hpu') for n in ('q','k','v'))
torch.hpu.synchronize();plan=hd._plan(q.shape[1])
audit=plan.lib.context_quad_audit_prepared;audit.argtypes=[];audit.restype=C.c_int
assert audit()==0
qs=q[0]*float(d['scale']);gates=hd.head_gates(qs,k[0],v[0])
kp=torch.nn.functional.pad(k[0],(0,0,0,hd.KEYS-hd.VALID))
kt=torch.cat([kp[:,i*hd.FIRST:min((i+1)*hd.FIRST,hd.KEYS)].transpose(1,2).reshape(q.shape[1],-1) for i in range(4)],dim=1)
plan.qh[0,:hd.VALID].copy_(qs[0]);plan.kth[0].copy_(kt[0]);plan.vh[0,:hd.VALID].copy_(v[0,0]);torch.hpu.synchronize()
rc=audit();assert rc==0,rc
print('H3_LAYER_PREPARE_AUDIT '+json.dumps(dict(pass_audit=True,heads=len(gates),passing=sum(gates),addresses={n:hex(plan._data_ptr(getattr(plan,n))) for n in ('qh','kth','vh','out','stats')},raw_ecbs_dispatched=0)),flush=True)
