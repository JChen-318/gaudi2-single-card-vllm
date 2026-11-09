import sys,json,pathlib,os
import torch
import habana_frameworks.torch.core
m,n,k=map(int,sys.argv[1:])
torch.manual_seed(17)
dtype=getattr(torch,os.environ.get('CAPTURE_DTYPE','bfloat16'))
a=torch.randint(-4,5,(m,k)).to(dtype)
b=torch.randint(-4,5,(k,n)).to(dtype)
out_dtype=getattr(torch,os.environ.get('CAPTURE_OUTPUT_DTYPE',os.environ.get('CAPTURE_DTYPE','bfloat16')))
f=torch.compile(lambda a,b:(a@b).to(out_dtype),backend='hpu_backend',fullgraph=True)
c=f(a.to('hpu'),b.to('hpu')).cpu()
ref=(a.float()@b.float()).to(out_dtype)
print('CAPTURE',m,n,k,'equal',torch.equal(c,ref),flush=True)
if not torch.equal(c,ref): raise RuntimeError('Mismatch')
