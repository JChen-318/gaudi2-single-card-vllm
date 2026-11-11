# SPDX-License-Identifier: MIT
"""Opt-in H3 attention injection; chains the existing benchmark profiler.
No production source files are modified and no accelerator is opened here.
"""
import importlib.abc
import importlib.machinery
import os
import runpy
import sys

profiler=os.environ.get('H3_BASE_SITECUSTOMIZE')
if profiler:
    runpy.run_path(profiler,run_name='_h3_existing_sitecustomize')

_TARGET='vllm_gaudi.omni.attention'
_OVERLAY=os.environ.get('H3_ATTENTION_OVERLAY')
def _overlay(module):
    # Test a fixed copy of vllm_gaudi/omni/attention.py without touching the deployed tree: its definitions
    # replace the module's (functions are looked up as module globals at call time).
    source=open(_OVERLAY,encoding='utf-8').read()
    exec(compile(source,_OVERLAY,'exec'),module.__dict__)
    print('[h3direct] attention overlay applied: '+_OVERLAY+' exact_attention='+str(hasattr(module,'exact_attention'))
          +' maxfree_guard='+str(getattr(module,'HPU_SDPA_MAX_FREE_GUARD',None)),flush=True)
def _install(module):
    if _OVERLAY:_overlay(module)
    if os.environ.get('H3DIRECT_ENABLE')!='1':return
    import h3_direct_attention as direct
    direct.install(module)
    print('[h3direct] max_free_attention patched; original layer skips preserved; '
          'mode='+('wide-relaxed' if os.environ.get('H3DIRECT_RELAXED')=='1' else 'wide')+'; pow2_q='+str(direct.POW2_Q),flush=True)

class _Loader(importlib.abc.Loader):
    def __init__(self,inner):self.inner=inner
    def create_module(self,spec):return self.inner.create_module(spec)
    def exec_module(self,module):
        self.inner.exec_module(module)
        _install(module)

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path,target=None):
        if name!=_TARGET:return None
        spec=importlib.machinery.PathFinder.find_spec(name,path)
        if spec is not None and spec.loader is not None:spec.loader=_Loader(spec.loader)
        return spec

if os.environ.get('H3DIRECT_ENABLE')=='1' or _OVERLAY:
    if _TARGET in sys.modules:_install(sys.modules[_TARGET])
    else:sys.meta_path.insert(0,_Finder())
