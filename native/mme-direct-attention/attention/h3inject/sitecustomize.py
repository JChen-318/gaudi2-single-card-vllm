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
def _install(module):
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

if os.environ.get('H3DIRECT_ENABLE')=='1':
    if _TARGET in sys.modules:_install(sys.modules[_TARGET])
    else:sys.meta_path.insert(0,_Finder())
