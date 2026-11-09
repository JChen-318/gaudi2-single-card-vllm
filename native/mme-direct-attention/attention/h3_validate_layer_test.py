from pathlib import Path
import os,runpy,sys,importlib.util,json
r=Path(__file__).resolve().parent
folder=r/'evidence/h3-reference-layer-test'
folder.mkdir(exist_ok=False)
(folder/'validate-control').write_text('32\n')
os.environ['H3PROF_DIR']=str(folder)
spec=importlib.util.spec_from_file_location('h3_direct_attention',r/'h3_validate_attention.py')
hd=importlib.util.module_from_spec(spec);sys.modules['h3_direct_attention']=hd;spec.loader.exec_module(hd)
ns=runpy.run_path(str(r/'h3_direct_layer_test.py'),run_name='__main__')
assert set(ns['hashes_d'])=={'0b66cf40ed2bbcadd4c19ab94754fdadbd3da34b3ebab445e6cf291061d48138'}
records=[json.loads(x) for x in (folder/'attention-reference.jsonl').read_text().splitlines()]
assert len(records)==len(ns['hashes_d']) and all(x['max_rms']<.01 for x in records)
print('H3_LAYER_REFERENCE_HOOK_VALIDATED '+json.dumps(dict(sha_matches_validated_path=True,records=len(records),max_rms=max(x['max_rms'] for x in records))),flush=True)
