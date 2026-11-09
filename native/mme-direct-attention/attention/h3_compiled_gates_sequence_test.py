import importlib.util,sys,runpy
from pathlib import Path
r=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("h3_direct_attention",r/"h3_compiled_gates_attention.py")
hd=importlib.util.module_from_spec(spec);sys.modules["h3_direct_attention"]=hd;spec.loader.exec_module(hd)
runpy.run_path(str(r/"h3_sequence_test.py"),run_name="__main__")
