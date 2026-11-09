# SPDX-License-Identifier: MIT
"""CPU-only checks for pinned MME QK activation coalescing."""
from pathlib import Path
import copy
import mme_dump
root=Path(__file__).resolve().parent
for shape, count in (("g-m128n37760k128",2),("g-m256n18944k128",2),("g-m256n18816k128",4),("g-m256n37760k128",4),("g-m512n9472k128",2),("g-m512n9344k128",4)):
    ds=mme_dump.load_set(root/"fixtures"/shape/"mme_debug")
    got=mme_dump.coalesce_qk(ds)
    assert len(got)==count
    for e in (0,1):
        def coverage(seq):
            result=[]
            for _,engine,w in seq:
                if engine!=e:continue
                start=w[(0x22c-8)//4];step=w[(0x104-8)//4];n=w[(0x280-8)//4]+1
                roi=w[(0x118-8)//4]
                result.extend((start+i*step,roi) for i in range(n))
            return result
        assert coverage(ds)==coverage(got),(shape,e)
    bad=copy.deepcopy(ds);bad[2][2][(0x22c-8)//4]+=1
    try: mme_dump.coalesce_qk(bad)
    except ValueError: pass
    else: raise AssertionError("Malformed AGU accepted")
    assert len(mme_dump.coalesce_qk(got))==count
    print(shape,"PASS",len(ds),"->",count)
