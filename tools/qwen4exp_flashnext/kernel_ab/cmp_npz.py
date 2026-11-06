import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
for k in a.files:
    d = int((a[k] != b[k]).sum()); print(f"{k:18s} {'identical' if d == 0 else f'{d} of {a[k].size} differ'}")
