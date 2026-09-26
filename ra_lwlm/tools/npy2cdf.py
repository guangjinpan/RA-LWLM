"""
npy2cdf.py
=====================================================================
把逐样本误差的 .npy 转成两列 CDF 文件 / turn a per-sample error .npy into a
two-column CDF file (error(m)  cumulative probability), 1000 points.

    python npy2cdf.py errors.npy out/cdf.txt
"""
import numpy as np, sys, os
src, dst = sys.argv[1], sys.argv[2]
e = np.sort(np.load(src)); N = len(e); idx = np.linspace(0, N-1, min(1000, N)).round().astype(int)
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "w") as f:
    for i in idx: f.write(f"{e[i]:.6f}  {(i+1)/N:.6f}\n")
print(f"{os.path.basename(src)}: n={N} mean={e.mean():.3f} median={np.median(e):.3f} p90={np.percentile(e,90):.3f} -> {dst}")
