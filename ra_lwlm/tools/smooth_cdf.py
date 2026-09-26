#!/usr/bin/env python3
"""
smooth_cdf.py
=====================================================================
对 CDF 曲线做高斯核平滑 / smooth an empirical CDF with a Gaussian kernel.

中文：输入是两列的 CDF 文件（误差(m)  累积概率）。用高斯核平滑经验 CDF，并在 0 处
      做反射保证 F(0)=0、单调、F(最大误差)=1；默认按 **纵轴等间隔** 采样，即
      p = 1/n_grid, 2/n_grid, ..., 1，横坐标由平滑 CDF 反解（二分）得到。
      带宽 h 用 Silverman 规则自动选，--bw 是它的倍数（>1 更平滑）。
      纯 Python 实现，不依赖 numpy，可以在登录节点直接跑。

      python smooth_cdf.py results/.../xxx_cdf.txt [--bw 1.0] [--n_grid 1000]

EN: Smooth an empirical CDF file (two columns: err  cumfrac) with a Gaussian kernel.

F_s(x) = mean_i [ Phi((x-e_i)/h) + Phi((x+e_i)/h) - 1 ]      (kernel CDF, reflected at 0 so F(0)=0)
h = bw * 0.9*min(sigma, IQR/1.34) * n^(-1/5)                  (Silverman; --bw scales it)
Output has the same two-column format, sampled at UNIFORM cumulative-probability levels
(p = 1/n_grid, 2/n_grid, ..., 1-1/n_grid; x = F_s^{-1}(p) by inverting the smoothed CDF) -> <input>_smooth.txt.
Use --x_uniform for a uniform x grid instead.
Pure python (no numpy) so it runs on the login node.
    python ra_lwlm/tools/smooth_cdf.py results/deepmimo_bs1_h15_4k/omp/omp_cdf.txt [--bw 1.0] [--n_grid 500] [--xmax 5]
"""
import argparse, math, sys, os
p = argparse.ArgumentParser()
p.add_argument("files", nargs="+"); p.add_argument("--bw", type=float, default=1.0, help="bandwidth multiplier on Silverman's rule")
p.add_argument("--n_grid", type=int, default=500); p.add_argument("--xmax", type=float, default=None, help="grid upper end (default: max error)")
p.add_argument("--out", type=str, default=None, help="output path (single input only)")
p.add_argument("--x_uniform", action="store_true", help="sample on a uniform x grid instead of uniform probability levels")
p.add_argument("--p_max", type=float, default=1.0, help="highest probability level when sampling uniformly in p (p=1 -> x = max observed error)")
a = p.parse_args()
Phi = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
def q(s, f):  # quantile of sorted list
    return s[min(len(s) - 1, int(f * len(s)))]
for fn in a.files:
    rows = [l.split() for l in open(fn) if l.strip()]
    e = sorted(float(r[0]) for r in rows); n = len(e)
    mu = sum(e) / n; sd = math.sqrt(sum((x - mu) ** 2 for x in e) / n); iqr = q(e, .75) - q(e, .25)
    h = a.bw * 0.9 * min(sd, iqr / 1.34) * n ** (-0.2)
    xmax = a.xmax if a.xmax else e[-1]
    Fs = lambda x: sum(Phi((x - v) / h) + Phi((x + v) / h) - 1.0 for v in e) / n
    if a.x_uniform:
        xs = [xmax * i / (a.n_grid - 1) for i in range(a.n_grid)]; F = [Fs(x) for x in xs]
    else:   # uniform in probability: invert F_s by bisection (F_s is monotone)
        F = [a.p_max * (i + 1) / a.n_grid for i in range(a.n_grid)]
        def inv(pv, lo=0.0, hi=e[-1] + 6 * h):
            if pv >= 1.0 - 1e-12: return e[-1]
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if Fs(mid) < pv: lo = mid
                else: hi = mid
            return 0.5 * (lo + hi)
        xs = [inv(pv) for pv in F]
    out = a.out if a.out else fn.replace(".txt", "_smooth.txt")
    with open(out, "w") as f:
        for x, y in zip(xs, F): f.write(f"{x:.6f}  {y:.6f}\n")
    p90_raw = q(e, .9); p90_s = next(x for x, y in zip(xs, F) if y >= 0.9 - 1e-9)
    print(f"{os.path.relpath(fn):60s} n={n} h={h:.4f}m  p90 raw={p90_raw:.3f} smooth={p90_s:.3f} -> {os.path.relpath(out)}")
