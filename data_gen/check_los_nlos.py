"""
check_los_nlos.py
=====================================================================
LOS / NLOS 标注与统计 / LOS-NLOS labelling and statistics.

EN: For every scene, decide whether each UE is LOS or NLOS w.r.t. the BS using a
    slab intersection test against the axis-aligned building boxes, and report the
    LOS / NLOS counts per scene. The same predicate is reimplemented inside
    ra_lwlm/eval_ra_lwlm.py so the evaluation tables split exactly the same way.

中文说明如下：
遍历 RAG_dataset 下每个场景，根据场景的建筑几何信息判断每个 UE 与 BS 之间
是 LOS 还是 NLOS，统计每个场景下 LOS / NLOS 数据量。

判据：
  * UE 高度固定 1.5m，BS 高度来自 config.json
  * 建筑视为轴对齐长方体（中心 cx/cy，长宽 length/width，高度 height=10m）
  * BS→UE 直线，用 slab 法求进入/离开建筑 xy 脚印的参数 (t_enter, t_exit)
  * 线段高度 z(t) 在 [t_enter, t_exit] 单调递减，最小值在 t_exit 处
  * 若 z(t_exit) < 建筑高度 → 直线在建筑内部 → NLOS

UE 位置取自 H5 文件 UElocation 字段（存储时 /10，这里乘回 10）。
"""

import os
import sys
import json
import argparse
from glob import glob
from multiprocessing import Pool

import numpy as np
import h5py


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATASET_DIR as DEFAULT_ROOT   # 默认路径见 paths.py / default from paths.py
UE_HEIGHT = 1.5


def line_blocked_by_building(bs_xyz, ue_xyz, b):
    """判断 BS->UE 线段是否被单个建筑 b 阻挡。b: dict with cx,cy,length,width,height"""
    bx_min = b["cx"] - b["length"] / 2.0
    bx_max = b["cx"] + b["length"] / 2.0
    by_min = b["cy"] - b["width"] / 2.0
    by_max = b["cy"] + b["width"] / 2.0
    bh = b["height"]

    p0 = np.asarray(bs_xyz, dtype=np.float64)
    p1 = np.asarray(ue_xyz, dtype=np.float64)
    d  = p1 - p0

    t_enter, t_exit = 0.0, 1.0
    for axis, lo, hi in ((0, bx_min, bx_max), (1, by_min, by_max)):
        if abs(d[axis]) < 1e-9:
            if p0[axis] < lo or p0[axis] > hi:
                return False
            continue
        t1 = (lo - p0[axis]) / d[axis]
        t2 = (hi - p0[axis]) / d[axis]
        if t1 > t2:
            t1, t2 = t2, t1
        t_enter = max(t_enter, t1)
        t_exit  = min(t_exit,  t2)
        if t_enter >= t_exit:
            return False

    z_exit = p0[2] + t_exit * d[2]
    return z_exit < bh


def is_nlos(ue_xy, bs_xyz, buildings):
    ue_xyz = (float(ue_xy[0]), float(ue_xy[1]), UE_HEIGHT)
    for b in buildings:
        if line_blocked_by_building(bs_xyz, ue_xyz, b):
            return True
    return False


def process_scene(args):
    """统计一个场景的 LOS / NLOS 数量（多进程调用）。
    Count the LOS / NLOS samples of one scene (called from a process pool)."""
    scene_dir, scene_id = args
    cfg_path = os.path.join(scene_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return scene_id, 0, 0, 0, "missing config.json"

    with open(cfg_path) as f:
        cfg = json.load(f)
    bs_pos    = cfg["bs_position"]
    buildings = cfg["buildings"]

    h5_files = sorted(glob(os.path.join(scene_dir, "*.h5py")))
    n_los = n_nlos = 0
    for fp in h5_files:
        try:
            with h5py.File(fp, "r") as f:
                ue_loc = f["UElocation"][:]     # stored as /10
        except Exception:
            continue
        ue_xy = np.asarray(ue_loc) * 10.0
        if is_nlos(ue_xy, bs_pos, buildings):
            n_nlos += 1
        else:
            n_los += 1
    return scene_id, n_los, n_nlos, len(h5_files), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT, help="RAG_dataset 根目录")
    parser.add_argument("--scene_start", type=int, default=0)
    parser.add_argument("--scene_end",   type=int, default=110)
    parser.add_argument("--workers",     type=int, default=8)
    parser.add_argument("--out_csv",     default=None, help="可选：保存统计结果 CSV")
    args = parser.parse_args()

    jobs = []
    for sid in range(args.scene_start, args.scene_end):
        scene_dir = os.path.join(args.root, f"scene_{sid:03d}")
        if os.path.isdir(scene_dir):
            jobs.append((scene_dir, sid))

    print(f"扫描 {len(jobs)} 个场景，workers={args.workers}")
    print(f"{'scene':>6} | {'LOS':>7} | {'NLOS':>7} | {'total':>7} | LOS%")
    print("-" * 50)

    results = []
    tot_los = tot_nlos = tot_all = 0
    with Pool(args.workers) as pool:
        for sid, n_los, n_nlos, n_total, err in pool.imap_unordered(process_scene, jobs):
            if err:
                print(f"{sid:>6} | ERROR: {err}")
                continue
            results.append((sid, n_los, n_nlos, n_total))
            tot_los  += n_los
            tot_nlos += n_nlos
            tot_all  += n_total

    results.sort()
    for sid, n_los, n_nlos, n_total in results:
        pct = 100.0 * n_los / n_total if n_total else 0.0
        print(f"{sid:>6} | {n_los:>7} | {n_nlos:>7} | {n_total:>7} | {pct:5.1f}%")

    print("-" * 50)
    pct_tot = 100.0 * tot_los / tot_all if tot_all else 0.0
    print(f"{'TOTAL':>6} | {tot_los:>7} | {tot_nlos:>7} | {tot_all:>7} | {pct_tot:5.1f}%")

    if args.out_csv:
        with open(args.out_csv, "w") as f:
            f.write("scene_id,los,nlos,total\n")
            for sid, n_los, n_nlos, n_total in results:
                f.write(f"{sid},{n_los},{n_nlos},{n_total}\n")
        print(f"saved → {args.out_csv}")


if __name__ == "__main__":
    main()
