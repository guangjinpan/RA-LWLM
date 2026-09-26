"""
gen_deepmimo_scenes.py
=====================================================================
跨数据集验证用：把 DeepMIMO O1_3p5 转成本仓库的场景格式。
Cross-dataset validation: convert DeepMIMO O1_3p5 into this repo's scene format.

中文：用来检验在 Sionna 上训练的 RA-LWLM 能否直接迁移到另一个光线追踪数据集
      （零样本）。坐标系、阵列朝向都对齐到 Sionna 的约定：基站放在 32×32 m
      窗口的角上，阵列绕 z 轴旋转 --az 度，UE 只取阵列正面的集合。
EN:   Used to test whether a Sionna-trained RA-LWLM transfers to a different
      ray-tracing dataset with no adaptation. The coordinate frame and array
      orientation follow the Sionna convention: the BS sits at the corner of a
      32×32 m window, the array is rotated --az degrees about z, and only the UEs
      in front of the array are kept.


gen_deepmimo_scenes.py
=====================================================================
Build RA-LWLM-compatible scenes from DeepMIMO O1_3p5 (3.5 GHz, ray tracing,
urban street canyon) -- a channel family disjoint from our Sionna generator.

One scene per BS. For each BS we (A) obtain locations + LoS flags for the
whole user grid cheaply (1 path, 1 antenna), (B) slide a 32x32 m window over
the grid and pick the one closest to the BS that holds >= n_ue users with a
LoS fraction in [los_lo, los_hi] (Sionna SS scenes average 55 % LoS; a window
centred on the BS is 100 % LoS), (C) regenerate full channels for the users in
that window (32-element ULA, 128 subcarriers, 10 MHz, 20 paths) and write
them in the RAG_dataset layout:

  <out>/scene_{300+bs:03d}/{ue:05d}.h5py   channel_real/imag (F,T), UElocation(/10),
                                          BSlocation, distance(/100), DoD_phi, bandwidth
  <out>/scene_XXX/config.json             same keys as Sionna scenes (+ source, los_from_file)
  <out>/scene_XXX/los.npy                 DeepMIMO LoS flag per UE (1=LoS, 0=NLoS)
Coordinates are shifted so the window's lower-left corner is (0,0), like Sionna.
"""
import os, sys, json, argparse, time
import numpy as np, h5py
sys.path.insert(0, "/home/guangjin/project_GP/from_alvis/DeepMIMO-python/src")
import DeepMIMOv3 as DM

import sys as _sys
_sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_ROOT          # 默认路径见 paths.py / defaults live in paths.py

ap = argparse.ArgumentParser()
ap.add_argument("--bs", type=str, default="1,2,3,4,5,6,9,10,13,14,15,16")
ap.add_argument("--n_ue", type=int, default=10000); ap.add_argument("--map", type=float, default=32.0)
ap.add_argument("--n_ant", type=int, default=32); ap.add_argument("--n_sc", type=int, default=128)
ap.add_argument("--bw_mhz", type=float, default=10.0); ap.add_argument("--paths", type=int, default=20)
ap.add_argument("--los_lo", type=float, default=0.10); ap.add_argument("--los_hi", type=float, default=0.90)
ap.add_argument("--max_dist", type=float, default=300.0, help="max window-centre distance to BS (m)")
ap.add_argument("--mode", type=str, default="near", choices=["near","mixed","corner"],
    help="near: closest window to the BS with >= n_ue users (LoS-dominant street canyon), scene id 300+bs;  "
         "mixed: closest window with LoS fraction in [los_lo, los_hi] (NLoS-rich, farther), scene id 350+bs;  "
         "corner: Sionna-like geometry -- window = [bs_x+off, bs_x+off+map] x [bs_y+off, bs_y+off+map] so the BS sits\n"
         "         just outside the lower-left corner and all UEs are in front of the array; array rotated by --az deg")
ap.add_argument("--rows", type=int, default=3852); ap.add_argument("--chunk", type=int, default=100)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--az", type=float, default=45.0, help="BS array rotation about z (deg) = boresight azimuth written to config")
ap.add_argument("--corner_off", type=float, default=1.0, help="corner mode: gap (m) between BS and window corner")
# 输出目录默认放在 DATA_ROOT 下；DeepMIMO 原始数据目录用 RALWLM_DEEPMIMO_DIR 指定
# The output goes under DATA_ROOT by default; point RALWLM_DEEPMIMO_DIR at the raw scenario
ap.add_argument("--out", type=str, default=os.path.join(DATA_ROOT, "DeepMIMO_dataset"))
ap.add_argument("--scenario_dir", type=str,
                default=os.environ.get("RALWLM_DEEPMIMO_DIR", os.path.join(DATA_ROOT, "DeepMIMO", "O1_3p5")))
a = ap.parse_args(); rng = np.random.default_rng(a.seed); os.makedirs(a.out, exist_ok=True)
BS = [int(b) for b in a.bs.split(",")]

def base_params(bs, rows, full):
    p = DM.default_params(); p["scenario"] = "O1_3p5"; p["dataset_folder"] = a.scenario_dir
    p["active_BS"] = np.array([bs]); p["user_rows"] = np.asarray(rows); p["enable_BS2BS"] = 0
    p["ue_antenna"]["shape"] = np.array([1, 1])
    if full:
        p["num_paths"] = a.paths; p["bs_antenna"]["shape"] = np.array([a.n_ant, 1])
        p["bs_antenna"]["rotation"] = np.array([0.0, 0.0, a.az])      # rotate ULA about z: boresight = az
        p["OFDM"]["bandwidth"] = a.bw_mhz / 1000.0; p["OFDM"]["subcarriers"] = a.n_sc
        p["OFDM"]["selected_subcarriers"] = np.arange(a.n_sc)
    else:
        p["num_paths"] = 1; p["bs_antenna"]["shape"] = np.array([1, 1])
        p["OFDM"]["subcarriers"] = 4; p["OFDM"]["selected_subcarriers"] = np.arange(4)
    return p

for bs in BS:
    t0 = time.time(); sid = {"near": 300, "mixed": 350, "corner": 400}[a.mode] + bs; sdir = os.path.join(a.out, f"scene_{sid:03d}")
    if os.path.exists(os.path.join(sdir, "config.json")): print(f"[BS {bs}] scene_{sid} exists, skip"); continue
    # ── A. cheap pass over the whole grid, chunked by rows so we know each user's row chunk ──
    locs, los, chunk_id = [], [], []
    for c0 in range(0, a.rows, a.chunk):
        rows = np.arange(c0, min(c0 + a.chunk, a.rows))
        d = DM.generate_data(base_params(bs, rows, full=False))[0]
        locs.append(d["user"]["location"]); los.append(d["user"]["LoS"]); chunk_id.append(np.full(len(d["user"]["LoS"]), c0))
        bs_loc = d["location"]
    loc = np.concatenate(locs); los = np.concatenate(los).astype(int); chunk_id = np.concatenate(chunk_id)
    valid = los >= 0                                              # -1 = no paths (blocked); exclude
    # ── B. choose the 32x32 window ──
    half = a.map / 2; best = None
    if a.mode == "corner":
        cx, cy = bs_loc[0] + a.corner_off + half, bs_loc[1] + a.corner_off + half
        m = (np.abs(loc[:, 0] - cx) <= half) & (np.abs(loc[:, 1] - cy) <= half) & valid
        n = int(m.sum()); fl = float(los[m].mean()) if n else float("nan")
        print(f"[BS {bs}] corner window origin=({cx-half:.1f},{cy-half:.1f}) users={n} LoS={fl:.2f}")
        if n < a.n_ue: print(f"[BS {bs}] corner window has only {n} users (< {a.n_ue}) -- skipped"); continue
        best = (0.0, cx, cy, n, fl, float(np.hypot(half, half)))
    xs = np.arange(loc[:, 0].min() + half, loc[:, 0].max() - half + 1e-6, 2.0) if a.mode != "corner" else np.array([])
    ys = np.arange(loc[:, 1].min() + half, loc[:, 1].max() - half + 1e-6, 2.0)
    for cx in xs:
        mx = np.abs(loc[:, 0] - cx) <= half
        for cy in ys:
            m = mx & (np.abs(loc[:, 1] - cy) <= half) & valid
            n = int(m.sum())
            if n < a.n_ue: continue
            fl = float(los[m].mean())
            dist = float(np.hypot(cx - bs_loc[0], cy - bs_loc[1]))
            if a.mode == "mixed":
                if not (a.los_lo <= fl <= a.los_hi) or dist > a.max_dist: continue
                score = dist + 30.0 * abs(fl - 0.55)      # proximity first, then a balanced LoS mix
            else:
                score = dist                              # near: simply the closest populated window
            if best is None or score < best[0]: best = (score, cx, cy, n, fl, dist)
    cands = []
    for cx in xs:
        mx = np.abs(loc[:, 0] - cx) <= half
        for cy in ys:
            m = mx & (np.abs(loc[:, 1] - cy) <= half) & valid; n = int(m.sum())
            if n < a.n_ue: continue
            fl = float(los[m].mean()); dist = float(np.hypot(cx - bs_loc[0], cy - bs_loc[1]))
            cands.append((dist, fl, n, cx, cy))
    cands.sort()
    if a.mode != "corner": print(f"[BS {bs}] nearest windows (dist, LoS, n): " + "; ".join(f"({d_:.0f}m,{f_:.2f},{n_})" for d_, f_, n_, *_ in cands[:8]))
    if best is None:
        print(f"[BS {bs}] no window satisfies n>={a.n_ue} & LoS in [{a.los_lo},{a.los_hi}] -- skipped"); continue
    _, cx, cy, n, fl, dist = best
    print(f"[BS {bs}] window centre=({cx:.0f},{cy:.0f})  users={n}  LoS={fl:.2f}  dist to BS={dist:.0f} m  (pass A {time.time()-t0:.0f}s)")
    win = (np.abs(loc[:, 0] - cx) <= half) & (np.abs(loc[:, 1] - cy) <= half) & valid
    chunks = sorted(set(chunk_id[win].tolist()))
    # ── C. full channels for the rows that intersect the window ──
    rows = np.concatenate([np.arange(c0, min(c0 + a.chunk, a.rows)) for c0 in chunks])
    d = DM.generate_data(base_params(bs, rows, full=True))[0]
    L = d["user"]["location"]; Lo = d["user"]["LoS"].astype(int)
    m = (np.abs(L[:, 0] - cx) <= half) & (np.abs(L[:, 1] - cy) <= half) & (Lo >= 0)
    idx = np.where(m)[0]; idx = rng.permutation(idx)[:a.n_ue]
    os.makedirs(sdir, exist_ok=True)
    x0, y0 = cx - half, cy - half                                  # window origin -> (0,0)
    bs_rel = np.array([bs_loc[0] - x0, bs_loc[1] - y0, bs_loc[2]], dtype=np.float64)
    az = float(a.az)                                                     # actual ULA boresight (rotation about z)
    los_out = np.zeros(len(idx), dtype=np.int8)
    for k, u in enumerate(idx):
        H = d["user"]["channel"][u][0]                            # (n_ant, n_sc) = (T, F)
        ue = np.array([L[u, 0] - x0, L[u, 1] - y0], dtype=np.float64)
        dist_u = float(np.linalg.norm(np.array([L[u, 0], L[u, 1], L[u, 2]]) - bs_loc))
        with h5py.File(os.path.join(sdir, f"{k:05d}.h5py"), "w") as f:
            f.create_dataset("channel_real", data=np.real(H).T.astype(np.float32))   # (F, T) like Sionna
            f.create_dataset("channel_imag", data=np.imag(H).T.astype(np.float32))
            f.create_dataset("UElocation", data=ue / 10.0)                            # stored /10
            f.create_dataset("BSlocation", data=bs_rel[:2])
            f.create_dataset("distance", data=dist_u / 100.0)                          # stored /100
            f.create_dataset("DoD_phi", data=0.0)
            f.create_dataset("bandwidth", data=a.bw_mhz * 1e6)
            f.create_dataset("LoS", data=int(Lo[u]))
        los_out[k] = Lo[u]
    np.save(os.path.join(sdir, "los.npy"), los_out)
    json.dump({"scene_id": sid, "source": "DeepMIMO O1_3p5", "deepmimo_bs": bs, "window_mode": a.mode, "array_rotation_deg": [0.0, 0.0, a.az], "window_origin_xy": [x0, y0],
               "bs_position": bs_rel.tolist(), "bs_azimuth_deg": az, "bs_downtilt_deg": 0.0, "buildings": [],
               "los_from_file": True, "n_ue": int(len(idx)), "map_size": a.map, "bs_height": float(bs_loc[2]),
               "attenna_cols": a.n_ant, "bandwidth": a.bw_mhz * 1e6, "ue_height": float(L[idx, 2].mean()),
               "los_fraction": float(los_out.mean())},
              open(os.path.join(sdir, "config.json"), "w"), indent=2)
    print(f"[BS {bs}] -> scene_{sid}: {len(idx)} UEs written, LoS {los_out.mean():.2f}, BS rel=({bs_rel[0]:.1f},{bs_rel[1]:.1f},{bs_rel[2]:.1f}) az={az:.0f}  ({time.time()-t0:.0f}s)")
print("=== DONE ===")
