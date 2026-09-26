"""
eval_ra_lwlm.py
=====================================================================
RA-LWLM 评测 / Evaluation of RA-LWLM.

中文说明
--------
只评论文的最终方案（K-MoE 软路由），输出：
  1. SS / US 两个划分：SS = 训练时见过的场景，US = 完全未见过的场景（零样本，不做适配）
  2. 每个划分再按 LOS / NLOS 拆开，报均值、中值、90% 分位
  3. CDF 文件（两列：误差(m)  累积概率），供画图
  4. json 汇总

评测协议：检索库 = 每个场景的 UE [0, n_train)；query = UE [test_ue_start, test_ue_end)。
两者不重叠，且 query 从不进入检索库。

EN
--
Evaluates the paper's final scheme (K-MoE with soft routing) only, and reports:
  1. the SS / US splits — SS = scenes seen during training, US = unseen scenes
     (zero-shot, no adaptation whatsoever)
  2. a LOS / NLOS breakdown per split with mean, median and 90th percentile
  3. CDF files (two columns: error(m)  cumulative probability) for plotting
  4. a json summary

Protocol: the database is UE [0, n_train) of each scene and the queries are
UE [test_ue_start, test_ue_end); the two never overlap.

LOS/NLOS 标注 / labelling: 场景目录里有 los.npy 就直接用（射线追踪标签），否则用
BS→UE 直线与建筑长方体的相交测试（与 data_gen/check_los_nlos.py 一致）。
If the scene directory holds los.npy it is used (ray-tracer flag); otherwise a
BS→UE segment / building-box intersection test is applied, identical to
data_gen/check_los_nlos.py.
"""
import functools, builtins
builtins.print = functools.partial(builtins.print, flush=True)

import os, sys, argparse, json, random
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, THIS_DIR); sys.path.insert(0, os.path.join(ROOT, "fm")); sys.path.insert(0, ROOT)

from paths import DATASET_DIR, PRETRAIN_CKPT, RESULTS_DIR
from train_model import Wrapper
from dataload_sionna_v2 import load_h5_sample, process_channel
from ra_lwlm_icl import RA_LWLM_ICL
from ra_lwlm_kmoe import RA_LWLM_KMoE, K_ACTIONS

p = argparse.ArgumentParser()
p.add_argument("--ra_ckpt", type=str, required=True,
               help="训练得到的 K-MoE checkpoint / the trained K-MoE checkpoint")
p.add_argument("--n_train", type=int, required=True,
               help="每场景参考样本数（检索库大小）/ labelled references per scene (DB size)")
# 场景区间 / scene ranges
p.add_argument("--val_scene_start", type=int, default=0)    # SS: 训练见过的场景 / seen scenes
p.add_argument("--val_scene_end",   type=int, default=20)
p.add_argument("--gen_test_start",  type=int, default=100)  # US: 未见场景 / unseen scenes
p.add_argument("--gen_test_end",    type=int, default=110)
# query UE 区间 / query UE range
p.add_argument("--test_ue_start", type=int, default=9000)
p.add_argument("--test_ue_end",   type=int, default=10000)
p.add_argument("--K_max",      type=int, default=20)
p.add_argument("--batch_size", type=int, default=128)
p.add_argument("--num_workers",type=int, default=4)
p.add_argument("--seed",       type=int, default=42)
p.add_argument("--skip_ss", action="store_true", help="只评 US / evaluate the unseen split only")
# 输出 / outputs
p.add_argument("--out_dir",  type=str, default=os.path.join(RESULTS_DIR, "ra_lwlm"),
               help="CDF 与 json 的输出目录 / where the CDF files and json go")
p.add_argument("--tag",      type=str, default="", help="输出文件名后缀 / suffix for output names")
p.add_argument("--save_cdf", type=int, default=1, help="1=保存 CDF 文件 / write CDF files")
# 路径 / paths
p.add_argument("--dataset_dir",   type=str, default=DATASET_DIR)
p.add_argument("--pretrain_ckpt", type=str, default=PRETRAIN_CKPT)
# 模型结构（必须与训练时一致）/ architecture (must match training)
p.add_argument("--embed_dim", type=int, default=256)
p.add_argument("--depth",     type=int, default=4)
p.add_argument("--num_heads", type=int, default=4)
p.add_argument("--patch_t",   type=int, default=4)
p.add_argument("--patch_f",   type=int, default=4)
p.add_argument("--ra_num_layers", type=int,   default=2)
p.add_argument("--ra_token_dim",  type=int,   default=256)
p.add_argument("--ra_pos_scale",  type=float, default=32.0)
p.add_argument("--ra_dropout",    type=float, default=0.2)
p.add_argument("--cfg_dim",       type=int,   default=4)
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed)
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UE_HEIGHT = 1.5          # UE 高度(m)，与数据生成一致 / UE height (m), as in data generation


# ── LOS/NLOS 几何判据 / LOS-NLOS geometry ──────────────────────────────
def line_blocked_by_building(bs_xyz, ue_xyz, b):
    """slab 法：BS→UE 线段是否穿过某栋轴对齐长方体建筑。
    Slab test: does the BS→UE segment pass through this axis-aligned building box?"""
    bx_min, bx_max = b["cx"] - b["length"]/2.0, b["cx"] + b["length"]/2.0
    by_min, by_max = b["cy"] - b["width"]/2.0,  b["cy"] + b["width"]/2.0
    p0 = np.asarray(bs_xyz, float); p1 = np.asarray(ue_xyz, float); d = p1 - p0
    t_enter, t_exit = 0.0, 1.0
    for axis, lo, hi in ((0, bx_min, bx_max), (1, by_min, by_max)):
        if abs(d[axis]) < 1e-9:
            if p0[axis] < lo or p0[axis] > hi: return False
            continue
        t1, t2 = (lo - p0[axis])/d[axis], (hi - p0[axis])/d[axis]
        if t1 > t2: t1, t2 = t2, t1
        t_enter = max(t_enter, t1); t_exit = min(t_exit, t2)
        if t_enter >= t_exit: return False
    # 线段在建筑 xy footprint 内的最低点低于楼高 → 被遮挡
    # The lowest point inside the footprint is below the roof → blocked
    return (p0[2] + t_exit * d[2]) < b["height"]


def is_nlos(ue_xy, bs_xyz, buildings):
    ue_xyz = (float(ue_xy[0]), float(ue_xy[1]), UE_HEIGHT)
    return any(line_blocked_by_building(bs_xyz, ue_xyz, b) for b in buildings)


# ── 场景 / scenes ──────────────────────────────────────────────────────
SCENE_CFG, SCENE_NANT, SCENE_GEO = {}, {}, {}


def load_scene_config(sid):
    """读 config.json → 归一化配置向量 + 天线数 + 几何（BS 位置、建筑）。
    Read config.json → normalised config vector, n_ant and geometry (BS, buildings)."""
    pth = os.path.join(args.dataset_dir, f"scene_{sid:03d}", "config.json")
    if not os.path.exists(pth): return None
    with open(pth) as f: cj = json.load(f)
    n_ant = int(cj.get("attenna_cols", cj.get("n_ant", 0)))
    bw, h, az = float(cj["bandwidth"]), float(cj["bs_height"]), float(np.deg2rad(cj["bs_azimuth_deg"]))
    SCENE_CFG[sid]  = np.array([az, (bw-5e6)/15e6, (n_ant-8)/24.0, (h-15.0)/5.0], np.float32)
    SCENE_NANT[sid] = n_ant
    SCENE_GEO[sid]  = (cj["bs_position"], cj["buildings"])
    return True


def collect(rg): return [s for s in rg if load_scene_config(s)]


SS_SCENES = [] if args.skip_ss else collect(range(args.val_scene_start, args.val_scene_end))
US_SCENES = collect(range(args.gen_test_start, args.gen_test_end))
print(f"SS scenes: {len(SS_SCENES)}   US scenes: {len(US_SCENES)}   n_train={args.n_train}")


class CSIDataset(Dataset):
    """读一个场景内 UE [a, b) 的 CSI 与真实坐标 / CSI + ground-truth positions for UE [a, b)."""
    def __init__(self, sid, a, b): self.sid, self.ues = sid, list(range(a, b))
    def __len__(self): return len(self.ues)
    def __getitem__(self, i):
        ch, ue_loc, *_ = load_h5_sample(self.sid, self.ues[i], dataset_dir=args.dataset_dir)
        aa, _ri, _da = process_channel(ch, input_fmap=2)
        return {'aa_enc': torch.from_numpy(np.transpose(aa, (0, 2, 1)).copy()),
                'pos_m':  torch.from_numpy(np.asarray(ue_loc, np.float32) * 10.0)}


def make_envpara(n_ant):
    """编码器超参（必须与预训练时一致）/ encoder hyper-parameters (must match pretraining)."""
    return {"input_tdim": n_ant, "input_fdim": 128, "input_fmap": 2,
        "fshape": args.patch_f, "tshape": args.patch_t, "fstride": args.patch_f, "tstride": args.patch_t,
        "task": "pretrain_dti", "model_path": "", "load_pretrained_mdl_path": "",
        "pretrain_stage": False, "device": device, "embed_dim": args.embed_dim, "depth": args.depth,
        "latent_dim": 128, "num_heads": args.num_heads, "lr": 1e-4, "is_frozen": 1, "BW": 5,
        "FT_dataset": 1, "pilot_subcarrier_interval": 1, "pilot_antenna_interval": 1,
        "BS_Num": 1, "is_load": 0, "input_feature_dim": 2, "epochs": 1, "BSconf_dim": 4}


_enc = None
def get_encoder():
    """加载并冻结 DTI 预训练编码器（只加载一次）/ load the frozen DTI encoder once."""
    global _enc
    if _enc is None:
        print(f"Loading DTI encoder: {args.pretrain_ckpt}")
        e = Wrapper.load_from_checkpoint(args.pretrain_ckpt,
            EnvPara=make_envpara(SCENE_NANT[(SS_SCENES + US_SCENES)[0]]), strict=False)
        e.eval().to(device)
        for q in e.parameters(): q.requires_grad = False
        _enc = e
    return _enc


@torch.no_grad()
def encode(sid, a, b):
    """编码 UE [a,b)：返回 LST token、patch 均值、真实坐标(m)。
    Encode UE [a,b): returns the LST token, the patch mean and positions in metres."""
    enc = get_encoder()
    ldr = DataLoader(CSIDataset(sid, a, b), batch_size=args.batch_size,
                     num_workers=args.num_workers, pin_memory=True, shuffle=False)
    L, M, P = [], [], []
    for bt in ldr:
        e = enc.channel_fdmdl.fm_encoder(bt['aa_enc'].float().to(device, non_blocking=True))
        L.append(e[:, 0, :].cpu()); M.append(e[:, 1:, :].mean(1).cpu()); P.append(bt['pos_m'])
    return torch.cat(L), torch.cat(M), torch.cat(P)


def load_ra():
    """按训练时的结构重建 K-MoE 并载入权重 / rebuild the K-MoE and load its weights."""
    icls = [RA_LWLM_ICL(embed_dim=args.embed_dim, K_max=args.K_max, num_heads=4,
                        num_layers=args.ra_num_layers, dropout=args.ra_dropout,
                        pos_scale=args.ra_pos_scale, token_dim=args.ra_token_dim,
                        cfg_dim=args.cfg_dim) for _ in K_ACTIONS]
    ra = RA_LWLM_KMoE(embed_dim=args.embed_dim, K_max=args.K_max, num_heads=4,
        num_layers=args.ra_num_layers, dropout=args.ra_dropout, pos_scale=args.ra_pos_scale,
        token_dim=args.ra_token_dim, cfg_dim=args.cfg_dim, selector_dropout=args.ra_dropout,
        gate_temperature=1.0, freeze_icl=True, icl_models=icls).to(device)
    ck = torch.load(args.ra_ckpt, map_location='cpu')
    sd = ck.get('state_dict', ck)
    sd = {k.replace('ra_lwlm.', '', 1): v for k, v in sd.items() if k.startswith('ra_lwlm.')}
    sd = {k: v for k, v in sd.items() if not k.startswith('cccp_head')}
    missing, unexpected = ra.load_state_dict(sd, strict=False)
    assert not missing, f"missing weights in checkpoint: {missing[:5]}"
    ra.eval()
    return ra


RA = load_ra()
print(f"RA-LWLM loaded (k_actions={RA.k_actions}): {args.ra_ckpt}")


@torch.no_grad()
def evaluate(scene_ids, tag):
    """逐场景：编码库与 query → top-K 检索 → RA-LWLM 推理 → LOS/NLOS 标注。
    Per scene: encode the DB and the queries → top-K retrieval → RA-LWLM inference
    → LOS/NLOS labelling. Returns the per-query errors and the NLOS mask."""
    ERR, NL, EFFK = [], [], []
    for n, sid in enumerate(scene_ids, 1):
        db_l, db_m, db_p = encode(sid, 0, args.n_train)                 # 检索库 / database
        q_l,  q_m,  q_p  = encode(sid, args.test_ue_start, args.test_ue_end)
        db_ld, db_md, db_pd = db_l.to(device), db_m.to(device), db_p.to(device)
        q_ld = q_l.to(device)
        cfgv = torch.from_numpy(SCENE_CFG[sid]).to(device)

        # 检索：在 FM 特征空间做 L2 top-K / retrieval: L2 top-K in FM feature space
        K = min(args.K_max, args.n_train)
        dist, idx = torch.topk(torch.cdist(q_ld, db_ld), K, dim=-1, largest=False)
        ref_lst = db_ld[idx]; ref_mp = db_md[idx]; ref_pos = db_pd[idx]
        ref_cfg = cfgv.view(1, 1, -1).expand(idx.shape[0], K, -1).contiguous()
        w = torch.softmax(-dist, dim=-1)                                # 检索权重 / retrieval weights

        errs = []
        for st in range(0, idx.shape[0], 256):
            sl = slice(st, st + 256)
            qf = torch.cat([q_ld[sl], q_m[sl].to(device)], dim=-1)
            rf = torch.cat([ref_lst[sl], ref_mp[sl]], dim=-1)
            cf = cfgv.view(1, -1).expand(qf.shape[0], -1)
            pred = RA.forward_loc(qf, cf, rf, ref_pos[sl], ref_cfg[sl], w[sl], k=None)
            errs.append(torch.norm(pred - q_p[sl].to(device), dim=1).cpu())
            if RA.last_eff_k is not None: EFFK.append(RA.last_eff_k)
        ERR.append(torch.cat(errs))

        los_file = os.path.join(args.dataset_dir, f"scene_{sid:03d}", "los.npy")
        if os.path.exists(los_file):        # 射线追踪给出的 LoS 标签 / ray-tracer LoS flag
            los_arr = np.load(los_file)
            NL.append(torch.tensor(los_arr[args.test_ue_start:args.test_ue_end] == 0, dtype=torch.bool))
        else:                               # 几何遮挡判据 / geometric blockage test
            bs_xyz, blds = SCENE_GEO[sid]
            NL.append(torch.tensor([is_nlos(x, bs_xyz, blds) for x in q_p.numpy()], dtype=torch.bool))

        del db_ld, db_md, db_pd, q_ld; torch.cuda.empty_cache()
        if n % 5 == 0 or n == len(scene_ids): print(f"  [{tag}] {n}/{len(scene_ids)}")
    return torch.cat(ERR).numpy(), torch.cat(NL).numpy(), (float(np.mean(EFFK)) if EFFK else None)


def stats(e):
    """均值 / 中值 / 90% 分位 / mean, median and 90th percentile (empty group → nan)."""
    e = np.asarray(e)
    if e.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "n": 0}
    return {"mean": float(e.mean()), "median": float(np.median(e)),
            "p90": float(np.percentile(e, 90)), "n": int(e.size)}


def save_cdf(e, path, n_points=1000):
    """写两列 CDF 文件：误差(m) 累积概率 / write a two-column CDF: error(m) cumulative prob."""
    errs = np.sort(np.asarray(e)); N = len(errs)
    if N == 0: return
    idx = np.linspace(0, N-1, min(n_points, N)).round().astype(int)
    with open(path, "w") as f:
        for i in idx: f.write(f"{errs[i]:.6f}  {(i+1)/N:.6f}\n")


print(f"\nEvaluating RA-LWLM (DB = UE [0,{args.n_train}), queries = UE "
      f"[{args.test_ue_start},{args.test_ue_end}), K={min(args.K_max, args.n_train)})")
os.makedirs(args.out_dir, exist_ok=True)
RES = {"n_train": args.n_train, "ra_ckpt": args.ra_ckpt, "k_actions": RA.k_actions,
       "test_ue": [args.test_ue_start, args.test_ue_end], "splits": {}}

for split, scenes in (("SS", SS_SCENES), ("US", US_SCENES)):
    if not scenes: continue
    err, nl, eff_k = evaluate(scenes, f"{split}-test")
    a, l, n_ = stats(err), stats(err[~nl]), stats(err[nl])
    RES["splits"][split] = {"scenes": len(scenes), "n_los": int((~nl).sum()),
                            "n_nlos": int(nl.sum()), "effective_k": eff_k,
                            "all": a, "los": l, "nlos": n_}
    print(f"\n{'='*66}\n  {split}   {len(scenes)} scenes   n={err.size}   "
          f"LOS={int((~nl).sum())}  NLOS={int(nl.sum())}"
          + (f"   有效 k / effective k={eff_k:.2f}" if eff_k else "") + f"\n{'='*66}")
    print(f"  {'group':6s} | {'mean':>7s} {'median':>7s} {'p90':>7s}")
    print("  " + "-"*36)
    for name, g in (("ALL", a), ("LOS", l), ("NLOS", n_)):
        print(f"  {name:6s} | {g['mean']:7.3f} {g['median']:7.3f} {g['p90']:7.3f}")
    if args.save_cdf:
        sub = os.path.join(args.out_dir, split.lower()); os.makedirs(sub, exist_ok=True)
        save_cdf(err,      os.path.join(sub, f"ra_lwlm{args.tag}_cdf.txt"))
        save_cdf(err[~nl], os.path.join(sub, f"ra_lwlm{args.tag}_los_cdf.txt"))
        save_cdf(err[nl],  os.path.join(sub, f"ra_lwlm{args.tag}_nlos_cdf.txt"))

out_json = os.path.join(args.out_dir, f"ra_lwlm_n{args.n_train}{args.tag}.json")
with open(out_json, "w") as f: json.dump(RES, f, indent=2)
print(f"\nSaved: {out_json}")
if args.save_cdf: print(f"CDF files: {args.out_dir}/{{ss,us}}/ra_lwlm{args.tag}{{,_los,_nlos}}_cdf.txt")
print("=== DONE ===")
