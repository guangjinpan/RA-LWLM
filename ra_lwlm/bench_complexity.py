"""
bench_complexity.py
=====================================================================
RA-LWLM 复杂度实测 / Complexity measurements for RA-LWLM.

测量内容 / what is measured:
  * 参数量：冻结编码器 + 5 个 ICL 专家 + selector
    parameters: frozen encoder + the five ICL experts + the selector
  * 检索库存储：每个场景需要常驻的张量字节数（fp32）
    database storage: bytes of the tensors that must be kept per scene (fp32)
  * 检索耗时：对 N 个参考样本做 cdist + top-K
    retrieval time: cdist + top-K against the N references
  * 端到端推理延迟：CSI → 坐标，batch=1（在线）与 batch=64（吞吐）
    end-to-end inference latency: CSI → position, batch 1 (online) and 64 (throughput)
  * 逐部件拆解：编码器 / 检索 / selector / 每个专家
    per-component breakdown: encoder / retrieval / selector / each expert
  * 新场景上线成本：只需把参考库编码一遍，不需要训练
    new-scene cost: encode the reference database once, no training at all

计时方式：CUDA event 计时，10 次热身后取 --reps 次的中位数。
Timing: CUDA events, median of --reps runs after 10 warm-up iterations.

输出 / outputs: <out_dir>/complexity.json + complexity.md
"""
import functools, builtins
builtins.print = functools.partial(builtins.print, flush=True)
import os, sys, json, argparse
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np, torch
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
p.add_argument("--ra_ckpt", type=str, required=True)
p.add_argument("--scene",   type=int, default=0,    help="计时所用场景 / scene used for timing")
p.add_argument("--n_train", type=int, default=4000, help="检索库大小 / database size")
p.add_argument("--K",       type=int, default=20)
p.add_argument("--reps",    type=int, default=50)
p.add_argument("--q_start", type=int, default=9000, help="query 起始 UE / first query UE")
p.add_argument("--n_query", type=int, default=64,   help="batch 计时用的 query 数 / queries for batched timing")
p.add_argument("--out_dir", type=str, default=os.path.join(RESULTS_DIR, "complexity"))
p.add_argument("--dataset_dir",   type=str, default=DATASET_DIR)
p.add_argument("--pretrain_ckpt", type=str, default=PRETRAIN_CKPT)
args = p.parse_args(); os.makedirs(args.out_dir, exist_ok=True)
dev = torch.device("cuda")

cfg = json.load(open(os.path.join(args.dataset_dir, f"scene_{args.scene:03d}", "config.json")))
N_ANT = int(cfg.get("attenna_cols", cfg.get("n_ant", 0)))
CFG = torch.tensor([np.deg2rad(cfg["bs_azimuth_deg"]), (cfg["bandwidth"]-5e6)/15e6,
                    (N_ANT-8)/24.0, (cfg["bs_height"]-15.0)/5.0], dtype=torch.float32, device=dev)
print(f"scene {args.scene}: n_ant={N_ANT}, N_db={args.n_train}, K={args.K}")


def envpara(n_ant):
    return {"input_tdim": n_ant, "input_fdim": 128, "input_fmap": 2, "fshape": 4, "tshape": 4,
        "fstride": 4, "tstride": 4, "task": "pretrain_dti", "model_path": "",
        "load_pretrained_mdl_path": "", "pretrain_stage": False, "device": dev, "embed_dim": 256,
        "depth": 4, "latent_dim": 128, "num_heads": 4, "lr": 1e-4, "is_frozen": 1, "BW": 5,
        "FT_dataset": 1, "pilot_subcarrier_interval": 1, "pilot_antenna_interval": 1,
        "BS_Num": 1, "is_load": 0, "input_feature_dim": 2, "epochs": 1, "BSconf_dim": 4}


def nparams(m): return sum(q.numel() for q in m.parameters())
def mb(t): return t.numel() * t.element_size() / 2**20      # 张量占用 MB / tensor size in MB


def timeit(fn, reps=None, warm=10):
    """CUDA event 计时，取中位数（推理，不建自动求导图）。
    CUDA-event timing, median over reps (inference only, no autograd graph)."""
    reps = reps or args.reps
    torch.autograd.set_detect_anomaly(False)     # Wrapper.__init__ 会全局打开它 / enabled globally by Wrapper
    fn = torch.no_grad()(fn)
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return float(np.median(ts))


# ── 数据：参考库 + 一批 query / data: reference DB + a query batch ──────
class DS(Dataset):
    def __init__(s, a, b): s.ues = list(range(a, b))
    def __len__(s): return len(s.ues)
    def __getitem__(s, i):
        ch, loc, *_ = load_h5_sample(args.scene, s.ues[i], dataset_dir=args.dataset_dir)
        aa, _ri, _da = process_channel(ch, input_fmap=2)
        return (torch.from_numpy(np.transpose(aa, (0, 2, 1)).copy()),
                torch.from_numpy(np.asarray(loc, np.float32) * 10))


def load(a, b):
    xs, ps = [], []
    for x, q in DataLoader(DS(a, b), batch_size=256, num_workers=4):
        xs.append(x); ps.append(q)
    return torch.cat(xs).to(dev), torch.cat(ps).to(dev)


db_enc, db_pos = load(0, args.n_train)
q_enc,  q_pos  = load(args.q_start, args.q_start + args.n_query)
print("data loaded")

# ── 模型 / models ──────────────────────────────────────────────────────
enc = Wrapper.load_from_checkpoint(args.pretrain_ckpt, EnvPara=envpara(N_ANT), strict=False).eval().to(dev)
for q in enc.parameters(): q.requires_grad = False


@torch.no_grad()
def fm_feat(x):
    """编码器输出 → (LST token, patch 均值) / encoder output → (LST token, patch mean)."""
    e = enc.channel_fdmdl.fm_encoder(x.float()); return e[:, 0, :], e[:, 1:, :].mean(1)


def build_kmoe():
    icls = [RA_LWLM_ICL(embed_dim=256, K_max=args.K, num_heads=4, num_layers=2,
                        dropout=0.2, pos_scale=32.0, token_dim=256, cfg_dim=4)
            for _ in K_ACTIONS]
    ra = RA_LWLM_KMoE(embed_dim=256, K_max=args.K, num_heads=4, num_layers=2, dropout=0.2,
        pos_scale=32.0, token_dim=256, cfg_dim=4, selector_dropout=0.2,
        gate_temperature=1.0, freeze_icl=True, icl_models=icls).to(dev)
    sd = torch.load(args.ra_ckpt, map_location="cpu"); sd = sd.get("state_dict", sd)
    sd = {k.replace("ra_lwlm.", "", 1): v for k, v in sd.items() if k.startswith("ra_lwlm.")}
    ra.load_state_dict({k: v for k, v in sd.items() if not k.startswith("cccp_head")}, strict=False)
    return ra.eval()


ra = build_kmoe()
n_enc = nparams(enc.channel_fdmdl); n_icl = nparams(ra.icls[0]); n_sel = nparams(ra.selector)

# ── 存储与耗时 / storage and timings ───────────────────────────────────
db_lst, db_mp = fm_feat(db_enc)
q_lst,  q_mp  = fm_feat(q_enc)
# 纯检索数据：LST(256) + patch 均值(256) + 坐标(2)，全 fp32
# Pure retrieval payload: LST(256) + patch mean(256) + position(2), all fp32
db_MB = mb(db_lst) + mb(db_mp) + mb(db_pos)


def retrieve(qq):
    d = torch.cdist(qq, db_lst); return torch.topk(d, args.K, dim=-1, largest=False)


enc_b1  = timeit(lambda: fm_feat(q_enc[:1]))
enc_b64 = timeit(lambda: fm_feat(q_enc))
ret_b1  = timeit(lambda: retrieve(q_lst[:1]))
ret_b64 = timeit(lambda: retrieve(q_lst))
db_build = timeit(lambda: fm_feat(db_enc), reps=5, warm=2)     # 新场景上线成本 / cost of onboarding a scene


@torch.no_grad()
def ra_forward(bs):
    """RA-LWLM 的聚合部分：检索 + selector + 5 专家混合。
    The aggregation part of RA-LWLM: retrieval + selector + mixture of the 5 experts."""
    ql, qm = q_lst[:bs], q_mp[:bs]
    d, idx = retrieve(ql)
    rl, rm, rp = db_lst[idx], db_mp[idx], db_pos[idx]
    w = torch.softmax(-d, -1)
    qf = torch.cat([ql, qm], -1); rf = torch.cat([rl, rm], -1)
    c = CFG.expand(bs, -1); rc = c.unsqueeze(1).expand(-1, args.K, -1)
    return ra.forward_loc(qf, c, rf, rp, rc, w, k=None)


def e2e(bs):
    """端到端：编码 + 检索 + 聚合 / end-to-end: encode + retrieve + aggregate."""
    def f():
        with torch.no_grad(): fm_feat(q_enc[:bs]); ra_forward(bs)
    return timeit(f)


# ── 逐部件拆解（batch 1）/ per-component breakdown (batch 1) ───────────
with torch.no_grad():
    ql, qm = q_lst[:1], q_mp[:1]
    d1, idx1 = retrieve(ql)
    rl, rm, rp1 = db_lst[idx1], db_mp[idx1], db_pos[idx1]
    w1 = torch.softmax(-d1, -1)
    qf1, rf1 = torch.cat([ql, qm], -1), torch.cat([rl, rm], -1)
    c1 = CFG.expand(1, -1); rc1 = c1.unsqueeze(1).expand(-1, args.K, -1)
comp = {"encoder": enc_b1, "retrieval_cdist_topk": ret_b1,
        "selector_logits": timeit(lambda: ra._compute_logits(qf1, c1, rf1, rp1, w1))}
for j, k_ in enumerate(K_ACTIONS):
    comp[f"expert_k{k_}"] = timeit(lambda j=j, k_=k_: ra.icls[j].forward_loc(qf1, c1, rf1, rp1, rc1, w1, k=k_))
comp["all_experts"] = timeit(lambda: ra._all_ICL_forwards(qf1, c1, rf1, rp1, rc1, w1))
comp["forward_loc (selector + experts + mix)"] = timeit(
    lambda: ra.forward_loc(qf1, c1, rf1, rp1, rc1, w1, k=None))

R = {"setting": {"scene": args.scene, "n_ant": N_ANT, "N_db": args.n_train, "K": args.K,
                 "gpu": torch.cuda.get_device_name(0), "reps": args.reps},
     "params_M": {"encoder_frozen": round(n_enc/1e6, 3), "one_ICL_expert": round(n_icl/1e6, 3),
                  "all_experts": round(len(K_ACTIONS)*n_icl/1e6, 3), "selector": round(n_sel/1e6, 3),
                  "aggregator_total": round(nparams(ra)/1e6, 3),
                  "total": round((n_enc + nparams(ra))/1e6, 3)},
     "database_MB_per_scene": round(db_MB, 2),
     "retrieval_ms": {"b1": round(ret_b1, 3), "b64": round(ret_b64, 3)},
     "encoder_ms": {"b1": round(enc_b1, 3), "b64": round(enc_b64, 3)},
     "latency_ms": {"RA-LWLM": {"b1": round(e2e(1), 2), "b64": round(e2e(args.n_query), 2)}},
     "component_breakdown_ms_b1": {k: round(v, 3) for k, v in comp.items()},
     "new_scene_adaptation": f"encode {args.n_train} references: {db_build/1000:.1f} s, no training",
     "notes": {
        "database_MB": "fp32, per scene: LST(256) + patch mean(256) + position(2) per reference",
        "latency": "end-to-end per query: FM encoding + retrieval + aggregation",
        "timing": f"CUDA events, median of {args.reps} runs after 10 warm-ups"}}

print(f"\nGPU: {R['setting']['gpu']}")
print(f"params(M): encoder {n_enc/1e6:.2f} (frozen) | 1 expert {n_icl/1e6:.2f} | "
      f"{len(K_ACTIONS)} experts {len(K_ACTIONS)*n_icl/1e6:.2f} | selector {n_sel/1e6:.2f} | "
      f"total {(n_enc+nparams(ra))/1e6:.2f}")
print(f"database: {db_MB:.2f} MB / scene ({args.n_train} refs)   onboarding: {db_build/1000:.1f} s")
print(f"encoder: b1 {enc_b1:.2f} ms, b{args.n_query} {enc_b64:.2f} ms | "
      f"retrieval: b1 {ret_b1:.3f} ms, b{args.n_query} {ret_b64:.3f} ms")
print("\n=== latency (ms) ===")
for k, v in R["latency_ms"].items(): print(f"  {k:28s} b1={v['b1']}  b{args.n_query}={v['b64']}")
print("\n=== component breakdown, batch 1 (ms) ===")
for k, v in comp.items(): print(f"  {k:24s} {v:7.3f}")

json.dump(R, open(os.path.join(args.out_dir, "complexity.json"), "w"), indent=2)
with open(os.path.join(args.out_dir, "complexity.md"), "w") as f:
    f.write(f"GPU {R['setting']['gpu']}, scene {args.scene} (n_ant={N_ANT}), "
            f"N_db={args.n_train}, K={args.K}\n\n")
    f.write("| Method | Params (M) | Database storage (MB) | Retrieval (ms) | "
            "Inference latency (ms) | New-scene adaptation |\n|---|---|---|---|---|---|\n")
    for k, v in R["latency_ms"].items():
        f.write(f"| {k} | {R['params_M']['total']} | {db_MB:.2f} | {ret_b1:.3f} | {v['b1']} | "
                f"{R['new_scene_adaptation']} |\n")
print(f"\nSaved {args.out_dir}/complexity.{{json,md}}\n=== DONE ===")
