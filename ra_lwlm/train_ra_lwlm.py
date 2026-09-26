"""
train_ra_lwlm.py
=====================================================================
RA-LWLM (K-MoE-5) 两阶段训练脚本 / Two-stage training script for RA-LWLM.

中文说明
--------
阶段 1：对 k ∈ {3, 6, 9, 12, 15} 分别独立预训练一个 ICL 专家（固定上下文长度）。
阶段 2：把 5 个专家装进 K-MoE，冻结专家、联合训练 selector（软混合训练 +
        逐 query argmax 硬推理），可选 Switch-Transformer 负载均衡辅助损失。

速度关键：所有检索（top-K 索引 + 距离）在启动时按场景一次性算好并缓存在显存/内存中，
训练/评估阶段只做表查找，没有逐 batch 的编码器前向和 cdist。
  * 训练 query = 检索库自身（leave-one-out，排除自己）
  * 验证/测试 query = 固定的 UE 区间，编码一次即可

EN
--
Stage 1: pretrain one ICL expert per fixed context length k ∈ {3, 6, 9, 12, 15}.
Stage 2: wrap the 5 experts in a K-MoE, freeze them and train the selector
         (soft mixture while training, per-query argmax at inference), with an
         optional Switch-Transformer load-balancing auxiliary loss.

Speed trick: every retrieval (top-K indices + distances) is precomputed once per
scene at start-up and cached, so the hot path is a table lookup — no per-batch
encoder forward and no per-batch cdist.
  * training queries are the database itself (leave-one-out, self excluded)
  * val / test queries are fixed UE ranges, encoded once

Checkpoints / 产物: <ckpt_dir>/kmoe_n{N}_K{K}_lb{...}_frz_seed{S}_best.ckpt
"""
import functools, builtins
builtins.print = functools.partial(builtins.print, flush=True)

import os, sys, argparse, json, random, shutil
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader, Sampler

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "fm"))
sys.path.insert(0, ROOT)

from paths import DATASET_DIR, PRETRAIN_CKPT, CKPT_ROOT

from train_model import Wrapper
from dataload_sionna_v2 import load_h5_sample, process_channel
from ra_lwlm_icl import RA_LWLM_ICL
from ra_lwlm_kmoe import RA_LWLM_KMoE, K_ACTIONS


CFG_DIM = 4


# ─── 数据脚手架 / data scaffolding ─────────────────────────────────────
def load_scene_config(dataset_dir, scene_id):
    """读取场景配置，返回 4 维归一化配置向量 cfg 和天线数。
    Read a scene config and return the 4-D normalised config vector and n_ant.

    cfg = [方位角(rad), (带宽-5MHz)/15MHz, (天线数-8)/24, (BS高度-15)/5]
    cfg = [azimuth(rad), (BW-5MHz)/15MHz, (n_ant-8)/24, (bs_height-15)/5]
    归一化让不同场景配置落在同一量级，是跨场景泛化的前提。
    The normalisation puts heterogeneous scene configs on a common scale, which is
    what allows one model to serve scenes with different BS/array settings."""
    cfg_path = os.path.join(dataset_dir, f"scene_{scene_id:03d}", "config.json")
    if not os.path.exists(cfg_path): return None
    with open(cfg_path) as f: cj = json.load(f)
    n_ant = int(cj.get("attenna_cols", cj.get("n_ant", 0)))
    bw, h, az = float(cj["bandwidth"]), float(cj["bs_height"]), float(np.deg2rad(cj["bs_azimuth_deg"]))
    cfg_vec = np.array([az, (bw - 5e6)/15e6, (n_ant - 8)/24.0, (h - 15.0)/5.0], dtype=np.float32)
    return cfg_vec, n_ant


def collect_scenes(sid_range, target, dataset_dir, nant_cache, cfg_cache):
    for sid in sid_range:
        if sid in nant_cache:
            target.append(sid); continue
        out = load_scene_config(dataset_dir, sid)
        if out is None: continue
        cfg_vec, n_ant = out
        nant_cache[sid] = n_ant; cfg_cache[sid] = cfg_vec
        target.append(sid)


def read_sample(scene_id, ue_idx):
    """读一个 UE 样本：CSI → (角度,时延) 域实/虚双通道，位置换算成米。
    Load one UE sample: CSI → 2-channel (angle, delay) map; position in metres."""
    channel, ue_loc, *_ = load_h5_sample(scene_id, ue_idx, dataset_dir=args.dataset_dir)
    aa, _, _ = process_channel(channel, input_fmap=2)
    aa_enc = np.transpose(aa, (0, 2, 1)).copy()
    pos_m = (np.asarray(ue_loc, dtype=np.float32) * 10.0)
    bs_cfg = SCENE_CFG[scene_id].copy()
    return aa_enc, pos_m, bs_cfg


class SceneSubsetDataset(Dataset):
    def __init__(self, sid, ue_start, ue_end):
        self.sid = sid; self.ues = list(range(ue_start, ue_end))
    def __len__(self): return len(self.ues)
    def __getitem__(self, i):
        aa, pos, cfg = read_sample(self.sid, self.ues[i])
        return (torch.from_numpy(aa), torch.from_numpy(pos), torch.from_numpy(cfg))


def make_envpara(n_ant):
    return {
        "input_tdim": n_ant, "input_fdim": 128, "input_fmap": 2,
        "fshape": args.patch_f, "tshape": args.patch_t,
        "fstride": args.patch_f, "tstride": args.patch_t,
        "task": "pretrain_dti", "model_path": "", "load_pretrained_mdl_path": "",
        "pretrain_stage": False, "device": device,
        "embed_dim": args.embed_dim, "depth": args.depth, "latent_dim": 128,
        "num_heads": args.num_heads, "lr": 1e-4, "is_frozen": 1, "BW": 5, "FT_dataset": 1,
        "pilot_subcarrier_interval": 1, "pilot_antenna_interval": 1,
        "BS_Num": 1, "is_load": 0, "input_feature_dim": 2, "epochs": 1,
    }


class SceneDB:
    """单场景的参考数据库 + 预计算的库内 top-K（供训练 query 使用）。
    Per-scene reference database + precomputed intra-DB top-K (for train queries)."""
    # lst/mp : 编码器输出的 [LST token; patch 均值] / encoder outputs
    # pos/cfg: 参考点坐标 (m) 与场景配置向量 / reference positions (m) and scene config
    # topk_* : 库内 leave-one-out 的 top-K 索引与 -距离 / cached LOO top-K over the DB
    __slots__ = ("lst", "mp", "pos", "cfg", "n_ant", "topk_idx", "topk_negd")


@torch.no_grad()
def build_scene_db(scene_id, ue_start, ue_end):
    """把 UE [ue_start, ue_end) 编码成检索库（冻结编码器前向一次）。
    Encode UE [ue_start, ue_end) into the reference database (one frozen-encoder pass)."""
    ds = SceneSubsetDataset(scene_id, ue_start, ue_end)
    loader = DataLoader(ds, batch_size=256, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    lsts, mps, poss, cfgs = [], [], [], []
    for aa, pos, cfg in loader:
        aa = aa.to(device, non_blocking=True)
        enc = encoder.channel_fdmdl.fm_encoder(aa)
        lsts.append(enc[:, 0, :].cpu()); mps.append(enc[:, 1:, :].mean(1).cpu())
        poss.append(pos); cfgs.append(cfg)
    db = SceneDB()
    db.lst = torch.cat(lsts); db.mp = torch.cat(mps)
    db.pos = torch.cat(poss); db.cfg = torch.cat(cfgs)
    db.n_ant = SCENE_NANT[scene_id]
    db.topk_idx = None; db.topk_negd = None
    return db


@torch.no_grad()
def precompute_db_topk(db, K_max, dev, chunk=2048):
    """库内 leave-one-out 检索：对每一行 i，在库中（排除 i 自己）取 top-K_max 近邻。
    Intra-DB leave-one-out retrieval: for every row i take its top-K_max neighbours
    among the DB, excluding i itself. Returns (idx [M, K], -dist [M, K]) on CPU.

    训练 query 就是库本身，所以这张表在所有 epoch 之间都不变，只需算一次。
    Training queries are the DB itself, so this table is constant across epochs."""
    M = db.lst.shape[0]
    topk_idx  = torch.zeros(M, K_max, dtype=torch.long)
    topk_negd = torch.zeros(M, K_max, dtype=torch.float32)
    lst_g = db.lst.to(dev)
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        d = torch.cdist(lst_g[s:e], lst_g)                                # [chunk, M]
        rows = torch.arange(e - s, device=dev)
        cols = torch.arange(s, e, device=dev)
        d[rows, cols] = float('inf')                                      # 排除自己 / exclude self
        d_top, i_top = torch.topk(d, K_max, dim=-1, largest=False)
        topk_idx[s:e]  = i_top.cpu()
        topk_negd[s:e] = (-d_top).cpu()                                   # 取负，越大越近 / negate: higher = closer
    del lst_g
    db.topk_idx  = topk_idx
    db.topk_negd = topk_negd
    torch.cuda.empty_cache()


@torch.no_grad()
def precompute_query_pool(db, scene_id, ue_start, ue_end, K_max, dev, chunk=2048):
    """编码验证/测试 query（UE [ue_start, ue_end)）并预先算好它们对库的 top-K_max。
    Encode the val/test queries once and precompute their top-K_max against the DB.
    Returns q_lst, q_mp, q_pos, q_cfg, topk_idx, topk_negd (all on CPU)."""
    ds = SceneSubsetDataset(scene_id, ue_start, ue_end)
    loader = DataLoader(ds, batch_size=256, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    lsts, mps, poss, cfgs = [], [], [], []
    for aa, pos, cfg in loader:
        aa = aa.to(dev, non_blocking=True)
        enc = encoder.channel_fdmdl.fm_encoder(aa)
        lsts.append(enc[:, 0, :].cpu()); mps.append(enc[:, 1:, :].mean(1).cpu())
        poss.append(pos); cfgs.append(cfg)
    q_lst = torch.cat(lsts); q_mp = torch.cat(mps)
    q_pos = torch.cat(poss); q_cfg = torch.cat(cfgs)
    Q = q_lst.shape[0]
    topk_idx  = torch.zeros(Q, K_max, dtype=torch.long)
    topk_negd = torch.zeros(Q, K_max, dtype=torch.float32)
    db_g = db.lst.to(dev); q_g = q_lst.to(dev)
    for s in range(0, Q, chunk):
        e = min(s + chunk, Q)
        d = torch.cdist(q_g[s:e], db_g)
        d_top, i_top = torch.topk(d, K_max, dim=-1, largest=False)
        topk_idx[s:e]  = i_top.cpu()
        topk_negd[s:e] = (-d_top).cpu()
    del db_g, q_g
    torch.cuda.empty_cache()
    return {"q_lst": q_lst, "q_mp": q_mp, "q_pos": q_pos, "q_cfg": q_cfg,
            "topk_idx": topk_idx, "topk_negd": topk_negd}


def build_all_dbs(scene_ids, ue_start, ue_end, tag, K_max):
    print(f"\nBuilding {tag} scene DBs ×{len(scene_ids)} (UE [{ue_start},{ue_end}))  + pre-topK")
    dbs = {}
    for i, sid in enumerate(scene_ids):
        db = build_scene_db(sid, ue_start, ue_end)
        precompute_db_topk(db, K_max, device)
        dbs[sid] = db
        if (i+1) % 10 == 0 or (i+1) == len(scene_ids):
            print(f"  [{i+1}/{len(scene_ids)}] scene {sid}  N={db.lst.shape[0]}  topK done")
    return dbs


def build_query_pools(scene_ids, ue_start, ue_end, dbs, tag, K_max):
    print(f"\nBuilding {tag} query pools ×{len(scene_ids)} (UE [{ue_start},{ue_end}))  + pre-topK")
    pools = {}
    for i, sid in enumerate(scene_ids):
        if sid not in dbs: continue
        pools[sid] = precompute_query_pool(dbs[sid], sid, ue_start, ue_end, K_max, device)
        if (i+1) % 10 == 0 or (i+1) == len(scene_ids):
            print(f"  [{i+1}/{len(scene_ids)}] scene {sid}  Q={pools[sid]['q_lst'].shape[0]}")
    return pools


# ─── 查表（运行时不再做 cdist）/ lookup helpers (no cdist at runtime) ──
def _gather_refs(db, idx, k):
    """idx: [B, K_max] long → take first k columns and gather DB rows.
       Returns ref_lst, ref_mp, ref_pos, ref_cfg as CPU tensors."""
    sub = idx[:, :k]                                              # [B, k]
    return db.lst[sub], db.mp[sub], db.pos[sub], db.cfg[sub]


def _train_lookup(db, local_idx, k, dev):
    """训练：query 就在库里，用预计算的 LOO top-K；权重 w = softmax(-距离)。
    Training: the query lives in the DB, so use the precomputed LOO top-K.
    Retrieval weights are w = softmax(-distance)."""
    li = local_idx.cpu() if torch.is_tensor(local_idx) else torch.as_tensor(local_idx)
    sub_idx = db.topk_idx[li, :k]
    sub_nd  = db.topk_negd[li, :k]
    w = torch.softmax(sub_nd.to(dev), dim=-1)
    ref_lst, ref_mp, ref_pos, ref_cfg = _gather_refs(db, db.topk_idx[li], k)
    return ref_lst.to(dev), ref_mp.to(dev), ref_pos.to(dev), ref_cfg.to(dev), w


def _query_lookup(db, pool_topk_idx, pool_topk_negd, qi, k, dev):
    """验证/测试：query 特征与 top-K 都已在 pool 中预计算好。
    Val/test: both the query features and their top-K are precomputed in the pool."""
    qi_c = qi.cpu() if torch.is_tensor(qi) else torch.as_tensor(qi)
    sub_idx = pool_topk_idx[qi_c, :k]
    sub_nd  = pool_topk_negd[qi_c, :k]
    w = torch.softmax(sub_nd.to(dev), dim=-1)
    ref_lst = db.lst[sub_idx]; ref_mp = db.mp[sub_idx]
    ref_pos = db.pos[sub_idx]; ref_cfg = db.cfg[sub_idx]
    return ref_lst.to(dev), ref_mp.to(dev), ref_pos.to(dev), ref_cfg.to(dev), w


# ─── Datasets ──────────────────────────────────────────────────────────
class TrainSceneDataset(Dataset):
    """训练集：query = 库本身（leave-one-out，检索时排除自己）。
    Training set: queries are the DB itself (leave-one-out; self excluded)."""
    def __init__(self, scene_dbs):
        self.scene_dbs = scene_dbs; self.samples = []; self.scene_to_global = {}
        for sid, db in scene_dbs.items():
            local = []
            for i in range(db.lst.shape[0]):
                local.append(len(self.samples)); self.samples.append((sid, i))
            self.scene_to_global[sid] = local
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        sid, idx = self.samples[i]; db = self.scene_dbs[sid]
        return {'scene_id': sid, 'local_idx': idx,
                'lst': db.lst[idx].float(), 'mp': db.mp[idx].float(),
                'pos': db.pos[idx].float(), 'cfg': db.cfg[idx].float()}


class ValPoolDataset(Dataset):
    """验证/测试集：直接用预计算的 query 特征与 top-K。
    Val/test set: reads the precomputed query features and top-K tables."""
    def __init__(self, pools):
        self.pools = pools; self.samples = []; self.scene_to_global = {}
        for sid, pool in pools.items():
            local = []
            for qi in range(pool['q_lst'].shape[0]):
                local.append(len(self.samples)); self.samples.append((sid, qi))
            self.scene_to_global[sid] = local
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        sid, qi = self.samples[i]; pool = self.pools[sid]
        return {'scene_id': sid, 'qi': qi,
                'lst': pool['q_lst'][qi].float(), 'mp': pool['q_mp'][qi].float(),
                'pos': pool['q_pos'][qi].float(), 'cfg': pool['q_cfg'][qi].float()}


class SceneBatchSampler(Sampler):
    """每个 batch 只来自同一个场景 —— 因为检索库、配置向量是按场景组织的。
    Every batch comes from a single scene, since the DB and config vector are
    per-scene. Shuffled batches are drawn scene by scene during training."""
    def __init__(self, sg, batch_size, shuffle=True, drop_last=True):
        self.sg = sg; self.scene_ids = sorted(sg.keys())
        self.batch_size = batch_size; self.shuffle = shuffle; self.drop_last = drop_last
    def __iter__(self):
        if self.shuffle:
            for _ in range(len(self)):
                sid = random.choice(self.scene_ids); pool = self.sg[sid]
                yield random.sample(pool, min(self.batch_size, len(pool)))
        else:
            for sid in self.scene_ids:
                pool = list(self.sg[sid])
                for s in range(0, len(pool), self.batch_size):
                    chunk = pool[s:s+self.batch_size]
                    if len(chunk) == self.batch_size or not self.drop_last:
                        yield chunk
    def __len__(self):
        total = 0
        for sid in self.scene_ids:
            n = len(self.sg[sid])
            total += (n // self.batch_size) if self.drop_last else ((n+self.batch_size-1)//self.batch_size)
        return total


# ─── Lightning modules ────────────────────────────────────────────────
class _BaseTrainer(pl.LightningModule):
    def __init__(self, train_dbs, val_pools):
        super().__init__()
        self.train_dbs = train_dbs; self.val_pools = val_pools
        self.train_loc_m = []; self.val_ra_m = []
        self.train_eff_k = []; self.train_lb = []

    def on_train_epoch_end(self):
        if self.train_loc_m:
            msg = f"Epoch {self.current_epoch} - Train Loc={torch.stack(self.train_loc_m).mean():.3f}m"
            if self.train_eff_k: msg += f"  eff_k={np.mean(self.train_eff_k):.2f}"
            if self.train_lb:    msg += f"  lb={np.mean(self.train_lb):.3f}"
            print(msg)
            self.train_loc_m.clear(); self.train_eff_k.clear(); self.train_lb.clear()

    def on_validation_epoch_end(self):
        if self.val_ra_m:
            ra = torch.stack(self.val_ra_m).mean()
            print(f"Epoch {self.current_epoch} - Val RA={ra:.3f}m")
            self.val_ra_m.clear()


class ICLPretrainFixedK(_BaseTrainer):
    """阶段 1：在固定上下文长度 k = k_value 下预训练单个 ICL 专家。
    Stage 1: pretrain one ICL expert at the fixed context length k = k_value."""
    def __init__(self, train_dbs, val_pools, k_value):
        super().__init__(train_dbs, val_pools)
        self.k_value = k_value
        self.ra_lwlm = RA_LWLM_ICL(
            embed_dim=args.embed_dim, K_max=args.K_max,
            num_heads=4, num_layers=args.ra_num_layers, dropout=args.dropout,
            pos_scale=args.pos_scale,
            token_dim=args.token_dim if args.token_dim > 0 else None, cfg_dim=CFG_DIM,
        )

    def _run(self, q_lst, q_mp, q_cfg, ref_lst, ref_mp, ref_pos, ref_cfg, weights, k):
        q_feat = torch.cat([q_lst, q_mp], dim=-1)
        ref_feat = torch.cat([ref_lst, ref_mp], dim=-1)
        return self.ra_lwlm.forward_loc(q_feat, q_cfg, ref_feat, ref_pos, ref_cfg, weights, k=k)

    def training_step(self, batch, batch_idx):
        sids = batch['scene_id']; sid = int(sids[0]) if torch.is_tensor(sids) else int(sids[0])
        db = self.train_dbs[sid]; local_idx = batch['local_idx']
        q_lst = batch['lst'].float(); q_mp = batch['mp'].float()
        q_pos = batch['pos'].float(); q_cfg = batch['cfg'].float(); dev = q_lst.device
        ref_lst, ref_mp, ref_pos, ref_cfg, weights = _train_lookup(db, local_idx, self.k_value, dev)
        pred = self._run(q_lst, q_mp, q_cfg, ref_lst.float(), ref_mp.float(),
                         ref_pos.float(), ref_cfg.float(), weights, self.k_value)
        S = self.ra_lwlm.POS_SCALE
        loc_loss = torch.mean(torch.sqrt(torch.sum(((pred - q_pos) / S) ** 2, dim=1)))
        loc_loss_m = loc_loss.detach() * S
        self.train_loc_m.append(loc_loss_m)
        self.log('train/loc_m', loc_loss_m, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return {'loss': loc_loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        sids = batch['scene_id']; sid = int(sids[0]) if torch.is_tensor(sids) else int(sids[0])
        db = self.train_dbs.get(sid); pool = self.val_pools.get(sid)
        if db is None or pool is None: return
        qi = batch['qi']
        q_lst = batch['lst'].float(); q_mp = batch['mp'].float()
        pos = batch['pos'].float(); cfg = batch['cfg'].float(); dev = q_lst.device
        ref_lst, ref_mp, ref_pos, ref_cfg, weights = _query_lookup(
            db, pool['topk_idx'], pool['topk_negd'], qi, args.K_max, dev)
        pred = self._run(q_lst, q_mp, cfg, ref_lst.float(), ref_mp.float(),
                         ref_pos.float(), ref_cfg.float(), weights, self.k_value)
        ra_err = torch.norm(pred - pos, dim=1).mean()
        self.val_ra_m.append(ra_err.detach())
        self.log('val/ave_loss', ra_err, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.ra_lwlm.parameters(), lr=args.lr_icl)


class KMoEJointTrainer(_BaseTrainer):
    """阶段 2：联合训练 K-MoE（软混合前向）+ 负载均衡辅助损失；专家可冻结。
    Stage 2: jointly train the K-MoE (soft-mixture forward) with the load-balancing
    auxiliary loss. With --freeze_icl only the selector is updated."""
    def __init__(self, train_dbs, val_pools, ra_lwlm: RA_LWLM_KMoE, lr: float):
        super().__init__(train_dbs, val_pools)
        self.ra_lwlm = ra_lwlm
        self._stage_lr = lr

    def _run(self, q_lst, q_mp, q_cfg, ref_lst, ref_mp, ref_pos, ref_cfg, weights):
        q_feat = torch.cat([q_lst, q_mp], dim=-1)
        ref_feat = torch.cat([ref_lst, ref_mp], dim=-1)
        return self.ra_lwlm.forward_loc(q_feat, q_cfg, ref_feat, ref_pos, ref_cfg, weights, k=None)

    def training_step(self, batch, batch_idx):
        sids = batch['scene_id']; sid = int(sids[0]) if torch.is_tensor(sids) else int(sids[0])
        db = self.train_dbs[sid]; local_idx = batch['local_idx']
        q_lst = batch['lst'].float(); q_mp = batch['mp'].float()
        q_pos = batch['pos'].float(); q_cfg = batch['cfg'].float(); dev = q_lst.device
        ref_lst, ref_mp, ref_pos, ref_cfg, weights = _train_lookup(db, local_idx, args.K_max, dev)
        pred = self._run(q_lst, q_mp, q_cfg, ref_lst.float(), ref_mp.float(),
                         ref_pos.float(), ref_cfg.float(), weights)
        S = self.ra_lwlm.POS_SCALE
        loc_loss = torch.mean(torch.sqrt(torch.sum(((pred - q_pos) / S) ** 2, dim=1)))
        loc_loss_m = loc_loss.detach() * S
        self.train_loc_m.append(loc_loss_m)
        if self.ra_lwlm.last_eff_k is not None:
            self.train_eff_k.append(self.ra_lwlm.last_eff_k)

        loss = loc_loss
        if args.lb_lambda > 0:
            lb = self.ra_lwlm.get_load_balance_loss()
            loss = loss + args.lb_lambda * lb
            self.train_lb.append(float(lb.detach()))
            self.log('train/lb', lb.detach(), on_step=False, on_epoch=True, sync_dist=True)

        self.log('train/loc_m', loc_loss_m, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return {'loss': loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        sids = batch['scene_id']; sid = int(sids[0]) if torch.is_tensor(sids) else int(sids[0])
        db = self.train_dbs.get(sid); pool = self.val_pools.get(sid)
        if db is None or pool is None: return
        qi = batch['qi']
        q_lst = batch['lst'].float(); q_mp = batch['mp'].float()
        pos = batch['pos'].float(); cfg = batch['cfg'].float(); dev = q_lst.device
        ref_lst, ref_mp, ref_pos, ref_cfg, weights = _query_lookup(
            db, pool['topk_idx'], pool['topk_negd'], qi, args.K_max, dev)
        pred = self._run(q_lst, q_mp, cfg, ref_lst.float(), ref_mp.float(),
                         ref_pos.float(), ref_cfg.float(), weights)
        ra_err = torch.norm(pred - pos, dim=1).mean()
        self.val_ra_m.append(ra_err.detach())
        self.log('val/ave_loss', ra_err, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        params = [p for p in self.ra_lwlm.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self._stage_lr)


def load_ra_lwlm_state(ra_lwlm, ckpt_file):
    """从 Lightning checkpoint 里取出 ra_lwlm.* 权重（丢掉早期版本的 CCCP 辅助头）。
    Pull the ra_lwlm.* weights out of a Lightning checkpoint (dropping the CCCP
    auxiliary head of older checkpoints) and load them into the given module."""
    ckpt = torch.load(ckpt_file, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    sd = {k.replace('ra_lwlm.', '', 1): v for k, v in state.items() if k.startswith('ra_lwlm.')}
    sd = {k: v for k, v in sd.items() if not k.startswith('cccp_head')}
    ra_lwlm.load_state_dict(sd, strict=True)


@torch.no_grad()
def eval_kmoe(module, loader, dbs, pools, name):
    """评测 K-MoE：报 RA-LWLM 的误差，并打印 selector 的专家使用分布。
    Evaluate the K-MoE: RA-LWLM error plus the selector's expert-usage histogram."""
    if loader is None or not dbs or not pools:
        print(f"\n[skip] {name}"); return
    err, eff_ks = [], []
    n_actions = module.ra_lwlm.n_actions; k_actions = module.ra_lwlm.k_actions
    action_counts = np.zeros(n_actions, dtype=np.int64)
    module.ra_lwlm.eval()
    for batch in loader:
        sid = int(batch['scene_id'][0]) if torch.is_tensor(batch['scene_id']) else int(batch['scene_id'][0])
        db = dbs.get(sid); pool = pools.get(sid)
        if db is None or pool is None: continue
        qi = batch['qi']
        q_lst = batch['lst'].float().to(device); q_mp = batch['mp'].float().to(device)
        pos = batch['pos'].float().to(device); cfg = batch['cfg'].float().to(device)
        ref_lst, ref_mp, ref_pos, ref_cfg, weights = _query_lookup(
            db, pool['topk_idx'], pool['topk_negd'], qi, args.K_max, device)
        ref_lst = ref_lst.float(); ref_mp = ref_mp.float()
        ref_pos = ref_pos.float(); ref_cfg = ref_cfg.float()
        pred = module._run(q_lst, q_mp, cfg, ref_lst, ref_mp, ref_pos, ref_cfg, weights)
        err.extend(torch.norm(pred - pos, dim=1).cpu().tolist())
        if module.ra_lwlm.last_eff_k is not None:
            eff_ks.append(module.ra_lwlm.last_eff_k)
        gate = module.ra_lwlm.last_gate
        if gate is not None:
            sel = gate.argmax(dim=-1).cpu().numpy()
            for s in sel: action_counts[int(s)] += 1
    err = np.array(err)
    n_total = int(action_counts.sum())
    print(f"\n{'─'*60}\n  {name}   n={len(err)}\n{'─'*60}")
    print(f"  RA-LWLM : mean={err.mean():.3f}m   median={np.median(err):.3f}m  "
          f"p90={np.percentile(err, 90):.3f}m")
    if eff_ks:
        print(f"  eff_k   : mean={np.mean(eff_ks):.2f}")
    if n_total > 0:
        print(f"  专家使用分布（selector argmax）/ expert usage histogram (total={n_total}):")
        for ki, cnt in zip(k_actions, action_counts):
            pct = 100.0 * cnt / n_total
            bar = '█' * int(pct / 2)
            print(f"    k={ki:>3d}: {cnt:>6d} ({pct:5.1f}%) {bar}")


# ─── 命令行参数 / argparser ───────────────────────────────────────────
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain_epochs", type=int, default=50)
    p.add_argument("--joint_epochs",    type=int, default=50)
    p.add_argument("--skip_pretrain", action="store_true")
    p.add_argument("--lb_lambda",  type=float, default=0.01)
    p.add_argument("--gate_temp",  type=float, default=1.0)
    p.add_argument("--freeze_icl", action="store_true",
                   help="阶段 2 冻结 5 个专家、只训 selector（论文设置）/ "
                        "stage 2 freezes the five experts and trains only the selector (paper setting)")

    p.add_argument("--scene_start",     type=int, default=0)
    p.add_argument("--scene_end",       type=int, default=20)
    p.add_argument("--val_scene_start", type=int, default=0)
    p.add_argument("--val_scene_end",   type=int, default=20)
    p.add_argument("--gen_test_start",  type=int, default=100)
    p.add_argument("--gen_test_end",    type=int, default=110)
    p.add_argument("--n_train",         type=int, default=2000)
    # query UE 区间：库是 UE [0, n_train)，下面三段必须与库不重叠
    # query UE ranges; the DB is UE [0, n_train) and these must not overlap it
    p.add_argument("--val_ue_start",  type=int, default=8000)   # 训练期验证 / validation during training
    p.add_argument("--val_ue_end",    type=int, default=9000)
    p.add_argument("--test_ue_start", type=int, default=9000)   # 最终测试 / final test
    p.add_argument("--test_ue_end",   type=int, default=10000)
    p.add_argument("--lr_icl",     type=float, default=1e-4)
    p.add_argument("--lr_kmoe",    type=float, default=3e-5)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--num_workers",type=int,   default=4)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--K_max",         type=int,   default=20)
    p.add_argument("--ra_num_layers", type=int,   default=2)
    p.add_argument("--token_dim",     type=int,   default=256)
    p.add_argument("--pos_scale",     type=float, default=32.0)
    p.add_argument("--dropout",       type=float, default=0.2)

    # 路径默认值来自 paths.py / path defaults come from paths.py
    p.add_argument("--dataset_dir",   type=str, default=DATASET_DIR)
    p.add_argument("--pretrain_ckpt", type=str, default=PRETRAIN_CKPT)
    p.add_argument("--ckpt_dir",      type=str, default=None,
                   help="默认 <CKPT_ROOT>/ra_lwlm_kmoe_<scene_start>_<scene_end> / "
                        "defaults to <CKPT_ROOT>/ra_lwlm_kmoe_<scene_start>_<scene_end>")

    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth",     type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--patch_t",   type=int, default=4)
    p.add_argument("--patch_f",   type=int, default=4)
    return p


# ─── runtime ──────────────────────────────────────────────────────────
args = build_argparser().parse_args()

if args.ckpt_dir is None:
    args.ckpt_dir = os.path.join(
        CKPT_ROOT, f"ra_lwlm_kmoe_{args.scene_start:03d}_{args.scene_end:03d}")

random.seed(args.seed); np.random.seed(args.seed)
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.ckpt_dir, exist_ok=True)

print(f"\n=== RA-LWLM K-MoE  (precomputed retrieval) ===")
print(f"  ckpt_dir:   {args.ckpt_dir}")
print(f"  k_actions:  {K_ACTIONS}")
print(f"  freeze_icl: {args.freeze_icl}")
print(f"  lb_lambda:  {args.lb_lambda}   dropout={args.dropout}")
print(f"  pretrain:   {args.pretrain_epochs} ep × {len(K_ACTIONS)} ICLs (sequential)")
print(f"  joint ft:   {args.joint_epochs} ep")

# 场景列表：训练 / 同场景验证(SS) / 未见场景测试(US)
# Scene lists: training / seen-scene validation (SS) / unseen-scene test (US)
TRAIN_SCENES, VAL_SCENES, GEN_TEST_SCENES = [], [], []
SCENE_NANT, SCENE_CFG = {}, {}
collect_scenes(range(args.scene_start, args.scene_end), TRAIN_SCENES, args.dataset_dir, SCENE_NANT, SCENE_CFG)
collect_scenes(range(args.val_scene_start, args.val_scene_end), VAL_SCENES, args.dataset_dir, SCENE_NANT, SCENE_CFG)
collect_scenes(range(args.gen_test_start, args.gen_test_end), GEN_TEST_SCENES, args.dataset_dir, SCENE_NANT, SCENE_CFG)
print(f"  scenes: train [{args.scene_start},{args.scene_end}) ({len(TRAIN_SCENES)})"
      f"  val [{args.val_scene_start},{args.val_scene_end}) ({len(VAL_SCENES)})"
      f"  gen [{args.gen_test_start},{args.gen_test_end}) ({len(GEN_TEST_SCENES)})")

# 冻结的 DTI 预训练编码器（RA-LWLM 不更新它）
# The frozen DTI-pretrained encoder — RA-LWLM never updates it
_first_sid = (TRAIN_SCENES if TRAIN_SCENES else VAL_SCENES)[0]
print(f"\nLoading pretrained encoder: {args.pretrain_ckpt}")
encoder = Wrapper.load_from_checkpoint(
    args.pretrain_ckpt, EnvPara=make_envpara(SCENE_NANT[_first_sid]), strict=False)
encoder.eval().to(device)
for p in encoder.parameters(): p.requires_grad = False

# 各场景检索库（含预计算的库内 LOO top-K）
# Per-scene reference databases (with the precomputed LOO top-K)
TRAIN_DBS = build_all_dbs(TRAIN_SCENES, 0, args.n_train, "train", args.K_max)
VAL_DBS   = build_all_dbs(VAL_SCENES,   0, args.n_train, "val",   args.K_max)
GEN_DBS   = build_all_dbs(GEN_TEST_SCENES, 0, args.n_train, "gen", args.K_max) if GEN_TEST_SCENES else {}

# query 池：编码 + 检索各算一次
# Query pools: encode once, retrieve once
VAL_TRAIN_POOLS = build_query_pools(VAL_SCENES, args.val_ue_start, args.val_ue_end, VAL_DBS, "val_during_train", args.K_max)
VAL_TEST_POOLS  = build_query_pools(VAL_SCENES, args.test_ue_start, args.test_ue_end, VAL_DBS, "in_test", args.K_max)
GEN_TEST_POOLS  = build_query_pools(GEN_TEST_SCENES, args.test_ue_start, args.test_ue_end, GEN_DBS, "gen_test", args.K_max) if GEN_TEST_SCENES else {}

# 数据集 / DataLoader（batch 内同场景）
# Datasets and loaders (each batch stays inside one scene)
train_ds = TrainSceneDataset(TRAIN_DBS)
train_loader = DataLoader(train_ds,
    batch_sampler=SceneBatchSampler(train_ds.scene_to_global, args.batch_size, shuffle=True),
    num_workers=args.num_workers, pin_memory=True)

val_in_ds = ValPoolDataset(VAL_TRAIN_POOLS)
val_in_loader = DataLoader(val_in_ds,
    batch_sampler=SceneBatchSampler(val_in_ds.scene_to_global, args.batch_size, shuffle=False, drop_last=False),
    num_workers=args.num_workers, pin_memory=True)

in_test_ds = ValPoolDataset(VAL_TEST_POOLS)
in_te_loader = DataLoader(in_test_ds,
    batch_sampler=SceneBatchSampler(in_test_ds.scene_to_global, args.batch_size, shuffle=False, drop_last=False),
    num_workers=args.num_workers, pin_memory=True)

if GEN_TEST_POOLS:
    gen_test_ds = ValPoolDataset(GEN_TEST_POOLS)
    gen_te_loader = DataLoader(gen_test_ds,
        batch_sampler=SceneBatchSampler(gen_test_ds.scene_to_global, args.batch_size, shuffle=False, drop_last=False),
        num_workers=args.num_workers, pin_memory=True)
else:
    gen_te_loader = None


# ─── 阶段 1：逐 k 预训练 5 个 ICL 专家 / Stage 1: pretrain the 5 experts ──
def stage1_pretrain_one(idx, k_value):
    name = f"icl_idx{idx}_k{k_value:02d}_n{args.n_train}_epo{args.pretrain_epochs}"
    ckpt_path = os.path.join(args.ckpt_dir, f"{name}_best.ckpt")
    if args.skip_pretrain and os.path.exists(ckpt_path):
        print(f"[stage 1 skip] {name}"); return ckpt_path
    print(f"\n{'='*68}\n  Stage 1.{idx+1}/{len(K_ACTIONS)}: pretrain ICL at k={k_value}  "
          f"({args.pretrain_epochs} ep)\n{'='*68}")
    module = ICLPretrainFixedK(TRAIN_DBS, VAL_TRAIN_POOLS, k_value=k_value).to(device)
    n = sum(p.numel() for p in module.ra_lwlm.parameters())
    print(f"  ICL params: {n/1e6:.2f}M")
    best = ModelCheckpoint(dirpath=args.ckpt_dir, filename=name + "_{epoch:03d}",
                           save_top_k=1, monitor="val/ave_loss", mode="min", save_weights_only=True)
    trainer = Trainer(max_epochs=args.pretrain_epochs, accelerator='gpu', devices=1, precision=16,
                      logger=pl.loggers.TensorBoardLogger(args.ckpt_dir, name=f"tb_{name}"),
                      callbacks=[best], log_every_n_steps=20, enable_progress_bar=False, gradient_clip_val=1.0)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_in_loader)
    print(f"  [done] best: {best.best_model_path}  (val {best.best_model_score})")
    if best.best_model_path and os.path.exists(best.best_model_path):
        shutil.copy(best.best_model_path, ckpt_path)
        print(f"  best copied to {ckpt_path}")
    return ckpt_path


icl_paths = [stage1_pretrain_one(i, k) for i, k in enumerate(K_ACTIONS)]


# ─── 阶段 2：冻结专家，训练 selector / Stage 2: freeze experts, train router ──
print(f"\n{'='*68}\n  Stage 2: joint finetune K-MoE  ({args.joint_epochs} ep)  "
      f"lb_lambda={args.lb_lambda}\n{'='*68}")
icls = []
for i, (k_i, p) in enumerate(zip(K_ACTIONS, icl_paths)):
    icl = RA_LWLM_ICL(
        embed_dim=args.embed_dim, K_max=args.K_max, num_heads=4, num_layers=args.ra_num_layers,
        dropout=args.dropout, pos_scale=args.pos_scale,
        token_dim=args.token_dim if args.token_dim > 0 else None, cfg_dim=CFG_DIM)
    load_ra_lwlm_state(icl, p)
    icls.append(icl)
    print(f"  loaded ICL[{i}] for k={k_i} from {os.path.basename(p)}")

multi = RA_LWLM_KMoE(
    embed_dim=args.embed_dim, K_max=args.K_max, num_heads=4, num_layers=args.ra_num_layers,
    dropout=args.dropout, pos_scale=args.pos_scale,
    token_dim=args.token_dim if args.token_dim > 0 else None, cfg_dim=CFG_DIM,
    selector_dropout=args.dropout, gate_temperature=args.gate_temp,
    freeze_icl=args.freeze_icl, icl_models=icls,
)
n_total = sum(p.numel() for p in multi.parameters())
n_train = sum(p.numel() for p in multi.parameters() if p.requires_grad)
print(f"\n  K-MoE params: {n_total/1e6:.2f}M total, {n_train/1e6:.2f}M trainable")
print(f"  k_actions: {multi.k_actions}")

joint_module = KMoEJointTrainer(TRAIN_DBS, VAL_TRAIN_POOLS, multi, lr=args.lr_kmoe).to(device)
_frz = "_frz" if args.freeze_icl else ""
basename = f"kmoe_n{args.n_train}_K{args.K_max}_lb{int(args.lb_lambda*1000):03d}{_frz}_seed{args.seed}"
best2 = ModelCheckpoint(dirpath=args.ckpt_dir, filename=basename + "_{epoch:03d}",
                       save_top_k=1, monitor="val/ave_loss", mode="min", save_weights_only=True)
trainer2 = Trainer(max_epochs=args.joint_epochs, accelerator='gpu', devices=1, precision=16,
                  logger=pl.loggers.TensorBoardLogger(args.ckpt_dir, name=f"tb_{basename}"),
                  callbacks=[best2], log_every_n_steps=20, enable_progress_bar=False, gradient_clip_val=1.0)
trainer2.fit(joint_module, train_dataloaders=train_loader, val_dataloaders=val_in_loader)
print(f"\n  [Stage 2 done] best: {best2.best_model_path}  (val {best2.best_model_score})")

final_ckpt = os.path.join(args.ckpt_dir, basename + "_best.ckpt")
if best2.best_model_path and os.path.exists(best2.best_model_path):
    shutil.copy(best2.best_model_path, final_ckpt)
    load_ra_lwlm_state(joint_module.ra_lwlm, final_ckpt)
    joint_module.to(device); joint_module.ra_lwlm.to(device)


# ─── 最终评测 / final evaluation ──────────────────────────────────────
print(f"\n{'='*68}\n  Final Eval — RA-LWLM (K-MoE)\n{'='*68}")
eval_kmoe(joint_module, in_te_loader, VAL_DBS, VAL_TEST_POOLS,
           f"In-scene Test (SS)  scenes [{args.val_scene_start},{args.val_scene_end})  ue [{args.test_ue_start},{args.test_ue_end})")
eval_kmoe(joint_module, gen_te_loader, GEN_DBS, GEN_TEST_POOLS,
           f"Unseen Test (US)    scenes [{args.gen_test_start},{args.gen_test_end})  ue [{args.test_ue_start},{args.test_ue_end})")

print("\n=== DONE ===")
