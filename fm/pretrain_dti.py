"""
pretrain_dti.py
=====================================================================
无线基础模型的 DTI 自监督预训练 / DTI self-supervised pretraining of the WFM.

中文
----
任务：把频域 CSI 重构到时延-角度域（Delay-Time/angle Imaging, DTI）。
  * 完全不需要位置标签，用的是海量无标注 CSI
  * 可以同时训练天线数不同的多个数据集（8/16/32 天线共享一套权重），
    这正是 RA-LWLM 能跨场景配置泛化的基础
  * 产物 pretrain_dti_latest.ckpt 会被 RA-LWLM 冻结加载

EN
--
Task: reconstruct the delay-angle domain from the frequency-domain CSI (DTI).
  * needs no position labels at all — it consumes unlabelled CSI
  * datasets with different antenna counts (8/16/32) can be trained jointly into
    one set of weights, which is what lets RA-LWLM generalise across scene configs
  * the resulting pretrain_dti_latest.ckpt is loaded frozen by RA-LWLM

用法（单数据集）/ single dataset:
    python pretrain_dti.py \
        --epochs 200 --batch_size 64 --lr 1e-4 \
        --n_antennas 16 --n_subcarriers 128 \
        --dataset_dir /path/to/dataset \
        --index_dir   /path/to/index \
        --model_path  /path/to/save

用法（多数据集，天线/子载波可不同）/ several datasets with different array sizes:
    python pretrain_dti.py \
        --epochs 200 --batch_size 64 --lr 1e-4 \
        --configs \
            "16,128,/ds1,/idx1" \
            "32,256,/ds2,/idx2" \
        --model_path /path/to/save

--configs 格式: "n_antennas,n_subcarriers,dataset_dir,index_dir"
"""

import os
import sys
import random
import argparse
import math
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Sampler
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_model import Wrapper
from dataload_sionna_v2 import SionnaDataset
from paths import CKPT_ROOT

# 默认把预训练权重写到 <CKPT_ROOT>/pretrain_rag（RA-LWLM 默认从这里加载）
# Pretrained weights default to <CKPT_ROOT>/pretrain_rag, where RA-LWLM looks for them
_DEFAULT_MODEL_PATH = os.path.join(CKPT_ROOT, "pretrain_rag")

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _auto_patch(dim, candidates=(4, 2, 1)):
    """Choose the largest candidate that evenly divides dim."""
    for p in candidates:
        if dim % p == 0:
            return p
    return 1


def _resolve_patches(n_antennas, n_subcarriers, patch_t=0, patch_f=0):
    pt = patch_t if patch_t > 0 else _auto_patch(n_antennas,    (4, 2, 1))
    pf = patch_f if patch_f > 0 else _auto_patch(n_subcarriers, (4, 8, 2, 1))
    assert n_antennas    % pt == 0, \
        f"n_antennas ({n_antennas}) not divisible by patch_t ({pt})"
    assert n_subcarriers % pf == 0, \
        f"n_subcarriers ({n_subcarriers}) not divisible by patch_f ({pf})"
    return pt, pf


# ──────────────────────────────────────────────────────────────────────────────
# Dataset wrappers
# ──────────────────────────────────────────────────────────────────────────────

class _DTISubset(Dataset):
    """Single-config DTI dataset: drops scene_ue_id, exposes (T, F) key."""

    def __init__(self, split, max_samples, n_antennas, n_subcarriers,
                 dataset_dir='', index_dir=''):
        self.shape_key = (n_antennas, n_subcarriers)
        self.base = SionnaDataset(
            split=split, max_samples=max_samples,
            input_tdim=n_antennas, input_fdim=n_subcarriers,
            dataset_dir=dataset_dir, index_dir=index_dir,
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        ch_aa, ch_ri, ch_da, ue_label, bs_conf, _ = self.base[idx]
        return ch_aa, ch_ri, ch_da, ue_label, bs_conf


class MultiConfigDataset(Dataset):
    """Concatenate multiple _DTISubset instances with (possibly) different shapes.

    Tracks per-sample shape_key so a SameShapeBatchSampler can ensure that
    every batch has homogeneous (T, F) — required for torch.stack in collate.
    """

    def __init__(self, configs, split, max_samples_per_config=None):
        """
        configs: list of (n_antennas, n_subcarriers, dataset_dir, index_dir)
        """
        self._datasets = []
        self._offsets  = []
        self.shape_keys = []   # shape_key per global sample index

        total = 0
        for n_ant, n_sub, ds_dir, idx_dir in configs:
            sub = _DTISubset(split, max_samples_per_config,
                             n_ant, n_sub, ds_dir, idx_dir)
            self._offsets.append(total)
            self._datasets.append(sub)
            self.shape_keys.extend([sub.shape_key] * len(sub))
            total += len(sub)

        self._total = total

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        # Binary search for the right sub-dataset
        lo, hi = 0, len(self._datasets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        local_idx = idx - self._offsets[lo]
        return self._datasets[lo][local_idx]


# ──────────────────────────────────────────────────────────────────────────────
# Same-shape batch sampler
# ──────────────────────────────────────────────────────────────────────────────

class SameShapeBatchSampler(Sampler):
    """Yield batches where every sample has the same (T, F) shape.

    Different batches may have different shapes (the transformer handles this
    via dynamic positional encoding), but within each batch shapes are equal
    so torch.stack succeeds in the default collate_fn.
    """

    def __init__(self, shape_keys, batch_size, shuffle=True, drop_last=True):
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.drop_last  = drop_last

        # Group global indices by shape
        groups = defaultdict(list)
        for i, key in enumerate(shape_keys):
            groups[key].append(i)
        self._groups = dict(groups)

        # Pre-compute total batches for __len__
        self._n_batches = 0
        for indices in self._groups.values():
            n = len(indices)
            self._n_batches += n // batch_size if drop_last else math.ceil(n / batch_size)

    def __iter__(self):
        all_batches = []
        for key, indices in self._groups.items():
            idx = list(indices)
            if self.shuffle:
                random.shuffle(idx)
            for start in range(0, len(idx), self.batch_size):
                chunk = idx[start:start + self.batch_size]
                if len(chunk) == self.batch_size or not self.drop_last:
                    all_batches.append(chunk)

        if self.shuffle:
            random.shuffle(all_batches)

        return iter(all_batches)

    def __len__(self):
        return self._n_batches


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--epochs",      type=int,   default=200)
parser.add_argument("--batch_size",  type=int,   default=64)
parser.add_argument("--lr",          type=float, default=1e-4)
parser.add_argument("--num_workers", type=int,   default=8)
parser.add_argument("--max_train_samples", type=int, default=None,
                    help="每个数据集限制训练样本数（None=全部）")
parser.add_argument("--max_val_samples",   type=int, default=8000,
                    help="每个数据集限制验证样本数")
parser.add_argument("--embed_dim",   type=int,   default=256)
parser.add_argument("--depth",       type=int,   default=4)
parser.add_argument("--num_heads",   type=int,   default=4)
parser.add_argument("--model_path",  type=str, default=_DEFAULT_MODEL_PATH,
                    help="检查点输出目录 / checkpoint output directory")
parser.add_argument("--seed",        type=int,   default=42)

# ── 单数据集模式（向后兼容） ────────────────────────────────────────────────
parser.add_argument("--n_antennas",    type=int, default=16)
parser.add_argument("--n_subcarriers", type=int, default=128)
parser.add_argument("--patch_t",       type=int, default=0,
                    help="天线维度 patch 大小（0=自动）")
parser.add_argument("--patch_f",       type=int, default=0,
                    help="子载波维度 patch 大小（0=自动）")
parser.add_argument("--dataset_dir",   type=str, default="",
                    help="H5 数据集根目录（空=使用模块默认值）")
parser.add_argument("--index_dir",     type=str, default="",
                    help="索引 npy 目录（空=使用模块默认值）")

# ── 多数据集模式 ──────────────────────────────────────────────────────────────
parser.add_argument("--configs", nargs='+', default=None,
                    help=(
                        '多数据集配置，每项格式: "n_antennas,n_subcarriers,dataset_dir,index_dir"\n'
                        '示例: --configs "16,128,/ds1,/idx1" "32,256,/ds2,/idx2"\n'
                        '若指定此参数，则忽略 --n_antennas/--n_subcarriers/--dataset_dir/--index_dir'
                    ))

args = parser.parse_args()
set_seed(args.seed)


# ──────────────────────────────────────────────────────────────────────────────
# Build config list
# ──────────────────────────────────────────────────────────────────────────────

def _parse_config_str(s):
    """Parse 'n_ant,n_sub,ds_dir,idx_dir' → (int, int, str, str)."""
    parts = s.split(',', 3)
    if len(parts) != 4:
        raise ValueError(
            f"--configs entry must be 'n_antennas,n_subcarriers,dataset_dir,index_dir', got: {s!r}")
    n_ant, n_sub, ds_dir, idx_dir = parts
    return int(n_ant), int(n_sub), ds_dir.strip(), idx_dir.strip()


if args.configs:
    config_list = [_parse_config_str(s) for s in args.configs]
else:
    config_list = [(args.n_antennas, args.n_subcarriers,
                    args.dataset_dir, args.index_dir)]

# Determine patch sizes from the first config (patch size must divide all configs)
first_n_ant, first_n_sub, _, _ = config_list[0]
patch_t, patch_f = _resolve_patches(first_n_ant, first_n_sub, args.patch_t, args.patch_f)

# Validate that patch sizes divide every config
for n_ant, n_sub, _, _ in config_list:
    assert n_ant % patch_t == 0, \
        f"n_antennas={n_ant} not divisible by patch_t={patch_t}"
    assert n_sub % patch_f == 0, \
        f"n_subcarriers={n_sub} not divisible by patch_f={patch_f}"


# ──────────────────────────────────────────────────────────────────────────────
# EnvPara (use first config dims for model init; model is fully dynamic)
# ──────────────────────────────────────────────────────────────────────────────

EnvPara = {
    "epochs":        args.epochs,
    "input_tdim":    first_n_ant,   # used only for PatchEmbed.num_patches logging
    "input_fdim":    first_n_sub,
    "input_fmap":    2,
    "fshape": patch_f, "tshape": patch_t,
    "fstride": patch_f, "tstride": patch_t,
    "task":          "pretrain_dti",
    "model_path":    args.model_path,
    "load_pretrained_mdl_path": "",
    "pretrain_stage": True,
    "device":        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "embed_dim":     args.embed_dim,
    "depth":         args.depth,
    "latent_dim":    128,
    "num_heads":     args.num_heads,
    "lr":            args.lr,
    "is_frozen":     0,
    "BW":            5,
    "FT_dataset":    640000,
    "pilot_subcarrier_interval": 1,
    "pilot_antenna_interval":    1,
    "BS_Num":        1,
    "is_load":       0,
    "input_feature_dim": 2,
}

os.makedirs(args.model_path, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Sionna DTI Pretraining ===")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"  embed_dim={args.embed_dim}  depth={args.depth}  heads={args.num_heads}")
    print(f"  patch_t={patch_t}  patch_f={patch_f}")
    print(f"  数据集配置 ({len(config_list)} 个):")
    for n_ant, n_sub, ds_dir, idx_dir in config_list:
        n_patches = (n_ant // patch_t) * (n_sub // patch_f)
        print(f"    antennas={n_ant}  subcarriers={n_sub}  "
              f"patches={n_patches}  dataset={ds_dir or '(default)'}")
    print(f"  model_path={args.model_path}")

    # ── Build datasets ────────────────────────────────────────────────────────
    train_ds = MultiConfigDataset(config_list, split='train',
                                  max_samples_per_config=args.max_train_samples)
    val_ds   = MultiConfigDataset(config_list, split='in_scene_test',
                                  max_samples_per_config=args.max_val_samples)

    print(f"  train samples={len(train_ds)}  val samples={len(val_ds)}")

    # ── Batch samplers (ensure same shape within each batch) ─────────────────
    train_sampler = SameShapeBatchSampler(
        train_ds.shape_keys, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_sampler   = SameShapeBatchSampler(
        val_ds.shape_keys,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_sampler=val_sampler,
                              num_workers=args.num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Wrapper(EnvPara=EnvPara)
    model.to(EnvPara["device"])

    # ── Callbacks ────────────────────────────────────────────────────────────
    _tag = f"d{args.embed_dim}_dp{args.depth}_h{args.num_heads}_pt{args.patch_t}_pf{args.patch_f}"
    best_ckpt = ModelCheckpoint(
        dirpath=args.model_path,
        filename=f"pretrain_dti_{_tag}_best_{{epoch:03d}}",
        save_top_k=1,
        monitor="val/ave_loss",
        mode="min",
        save_weights_only=True,
    )
    latest_ckpt = ModelCheckpoint(
        dirpath=args.model_path,
        filename=f"pretrain_dti_{_tag}_latest",
        save_top_k=1,
        save_last=True,
        every_n_epochs=10,
        save_weights_only=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        save_dir=args.model_path, name="tensorboard"
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=1,
        precision=16,
        logger=logger,
        callbacks=[best_ckpt, latest_ckpt],
        log_every_n_steps=50,
        enable_progress_bar=False,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"\n=== 预训练完成 ===")
    print(f"最佳模型: {best_ckpt.best_model_path}")
