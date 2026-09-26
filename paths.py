"""
paths.py
=====================================================================
全仓库唯一的路径配置 / The single place where all paths are configured.

中文：所有脚本的默认数据/检查点/日志路径都来自这里。换机器时只改这一个文件
      （或者设置对应的环境变量），不需要动任何脚本。
EN:   Every script takes its default dataset / checkpoint / log path from here.
      When moving to another machine, edit this file only (or export the matching
      environment variables) — no script needs to be touched.

不设任何环境变量时：集群目录存在就用它，否则一律落在 <仓库>/data 下面，
本地单卡机器开箱即用。With no environment variable set: the cluster directory is
used when present, otherwise everything lives under <repo>/data, so a local
single-GPU machine works out of the box.

环境变量 / environment variables:
    RALWLM_DATA_ROOT    数据与检查点的根目录 / root for data and checkpoints
    RALWLM_DATASET_DIR  场景数据集（scene_XXX/*.h5py）/ scene dataset
    RALWLM_INDEX_DIR    预训练索引 / pretraining index
    RALWLM_CKPT_ROOT    检查点根目录 / checkpoint root
    RALWLM_PRETRAIN_CKPT  DTI 预训练权重 / DTI-pretrained encoder weights
    RALWLM_LOG_DIR      SLURM 日志 / SLURM logs
"""
import os

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# 集群上的既有数据目录；不存在时（例如本地单卡机器）自动回落到 <仓库>/data
# The existing data directory on the cluster; when it is absent (e.g. a local
# single-GPU box) the root falls back to <repo>/data automatically.
_CLUSTER_ROOT = "/nobackup/proj/disk/wireless_fm_data/personal/guangjin/sionna_data"

# 根目录：数据集、索引、检查点、日志都挂在它下面
# Root directory holding the dataset, index, checkpoints and logs
DATA_ROOT = os.environ.get(
    "RALWLM_DATA_ROOT",
    _CLUSTER_ROOT if os.path.isdir(_CLUSTER_ROOT) else os.path.join(_REPO_ROOT, "data"))

# 场景数据集：每个场景一个目录 scene_XXX/{config.json, 00000.h5py, ...}
# Scene dataset: one directory per scene, scene_XXX/{config.json, 00000.h5py, ...}
DATASET_DIR = os.environ.get("RALWLM_DATASET_DIR", os.path.join(DATA_ROOT, "RAG_dataset"))

# 预训练索引（按天线数分组的 (scene_id, ue_idx) 列表）
# Pretraining index: (scene_id, ue_idx) lists grouped by antenna count
INDEX_DIR = os.environ.get("RALWLM_INDEX_DIR", os.path.join(DATASET_DIR, "RAG_index"))

# 检查点根目录 / checkpoint root
CKPT_ROOT = os.environ.get("RALWLM_CKPT_ROOT", os.path.join(DATA_ROOT, "checkpoints"))

# DTI 自监督预训练得到的编码器权重（RA-LWLM 冻结使用）
# Encoder weights from DTI self-supervised pretraining (frozen inside RA-LWLM)
PRETRAIN_CKPT = os.environ.get(
    "RALWLM_PRETRAIN_CKPT",
    os.path.join(CKPT_ROOT, "pretrain_rag", "pretrain_dti_latest.ckpt"))

# SLURM 日志目录 / SLURM log directory
LOG_DIR = os.environ.get("RALWLM_LOG_DIR", os.path.join(DATA_ROOT, "logs"))

# 本仓库根目录 / repository root
REPO_ROOT = _REPO_ROOT
# 结果目录（CDF、json 表格等）/ results directory (CDF files, json tables, ...)
RESULTS_DIR = os.environ.get("RALWLM_RESULTS_DIR", os.path.join(REPO_ROOT, "results"))
