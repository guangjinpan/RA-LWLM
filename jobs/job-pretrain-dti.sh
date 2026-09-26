#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gpus 1
#SBATCH -t 2-00:00:00
#SBATCH -A naiss2026-4-312-gpu
#SBATCH -p gpu
#SBATCH -J ra-pretrain
#SBATCH -o slurm-pretrain-%j.out
#SBATCH -e slurm-pretrain-%j.err
# ─────────────────────────────────────────────────────────────────────
# 基础模型 DTI 自监督预训练（无需位置标签）/ DTI self-supervised pretraining (no labels).
# 8/16/32 天线的数据一起训练，得到一套共享权重。
# The 8/16/32-antenna datasets are trained jointly into one set of weights.
#   sbatch job-pretrain-dti.sh              # embed_dim=256 depth=4 heads=4
# 产物 / output: <CKPT_ROOT>/pretrain_rag/pretrain_dti_latest.ckpt
# ─────────────────────────────────────────────────────────────────────
# SLURM 会把脚本复制到 spool 目录，所以不能用 $0 推仓库路径；换机器改 RALWLM_REPO 即可
# SLURM copies the script into a spool dir, so $0 cannot locate the repo; set RALWLM_REPO instead
source "${RALWLM_REPO:-/home/guangjin/project_GP/RA-LWLM}/jobs/_common.sh"
EMBED=${1:-256}; DEPTH=${2:-4}; HEADS=${3:-4}
DS=$(python -c "import sys;sys.path.insert(0,'$REPO');import paths;print(paths.DATASET_DIR)")
IX=$(python -c "import sys;sys.path.insert(0,'$REPO');import paths;print(paths.INDEX_DIR)")
cd $REPO/fm
python pretrain_dti.py \
    --configs "8,128,${DS},${IX}/n8" "16,128,${DS},${IX}/n16" "32,128,${DS},${IX}/n32" \
    --epochs 200 --batch_size 64 --lr 1e-4 \
    --embed_dim $EMBED --depth $DEPTH --num_heads $HEADS \
    --patch_t 4 --patch_f 4 --num_workers 8 --max_val_samples 60000 --seed 42
RC=$?; echo "End: $(date)"; exit $RC
