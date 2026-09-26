#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gpus 1
#SBATCH -t 1-12:00:00
#SBATCH -A naiss2026-4-312-gpu
#SBATCH -p gpu
#SBATCH -J ra-train
#SBATCH -o slurm-train-%j.out
#SBATCH -e slurm-train-%j.err
# ─────────────────────────────────────────────────────────────────────
# RA-LWLM 两阶段训练 / two-stage RA-LWLM training.
#   sbatch job-train-ra-lwlm.sh <scene_start> <scene_end> <n_train>
#   sbatch job-train-ra-lwlm.sh 0 20 4000          # 论文主配置 / the paper's main setting
# 参数 / arguments: 训练场景区间、每场景参考样本数 / scene range, references per scene
# 产物 / output: <CKPT_ROOT>/ra_lwlm_kmoe_000_020/kmoe_n4000_K20_lb000_frz_seed42_best.ckpt
# ─────────────────────────────────────────────────────────────────────
# SLURM 会把脚本复制到 spool 目录，所以不能用 $0 推仓库路径；换机器改 RALWLM_REPO 即可
# SLURM copies the script into a spool dir, so $0 cannot locate the repo; set RALWLM_REPO instead
source "${RALWLM_REPO:-/home/guangjin/project_GP/RA-LWLM}/jobs/_common.sh"
S=${1:-0}; E=${2:-20}; N=${3:-4000}; EXTRA="${@:4}"
cd $REPO/ra_lwlm
python train_ra_lwlm.py \
    --scene_start $S --scene_end $E \
    --val_scene_start $S --val_scene_end $E \
    --gen_test_start 100 --gen_test_end 110 \
    --n_train $N \
    --pretrain_epochs 100 --joint_epochs 50 \
    --freeze_icl --lb_lambda 0.0 --gate_temp 1.0 \
    --K_max 20 --ra_num_layers 2 --token_dim 256 --pos_scale 32.0 --dropout 0.1 \
    --lr_icl 1e-4 --lr_kmoe 3e-5 --batch_size 32 --num_workers 4 --seed 42 $EXTRA
RC=$?; echo "End: $(date)"; exit $RC
