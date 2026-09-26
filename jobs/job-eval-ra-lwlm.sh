#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gpus 1
#SBATCH -t 0-04:00:00
#SBATCH -A naiss2026-4-312-gpu
#SBATCH -p gpu
#SBATCH -J ra-eval
#SBATCH -o slurm-eval-%j.out
#SBATCH -e slurm-eval-%j.err
# ─────────────────────────────────────────────────────────────────────
# RA-LWLM 评测：SS / US × LOS / NLOS，并保存 CDF
# RA-LWLM evaluation: SS / US x LOS / NLOS, with CDF files.
#   sbatch job-eval-ra-lwlm.sh <ckpt> [n_train]
#   sbatch job-eval-ra-lwlm.sh $CKPT 4000
# ─────────────────────────────────────────────────────────────────────
# SLURM 会把脚本复制到 spool 目录，所以不能用 $0 推仓库路径；换机器改 RALWLM_REPO 即可
# SLURM copies the script into a spool dir, so $0 cannot locate the repo; set RALWLM_REPO instead
source "${RALWLM_REPO:-/home/guangjin/project_GP/RA-LWLM}/jobs/_common.sh"
CKPT=$1; N=${2:-4000}; EXTRA="${@:3}"
if [ -z "$CKPT" ]; then echo "usage: sbatch job-eval-ra-lwlm.sh <ckpt> [n_train]"; exit 1; fi
cd $REPO/ra_lwlm
python eval_ra_lwlm.py \
    --ra_ckpt $CKPT --n_train $N \
    --val_scene_start 0 --val_scene_end 20 \
    --gen_test_start 100 --gen_test_end 110 \
    --test_ue_start 9000 --test_ue_end 10000 \
    --K_max 20 --batch_size 128 --num_workers 4 --seed 42 \
    --out_dir $REPO/results/ra_lwlm_n${N} $EXTRA
RC=$?; echo "End: $(date)"; exit $RC
