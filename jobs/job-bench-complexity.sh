#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gpus 1
#SBATCH -t 0-02:00:00
#SBATCH -A naiss2026-4-312-gpu
#SBATCH -p gpu
#SBATCH -J ra-bench
#SBATCH -o slurm-bench-%j.out
#SBATCH -e slurm-bench-%j.err
# 参数量 / 检索库存储 / 检索耗时 / 推理延迟
# Parameters, database storage, retrieval time, inference latency.
#   sbatch job-bench-complexity.sh <ckpt> [n_train]
# SLURM 会把脚本复制到 spool 目录，所以不能用 $0 推仓库路径；换机器改 RALWLM_REPO 即可
# SLURM copies the script into a spool dir, so $0 cannot locate the repo; set RALWLM_REPO instead
source "${RALWLM_REPO:-/home/guangjin/project_GP/RA-LWLM}/jobs/_common.sh"
CKPT=$1; N=${2:-4000}
if [ -z "$CKPT" ]; then echo "usage: sbatch job-bench-complexity.sh <ckpt> [n_train]"; exit 1; fi
cd $REPO/ra_lwlm
python bench_complexity.py --ra_ckpt $CKPT --scene 0 --n_train $N --K 20 --reps 50 \
    --out_dir $REPO/results/complexity
RC=$?; echo "End: $(date)"; exit $RC
