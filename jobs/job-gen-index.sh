#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-02:00:00
#SBATCH -A naiss2026-4-312-gpu
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -J ra-gen-index
#SBATCH -o slurm-gen-index-%j.out
#SBATCH -e slurm-gen-index-%j.err
# 扫描数据集，按天线数分组生成预训练索引 / build the per-antenna-count pretraining index.
# SLURM 会把脚本复制到 spool 目录，所以不能用 $0 推仓库路径；换机器改 RALWLM_REPO 即可
# SLURM copies the script into a spool dir, so $0 cannot locate the repo; set RALWLM_REPO instead
source "${RALWLM_REPO:-/home/guangjin/project_GP/RA-LWLM}/jobs/_common.sh"
cd $REPO/data_gen
python gen_pretrain_index.py
RC=$?; echo "End: $(date)"; exit $RC
