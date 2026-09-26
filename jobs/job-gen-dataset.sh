#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gpus 1
#SBATCH -t 1-00:00:00
#SBATCH -A naiss2026-4-312-gpu
#SBATCH -p gpu
#SBATCH -J ra-gen-data
#SBATCH -o slurm-gen-data-%j.out
#SBATCH -e slurm-gen-data-%j.err
# ─────────────────────────────────────────────────────────────────────
# 用 Sionna 光线追踪生成场景数据集 / generate the scene dataset with Sionna RT.
#   sbatch job-gen-dataset.sh 0   20        # 训练场景（随机配置）/ training scenes (random config)
#   sbatch job-gen-dataset.sh 100 110 1     # 未见测试场景（固定配置）/ unseen test scenes (fixed config)
# 提示：Sionna 只能在 GPU 节点跑，登录节点导入会失败。
# Note: Sionna only runs on the GPU nodes; importing it on the login node fails.
# ─────────────────────────────────────────────────────────────────────
# SLURM 会把脚本复制到 spool 目录，所以不能用 $0 推仓库路径；换机器改 RALWLM_REPO 即可
# SLURM copies the script into a spool dir, so $0 cannot locate the repo; set RALWLM_REPO instead
source "${RALWLM_REPO:-/home/guangjin/project_GP/RA-LWLM}/jobs/_common.sh"
S=${1:-0}; E=${2:-20}; FIXED=${3:-0}
cd $REPO/data_gen
python gen_dataset.py --scene_start $S --scene_end $E --n_ue 10000 --seed 42 --fixed_cfg $FIXED
RC=$?; echo "End: $(date)"; exit $RC
