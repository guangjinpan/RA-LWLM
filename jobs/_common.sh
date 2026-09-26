# ─────────────────────────────────────────────────────────────────────
# _common.sh
# 所有 job 脚本共用的环境设置 / shared environment for every job script.
# 换集群时只改这个文件 / when moving to another cluster, edit only this file.
# 用法 / usage: source "${RALWLM_REPO:-<repo path>}/jobs/_common.sh"
# ─────────────────────────────────────────────────────────────────────

# 计算环境（模块 + Python 虚拟环境）/ compute environment (modules + venv)
module load GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
source /nobackup/proj/disk/wireless_fm_data/shared/python_env/bin/activate
export PYTHONPATH=/nobackup/proj/disk/wireless_fm_data/personal/guangjin/py_extra:$PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE

# 仓库根目录 / repository root
# 注意：job 在 SLURM 的 spool 目录里执行，$0 不指向本仓库，所以以 BASH_SOURCE 为准
# Note: jobs execute from a SLURM spool dir, so resolve the repo from BASH_SOURCE
export REPO="${RALWLM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# 数据与检查点路径（与 paths.py 的环境变量同名；留空则用 paths.py 的默认值）
# Data and checkpoint paths (same env var names as paths.py; leave unset to use its defaults)
# export RALWLM_DATA_ROOT=/nobackup/proj/disk/wireless_fm_data/personal/guangjin/sionna_data

echo "REPO=$REPO"
echo "Start: $(date)"
