"""
gen_pretrain_index.py
=====================================================================
预训练索引生成 / Build the pretraining index.

EN: Scan the dataset, group the scenes by antenna count (8/16/32) and write the
    shuffled (scene_id, ue_idx) index files that pretrain_dti.py consumes. Samples
    with different antenna counts cannot share a batch, hence one index per group.

中文说明如下：
扫描 RAG_dataset，按 n_ant 分组，生成训练/验证/测试索引文件。

RAG_dataset 结构：
    RAG_dataset/
        scene_000/
            config.json        ← 含 n_ant, bandwidth, n_ue 等字段
            00000.h5py
            00001.h5py
            ...
        scene_001/
            ...

输出结构：
    RAG_index/
        n8/
            train_shuffled.npy        # shape (N, 2): [scene_id, ue_idx]
            in_scene_test_shuffled.npy
            gen_test.npy
        n16/
            ...
        n32/
            ...
        scene_meta.json               # 每个 scene 的 n_ant / n_ue 汇总

用法：
    python gen_rag_index.py \
        --dataset_dir <RAG_dataset> \
        --output_dir  <RAG_index> \
        --train_ratio 0.7 \
        --val_ratio   0.15 \
        --seed        42
"""

import os
import json
import argparse
import numpy as np

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
from paths import DATASET_DIR, INDEX_DIR   # 默认路径见仓库根目录 paths.py / defaults live in paths.py

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR,
                    help='场景数据集目录 / scene dataset directory')
parser.add_argument('--output_dir',  type=str, default=INDEX_DIR,
                    help='索引输出目录 / index output directory')
parser.add_argument('--train_ratio', type=float, default=0.70,
                    help='训练集场景比例')
parser.add_argument('--val_ratio',   type=float, default=0.15,
                    help='验证集（in_scene_test）场景比例；剩余为 gen_test')
parser.add_argument('--seed',        type=int,   default=42)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: 扫描所有场景，读 config.json，按天线数分组
# Step 1: scan all scenes, read config.json and group them by antenna count
# ──────────────────────────────────────────────────────────────────────────────

print(f"扫描数据集: {args.dataset_dir}")

scene_dirs = sorted([
    d for d in os.listdir(args.dataset_dir)
    if d.startswith('scene_') and
       os.path.isdir(os.path.join(args.dataset_dir, d))
])

if len(scene_dirs) == 0:
    raise RuntimeError(f"未找到任何 scene_* 目录: {args.dataset_dir}")

print(f"找到 {len(scene_dirs)} 个场景目录")

# n_ant → list of (scene_id, n_ue)
groups = {}        # {n_ant: [(scene_id, n_ue), ...]}
scene_meta = {}    # {scene_id: {n_ant, n_ue, bandwidth}}
missing_config = []
missing_nue    = []

for d in scene_dirs:
    scene_id  = int(d.split('_')[1])
    cfg_path  = os.path.join(args.dataset_dir, d, 'config.json')

    if not os.path.exists(cfg_path):
        missing_config.append(scene_id)
        continue

    with open(cfg_path) as f:
        cfg = json.load(f)

    n_ant = cfg.get('n_ant')
    n_ue  = cfg.get('n_ue')

    if n_ant is None or n_ue is None:
        # 尝试从 h5 文件计数兜底
        scene_dir_path = os.path.join(args.dataset_dir, d)
        h5_files = [x for x in os.listdir(scene_dir_path) if x.endswith('.h5py')]

        if n_ant is None:
            # 读第一个 h5 获取 n_ant
            if h5_files:
                import h5py
                with h5py.File(os.path.join(scene_dir_path, sorted(h5_files)[0])) as f_h5:
                    n_ant = int(f_h5['n_ant'][()])
            else:
                missing_nue.append(scene_id)
                continue

        if n_ue is None:
            n_ue = len(h5_files)

    n_ant = int(n_ant)
    n_ue  = int(n_ue)

    scene_meta[scene_id] = {
        'n_ant':     n_ant,
        'n_ue':      n_ue,
        'bandwidth': cfg.get('bandwidth'),
    }

    if n_ant not in groups:
        groups[n_ant] = []
    groups[n_ant].append((scene_id, n_ue))

if missing_config:
    print(f"  [warn] {len(missing_config)} 个场景缺少 config.json: {missing_config[:10]}...")
if missing_nue:
    print(f"  [warn] {len(missing_nue)} 个场景无法确定 n_ue: {missing_nue[:10]}...")

print(f"\n按 n_ant 分组结果:")
for n_ant in sorted(groups):
    scenes = groups[n_ant]
    total_ue = sum(n for _, n in scenes)
    print(f"  n_ant={n_ant:2d}: {len(scenes):3d} 场景  {total_ue:,} UE")

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: 保存场景元数据 / Step 2: save the scene metadata (scene_meta.json)
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(args.output_dir, exist_ok=True)
meta_path = os.path.join(args.output_dir, 'scene_meta.json')
with open(meta_path, 'w') as f:
    json.dump(scene_meta, f, indent=2)
print(f"\n场景元数据已保存: {meta_path}")

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: 为每个天线数分组生成预训练索引
# Step 3: build the pretraining index for each antenna-count group
#
# 拆分策略（预训练用，简化版）：
#   - 只使用 scene_id 0 ~ 99
#   - 每个场景内按 UE 编号拆分：
#       train: ue_idx 0 ~ n_ue*(1-val_ratio) - 1
#       val:   ue_idx n_ue*(1-val_ratio) ~ n_ue - 1
#   - 不做场景级随机，只做全局 UE 级 shuffle
#
# 索引格式: (N, 2) int32, 每行 [scene_id, ue_idx]
# ──────────────────────────────────────────────────────────────────────────────

VAL_RATIO = args.val_ratio   # 每场景取末尾 val_ratio 的 UE 作为验证集

for n_ant in sorted(groups):
    scenes = groups[n_ant]                     # [(scene_id, n_ue), ...]

    # 只保留前 100 个场景
    scenes_100 = [(s, n) for s, n in scenes if s < 100]

    train_rows = []
    val_rows   = []

    for sid, n_ue in scenes_100:
        n_val_ue   = max(1, int(n_ue * VAL_RATIO))
        n_train_ue = n_ue - n_val_ue
        # train: ue_idx 0 ~ n_train_ue-1
        for ue_idx in range(n_train_ue):
            train_rows.append([sid, ue_idx])
        # val: ue_idx n_train_ue ~ n_ue-1
        for ue_idx in range(n_train_ue, n_ue):
            val_rows.append([sid, ue_idx])

    def to_shuffled_array(rows, shuffle=True):
        if len(rows) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        arr = np.array(rows, dtype=np.int32)
        if shuffle:
            arr = arr[rng.permutation(len(arr))]
        return arr

    train_idx = to_shuffled_array(train_rows, shuffle=True)
    val_idx   = to_shuffled_array(val_rows,   shuffle=True)

    # 保存
    out_dir = os.path.join(args.output_dir, f'n{n_ant}')
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'train_shuffled.npy'),         train_idx)
    np.save(os.path.join(out_dir, 'in_scene_test_shuffled.npy'), val_idx)
    # gen_test 暂不生成（scene 100+ 暂无数据）

    print(f"\nn_ant={n_ant}  →  {out_dir}")
    print(f"  scenes used:   {len(scenes_100):3d}  (scene_id 0~99)")
    print(f"  train:         {len(train_idx):8,} UE  "
          f"(每场景前 {100-int(VAL_RATIO*100)}%)")
    print(f"  val:           {len(val_idx):8,} UE  "
          f"(每场景末 {int(VAL_RATIO*100)}%)")

# ──────────────────────────────────────────────────────────────────────────────
# Step 4: 打印可直接复制的预训练命令 / Step 4: print a ready-to-copy pretraining command
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("索引生成完成。训练命令示例：")
print()

configs_str = " \\\n        ".join(
    f'"{n_ant},128,{args.dataset_dir},{os.path.join(args.output_dir, f"n{n_ant}")}"'
    for n_ant in sorted(groups)
)
print(f"python ../fm/pretrain_dti.py \\")
print(f"    --configs \\")
print(f"        {configs_str} \\")
print(f"    --epochs 200 --batch_size 64 --patch_t 4 --patch_f 4")
print("="*60)
