# RA-LWLM：基于无线基础模型的检索增强上下文定位

本仓库只包含论文 **RA-LWLM: Retrieval-Augmented In-Context Localization with Wireless
Foundation Models** 的**最终方案**：数据集生成 → 基础模型自监督预训练 → RA-LWLM 训练 → 评测。

不在这里的东西（都留在原实验仓库）：
* 所有对比基线：ResNet、LWLM-DTI 端到端微调、KNN / 加权 KNN、channel charting、
  GAT / GCN、MetaLoc、OMP 等；
* 所有中间方案与消融变体：top-1 / top-2 / Gumbel 硬路由、均匀路由、单专家、
  去掉加权质心 / 残差读出 / 配置条件等，以及训练期用过的 CCCP 自监督辅助头。

代码注释全部中英双语（中文说明 + English explanation）。
**所有步骤在一张普通 GPU 上就能跑**，不依赖任何集群设施；`jobs/` 里另附 SLURM 脚本。

---

## 1. 方法一句话

冻结的无线基础模型（WFM）把 CSI 编码成特征 → 在该特征空间对**本场景的带标签参考库**做
top-K 检索 → 检索到的 (特征, 坐标) 作为**上下文示例**送进上下文学习模块（ICL），
由"加权质心 + 残差"读出坐标。上下文长度 k 本身依赖 query（近邻密集时小 k 更准，
NLOS 时大 k 更稳），所以用 **K-MoE**：k ∈ {3, 6, 9, 12, 15} 各训一个 ICL 专家，
由 selector 给出混合权重。最终坐标 = Σ_i softmax(selector logits)_i · pos_i。

进入新场景时**不需要任何训练**，只要把该场景的参考库编码一遍即可。

```
              ┌──────────────── 每个场景一次（无需训练）────────────────┐
  参考库 CSI ─┤ 冻结 WFM 编码器 → LST 特征 + patch 均值 → 存成检索库      │
              └────────────────────────────────────────────────────────┘
                                        │
  query CSI → 冻结 WFM 编码器 → 特征 ──► top-K 检索（L2）──► K 个 (特征, 坐标)
                                        │                         │
                                        └──► selector ──► 5 个专家的混合权重 π
                                                                              │
                   每个专家：质心(检索权重加权) + 残差 ──► 坐标，再按 π 混合 ──► 坐标
```

---

## 2. 目录结构

```
RA-LWLM/
├── paths.py                 全仓库唯一的路径配置（换机器只改这里）
├── requirements.txt
├── data_gen/                数据集生成
│   ├── scene_builder.py         随机 32×32 m 场景（地面 + 建筑 + 角落基站）
│   ├── gen_dataset.py           Sionna 光线追踪算信道，写 h5py + config.json
│   ├── gen_pretrain_index.py    按天线数分组生成预训练索引
│   ├── check_los_nlos.py        LOS / NLOS 标注与统计
│   └── gen_deepmimo_scenes.py   （可选）把 DeepMIMO O1_3p5 转成同一格式做跨数据集验证
├── fm/                      无线基础模型与自监督预训练
│   ├── model.py                 wireless_loc_fm：Transformer 编码器 + DTI 解码器
│   ├── train_model.py           Lightning 包装器 Wrapper
│   ├── dataload_sionna_v2.py    数据读取与预处理（频域 → 时延-角度域）
│   └── pretrain_dti.py          DTI 自监督预训练入口
├── ra_lwlm/                 RA-LWLM 本体
│   ├── modules.py               PosEncoder / CfgEncoder / ReasoningTransformer
│   ├── ra_lwlm_icl.py           单个 ICL 专家（加权质心 + 残差读出）
│   ├── selector.py              K-MoE 路由器（selector）
│   ├── ra_lwlm_kmoe.py          K-MoE：5 个专家 + 软路由 + 负载均衡损失
│   ├── train_ra_lwlm.py         两阶段训练
│   ├── eval_ra_lwlm.py          评测：SS / US × LOS / NLOS + CDF
│   ├── bench_complexity.py      参数量 / 存储 / 检索耗时 / 推理延迟
│   └── tools/
│       ├── npy2cdf.py           逐样本误差 .npy → CDF 文件
│       └── smooth_cdf.py        CDF 高斯核平滑（纵轴等间隔）
├── jobs/                    （可选）SLURM 脚本
└── results/                 输出（CDF、json 表格、复杂度）
```

---

## 3. 环境

### 3.1 训练与评测（只需要 PyTorch）

```bash
conda create -n ralwlm python=3.11 -y && conda activate ralwlm
# 按自己的 CUDA 版本装 torch，见 https://pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install pytorch-lightning numpy h5py scipy torchvision tensorboardX
```

用 venv 也一样：`python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`。

自检：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
cd ra_lwlm && python train_ra_lwlm.py --help && python eval_ra_lwlm.py --help
```

CPU 也能跑通流程（脚本会自动退到 CPU），但会慢几十倍，只适合调试。

### 3.2 数据集生成（只有这一步需要 Sionna）

```bash
pip install sionna matplotlib          # Sionna ≥ 1.0，需要 GPU（Mitsuba/Dr.Jit）
```

如果你已经有生成好的数据集（`scene_XXX/` 目录），**完全不用装 Sionna**，直接从第 5 步的
预训练开始。想做跨数据集验证再加 `pip install DeepMIMOv3`。

### 3.3 资源需求（实测）

| 项目 | 规模 |
|---|---|
| 单个场景磁盘占用（10000 个 UE） | 8 天线 ≈ 157 MB，16 天线 ≈ 235 MB，32 天线 ≈ 392 MB（每个 UE 一个 40 KB 的 h5） |
| 论文配置总量（20 训练 + 10 测试场景） | ≈ 8 GB |
| 显存：DTI 预训练 | batch 64 约 8–10 GB；显存小就降 `--batch_size` |
| 显存：RA-LWLM 训练 | batch 32 约 6–8 GB（编码器冻结且只前向） |
| 显存：评测 | `--batch_size 128` 约 4 GB |
| 时间（单卡）| 数据生成每场景 1–3 h；DTI 预训练 200 epoch 约 1–2 天；RA-LWLM 阶段 1 (5×100 ep) + 阶段 2 (50 ep) 约 10–20 h；评测 10 个场景约 2 min |

一张 12 GB 以上的卡（3080Ti / 4070Ti / A5000 / L40 / A100 等）就够跑完整流程；
显存更小就按第 8 节的办法缩规模。

---

## 4. 路径配置

所有路径集中在 `paths.py`，**默认落在 `<仓库>/data` 下面，本地开箱即用**：

```
<repo>/data/RAG_dataset/            场景数据集
<repo>/data/RAG_dataset/RAG_index/  预训练索引
<repo>/data/checkpoints/            检查点
<repo>/results/                     结果（CDF、json）
```

想放到别的盘（数据量大，通常要放大盘）：

```bash
export RALWLM_DATA_ROOT=/mnt/data/ralwlm      # 数据集 + 索引 + 检查点都跟着走
```

更细的覆盖（都可单独设，或直接改 `paths.py`）：`RALWLM_DATASET_DIR`、`RALWLM_INDEX_DIR`、
`RALWLM_CKPT_ROOT`、`RALWLM_PRETRAIN_CKPT`、`RALWLM_RESULTS_DIR`。
每个脚本也都能用命令行参数临时覆盖：`--dataset_dir`、`--pretrain_ckpt`、`--ckpt_dir`、`--out_dir`。

---

## 5. 从头到尾的执行流程（本地单卡）

下面每一步都给出直接 `python` 的命令；集群用户把对应命令换成 `jobs/` 里的 `sbatch`
即可（见第 7 节）。建议先用第 8 节的"小规模试跑"把整条链路走通，再放大到论文规模。

### 第 1 步：生成场景数据集（需要 Sionna）

```bash
cd data_gen
# 训练/参考场景：BS 高度 15~20 m、方位角 25°~65°、带宽 5/10/20 MHz、天线 8/16/32 全部随机
python gen_dataset.py --scene_start 0 --scene_end 20 --n_ue 10000 --seed 42

# 未见场景测试集：只固定基站几何（h=18 m, az=45°），带宽与天线数仍然随机
python gen_dataset.py --scene_start 100 --scene_end 110 --n_ue 10000 --seed 42 --fixed_cfg 1
```

> 几何固定让各个测试场景之间可比，带宽/天线数保持变化才能检验对未见无线配置的泛化。
> 论文用的 10 个未见场景就是这样：h = 18 m、az = 45° 不变，天线数 8/16/32、带宽 5/10/20 MHz 混合。

想并行就把场景区间切开同时跑几个进程（彼此独立，互不干扰）：

```bash
python gen_dataset.py --scene_start 0  --scene_end 5  & \
python gen_dataset.py --scene_start 5  --scene_end 10 & wait
```

产物：

```
$DATASET_DIR/scene_000/config.json    场景几何 + 无线配置（建筑框、BS 位置、带宽、天线数…）
$DATASET_DIR/scene_000/00000.h5py     单个 UE：channel (F,T) 复数、UElocation、distance、DoD_phi、bandwidth
...
```

UE 编号约定（后面所有脚本都按这个划分，互不重叠）：

| UE 区间 | 用途 |
|---|---|
| `[0, n_train)` | 该场景的**带标签参考库**（检索库），论文主配置 n_train = 4000 |
| `[8000, 9000)` | 训练过程中的验证集 |
| `[9000, 10000)` | 最终测试 query |

场景编号约定：`[0, 20)` 是训练时见过的场景（SS，seen scenes），
`[100, 110)` 是完全没见过的场景（US，unseen scenes，零样本评测）。

可选：统计一下 LOS / NLOS 比例（评测时会自动按同样的判据分组）

```bash
python check_los_nlos.py
```

### 第 2 步：生成预训练索引

```bash
cd data_gen && python gen_pretrain_index.py
```

按天线数分组（`n8/ n16/ n32/`）输出 `(scene_id, ue_idx)` 列表 —— 天线数不同的样本
不能放进同一个 batch，所以必须分组。脚本结束时会打印一条可直接复制的预训练命令。

### 第 3 步：基础模型 DTI 自监督预训练

```bash
cd fm
DS=$(python -c "import sys;sys.path.insert(0,'..');import paths;print(paths.DATASET_DIR)")
IX=$(python -c "import sys;sys.path.insert(0,'..');import paths;print(paths.INDEX_DIR)")

python pretrain_dti.py \
    --configs "8,128,$DS,$IX/n8" "16,128,$DS,$IX/n16" "32,128,$DS,$IX/n32" \
    --epochs 200 --batch_size 64 --lr 1e-4 \
    --embed_dim 256 --depth 4 --num_heads 4 \
    --patch_t 4 --patch_f 4 --num_workers 8 --max_val_samples 60000 --seed 42
```

任务是"频域 CSI → 时延-角度域重构"，**完全不用位置标签**；8/16/32 天线的数据一起训练，
得到一套共享权重。产物：

```
$CKPT_ROOT/pretrain_rag/pretrain_dti_latest.ckpt      ← RA-LWLM 冻结加载它（约 21 MB）
```

这一步最耗时。如果只有一种天线数，`--configs` 就只写那一项；也可以用单数据集模式
`--n_antennas 32 --n_subcarriers 128 --dataset_dir ... --index_dir ...`。

### 第 4 步：训练 RA-LWLM

```bash
cd ra_lwlm
python train_ra_lwlm.py \
    --scene_start 0 --scene_end 20 --val_scene_start 0 --val_scene_end 20 \
    --gen_test_start 100 --gen_test_end 110 \
    --n_train 4000 \
    --pretrain_epochs 100 --joint_epochs 50 --freeze_icl \
    --lb_lambda 0.0 --gate_temp 1.0 \
    --K_max 20 --ra_num_layers 2 --token_dim 256 --pos_scale 32.0 --dropout 0.1 \
    --lr_icl 1e-4 --lr_kmoe 3e-5 --batch_size 32 --num_workers 4 --seed 42
```

内部是两个阶段：

* **阶段 1**：对 k ∈ {3, 6, 9, 12, 15} 各自独立预训练一个 ICL 专家（固定上下文长度，
  100 epoch）。编码器全程冻结。
* **阶段 2**：把 5 个专家装进 K-MoE，冻结专家、只训 selector（50 epoch），
  训练与推理都用软混合：坐标 = Σ_i π_i · pos_i。

关键实现：所有检索（top-K 索引与距离）在启动时按场景**一次性**算好缓存起来，
训练中只做查表，所以 batch 里没有编码器前向、也没有 cdist。训练 query 就是参考库本身
（leave-one-out，检索时排除自己）。

产物：

```
$CKPT_ROOT/ra_lwlm_kmoe_000_020/kmoe_n4000_K20_lb000_frz_seed42_best.ckpt
```

常用变体：

```bash
--n_train 200            # 稀疏参考库
--skip_pretrain          # 复用已有的阶段 1 专家（配合相同的 --ckpt_dir）
--lb_lambda 0.01         # 打开负载均衡辅助损失
--batch_size 16          # 显存不足时
```

### 第 5 步：评测

```bash
cd ra_lwlm
CKPT=$(python -c "import sys,os;sys.path.insert(0,'..');import paths;print(os.path.join(paths.CKPT_ROOT,'ra_lwlm_kmoe_000_020','kmoe_n4000_K20_lb000_frz_seed42_best.ckpt'))")

python eval_ra_lwlm.py --ra_ckpt $CKPT --n_train 4000 \
    --val_scene_start 0 --val_scene_end 20 \
    --gen_test_start 100 --gen_test_end 110 \
    --test_ue_start 9000 --test_ue_end 10000 \
    --K_max 20 --batch_size 128 --num_workers 4 --seed 42 \
    --out_dir ../results/ra_lwlm_n4000
```

一次跑完给出：

* **SS（见过的场景）/ US（未见场景，零样本）** 两个划分；
* 每个划分再按 **ALL / LOS / NLOS** 报均值、中值、90% 分位，并打印有效上下文长度 Σ π_i k_i；
* **CDF 文件**：`results/ra_lwlm_n4000/{ss,us}/ra_lwlm{,_los,_nlos}_cdf.txt`，两列 `误差(m)  累积概率`；
* **json 汇总**：`results/ra_lwlm_n4000/ra_lwlm_n4000.json`。

只评未见场景（更快）加 `--skip_ss`；换 query 区间用 `--test_ue_start/--test_ue_end`。

### 第 6 步：复杂度

```bash
cd ra_lwlm && python bench_complexity.py --ra_ckpt $CKPT --scene 0 --n_train 4000 --K 20 --reps 50
```

给出参数量（冻结编码器 / 单专家 / 5 个专家 / selector）、每场景检索库存储（MB）、
检索耗时、端到端推理延迟（batch 1 与 batch 64）、逐部件拆解（编码器 / 检索 / selector /
每个专家），以及新场景上线成本。输出 `results/complexity/complexity.{json,md}`。
延迟数字依赖显卡型号，换卡后需要重测。

### 第 7 步（可选）：跨数据集验证

```bash
pip install DeepMIMOv3
cd data_gen && python gen_deepmimo_scenes.py --bs 1 --mode corner --az 45 \
    --scenario_dir /path/to/O1_3p5
```

把 DeepMIMO O1_3p5 转成同一套格式（坐标系、阵列朝向都对齐到 Sionna 的约定），然后
`--gen_test_start/--gen_test_end` 指向新场景号直接零样本评测。

---


## 6. 小规模试跑（先把链路走通）

不想等一整天、或者显存/磁盘有限，用下面这套参数半天内能跑完一遍完整流程：

```bash
# 2 个训练场景 + 1 个测试场景，每场景 2000 个 UE（磁盘 < 300 MB）
cd data_gen
python gen_dataset.py --scene_start 0 --scene_end 2 --n_ue 2000 --seed 42
python gen_dataset.py --scene_start 100 --scene_end 101 --n_ue 2000 --seed 42 --fixed_cfg 1
python gen_pretrain_index.py

# 预训练 20 epoch 先拿到一个能用的编码器。
# --configs 要和实际存在的天线数分组对上：gen_pretrain_index.py 结束时会打印现成的命令，
# 也可以 ls $IX 看有哪些 n8/n16/n32 目录（2 个场景通常只会出现其中一两个）。
cd ../fm
DS=$(python -c "import sys;sys.path.insert(0,'..');import paths;print(paths.DATASET_DIR)")
IX=$(python -c "import sys;sys.path.insert(0,'..');import paths;print(paths.INDEX_DIR)")
ls $IX                                    # 例如只有 n32/，那 --configs 就只写 32 那一项
python pretrain_dti.py --configs "32,128,$DS,$IX/n32" --epochs 20 --batch_size 32 \
       --max_val_samples 2000 --num_workers 4

# RA-LWLM：参考库 500，阶段 1/2 各 10 epoch
cd ../ra_lwlm
python train_ra_lwlm.py --scene_start 0 --scene_end 2 --val_scene_start 0 --val_scene_end 2 \
       --gen_test_start 100 --gen_test_end 101 --n_train 500 \
       --pretrain_epochs 10 --joint_epochs 10 --freeze_icl \
       --val_ue_start 1500 --val_ue_end 1750 --test_ue_start 1750 --test_ue_end 2000 \
       --batch_size 16 --num_workers 2
```

注意 UE 区间要跟着 `--n_ue` 缩：库是 `[0, n_train)`，验证与测试区间必须在库之外且
不超过 `n_ue`（上面用了 1500–1750 和 1750–2000）。这套小配置的误差会明显比论文差，
只用于验证流程能跑通。

---
