# RA-LWLM：基于无线基础模型的检索增强上下文定位

本仓库只包含论文 **RA-LWLM: Retrieval-Augmented In-Context Localization with Wireless
Foundation Models** 的**最终方案**：数据集生成 → 基础模型自监督预训练 → RA-LWLM 训练 → 评测。



代码注释全部中英双语（中文说明 + English explanation）。

---

## 1. 方法一句话

冻结的无线基础模型（WFM）把 CSI 编码成特征 → 在该特征空间对**本场景的带标签参考库**做
top-K 检索 → 检索到的 (特征, 坐标) 作为**上下文示例**送进上下文学习模块（ICL），
由"加权质心 + 残差"读出坐标。上下文长度 k 本身依赖 query（近邻密集时小 k 更准，
NLOS 时大 k 更稳），所以用 **K-MoE**：k ∈ {3, 6, 9, 12, 15} 各训一个 ICL 专家，
训练时软混合、推理时逐 query 只跑选中的那一个专家。

最终坐标 = Σ_i softmax(selector logits)_i · pos_i，即按路由权重混合 5 个专家的预测。

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
├── jobs/                    SLURM 脚本（_common.sh 里是集群环境）
└── results/                 输出（CDF、json 表格、复杂度）
```

---

## 3. 环境准备

GPU 节点上：

```bash
module load GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb
source /nobackup/proj/disk/wireless_fm_data/shared/python_env/bin/activate
export PYTHONPATH=/nobackup/proj/disk/wireless_fm_data/personal/guangjin/py_extra:$PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE
```

这套设置已经写在 `jobs/_common.sh`，所有 job 脚本 `source` 它，不用重复写。
SLURM 会把脚本复制到自己的 spool 目录执行，所以 job 里不能靠 `$0` 找仓库；
仓库路径写在每个 job 脚本的那一行 `${RALWLM_REPO:-/home/guangjin/project_GP/RA-LWLM}` 里，
换位置时 `export RALWLM_REPO=<新路径>` 即可，不用改脚本。

> **注意**：登录节点和 GPU 节点架构不同，登录节点无法 import torch / Sionna，
> 所有训练、评测、数据生成都必须 `sbatch`。只有 `ra_lwlm/tools/smooth_cdf.py`
> 是纯 Python，可以在登录节点直接跑。

缺包时用 `pip install --no-deps <包名>` 装到 `py_extra`，**不要**直接 `pip install`
（会顺带拉一份几 GB 的 torch，覆盖 module 里的版本）。

---

## 4. 路径配置

所有默认路径集中在 `paths.py`，换机器只改这一个文件，或者设置同名环境变量：

| 变量 | 含义 | 默认值 |
|---|---|---|
| `RALWLM_DATA_ROOT` | 数据与检查点根目录 | `/nobackup/.../guangjin/sionna_data` |
| `RALWLM_DATASET_DIR` | 场景数据集 | `$DATA_ROOT/RAG_dataset` |
| `RALWLM_INDEX_DIR` | 预训练索引 | `$DATASET_DIR/RAG_index` |
| `RALWLM_CKPT_ROOT` | 检查点根目录 | `$DATA_ROOT/checkpoints` |
| `RALWLM_PRETRAIN_CKPT` | DTI 预训练权重 | `$CKPT_ROOT/pretrain_rag/pretrain_dti_latest.ckpt` |
| `RALWLM_RESULTS_DIR` | 结果输出 | `<仓库>/results` |

---

## 5. 从头到尾的执行流程

### 第 1 步：生成场景数据集（Sionna，GPU 节点）

```bash
cd jobs
# 训练/参考场景：无线配置随机（BS 高 15~20 m、方位角 25°~65°、带宽 5/10/20 MHz、天线 8/16/32）
sbatch job-gen-dataset.sh 0 20

# 未见场景测试集：固定配置（h=18 m, az=45°, 10 MHz, 32 天线），避免难度被随机配置搅乱
sbatch job-gen-dataset.sh 100 110 1
```

每个场景 10000 个 UE，产物：

```
$DATASET_DIR/scene_000/config.json      场景几何 + 无线配置（建筑框、BS 位置、带宽、天线数…）
$DATASET_DIR/scene_000/00000.h5py       单个 UE：channel (F,T) 复数、UElocation、distance、DoD_phi、bandwidth
...
```

UE 编号的约定（后面所有脚本都按这个划分，互不重叠）：

| UE 区间 | 用途 |
|---|---|
| `[0, n_train)` | 该场景的**带标签参考库**（检索库），论文主配置 n_train = 4000 |
| `[8000, 9000)` | 训练过程中的验证集 |
| `[9000, 10000)` | 最终测试 query |

场景编号的约定：`[0, 20)` 是训练时见过的场景（SS，seen scenes），
`[100, 110)` 是完全没见过的场景（US，unseen scenes，零样本评测）。

数据量参考：一个场景 10000 个 UE 大约几十 GB 量级，先跑 1~2 个场景确认再放开。

### 第 2 步：生成预训练索引

```bash
sbatch job-gen-index.sh
```

按天线数分组（`n8/ n16/ n32/`）输出 `(scene_id, ue_idx)` 列表 —— 天线数不同的样本
不能放进同一个 batch，所以必须分组。

### 第 3 步：基础模型 DTI 自监督预训练

```bash
sbatch job-pretrain-dti.sh            # embed_dim=256 depth=4 heads=4，200 epoch
```

任务是"频域 CSI → 时延-角度域重构"，**完全不用位置标签**；8/16/32 天线的数据一起训练，
得到一套共享权重。产物：

```
$CKPT_ROOT/pretrain_rag/pretrain_dti_latest.ckpt      ← RA-LWLM 冻结加载它
```

这一步最耗时（约 1~2 天）。如果已经有预训练权重，可以直接跳到第 4 步。

### 第 4 步：训练 RA-LWLM

```bash
sbatch job-train-ra-lwlm.sh 0 20 4000       # 20 个训练场景，每场景 4000 个参考样本
```

内部是两个阶段：

* **阶段 1**：对 k ∈ {3, 6, 9, 12, 15} 各自独立预训练一个 ICL 专家（固定上下文长度，
  100 epoch）。编码器全程冻结。
* **阶段 2**：把 5 个专家装进 K-MoE，冻结专家、只训 selector（50 epoch）。
  训练与推理都用软混合：坐标 = Σ_i π_i · pos_i。

关键实现：所有检索（top-K 索引与距离）在启动时按场景**一次性**算好缓存起来，
训练中只做查表，所以 batch 里没有编码器前向、也没有 cdist。训练 query 就是参考库本身
（leave-one-out，检索时排除自己）。

产物：

```
$CKPT_ROOT/ra_lwlm_kmoe_000_020/kmoe_n4000_K20_lb000_frz_seed42_best.ckpt
```

常用可调参数（直接追加在 sbatch 命令后面即可透传）：

```bash
sbatch job-train-ra-lwlm.sh 0 20 200                       # 稀疏参考库（N=200）
sbatch job-train-ra-lwlm.sh 0 20 4000 --skip_pretrain      # 复用已有的阶段 1 专家
sbatch job-train-ra-lwlm.sh 0 20 4000 --lb_lambda 0.01     # 打开负载均衡辅助损失
```

### 第 5 步：评测

```bash
CKPT=$CKPT_ROOT/ra_lwlm_kmoe_000_020/kmoe_n4000_K20_lb000_frz_seed42_best.ckpt
sbatch job-eval-ra-lwlm.sh $CKPT 4000
```

一次跑完给出：

* **SS（见过的场景）/ US（未见场景，零样本）** 两个划分；
* 每个划分再按 **ALL / LOS / NLOS** 报均值、中值、90% 分位，并打印有效上下文长度
  Σ π_i k_i；
* **CDF 文件**：`results/ra_lwlm_n4000/{ss,us}/ra_lwlm{,_los,_nlos}_cdf.txt`，
  两列 `误差(m)  累积概率`；
* **json 汇总**：`results/ra_lwlm_n4000/ra_lwlm_n4000.json`。

只评未见场景（更快）加 `--skip_ss`；换 query 区间用 `--test_ue_start/--test_ue_end`。

### 第 6 步：复杂度

```bash
sbatch job-bench-complexity.sh $CKPT 4000
```

给出参数量（冻结编码器 / 单专家 / 5 个专家 / selector）、每场景检索库存储（MB）、
检索耗时、端到端推理延迟（batch 1 与 batch 64）、逐部件拆解（编码器 / 检索 / selector /
每个专家），以及新场景上线成本（只需把参考库编码一遍，无需训练）。
输出 `results/complexity/complexity.{json,md}`。

### 第 7 步（可选）：跨数据集验证

```bash
cd $REPO/data_gen
python gen_deepmimo_scenes.py --bs 1 --mode corner --az 45      # 需在 GPU 节点，且装好 DeepMIMOv3
```

把 DeepMIMO O1_3p5 转成同一套格式（坐标系、阵列朝向都对齐到 Sionna 的约定），
然后用 `--gen_test_start/--gen_test_end` 指向新场景号直接零样本评测。

---

## 6. 画图

评测脚本给出的 CDF 是经验 CDF（1000 个点，横轴是排序后的误差）。要画得平滑一些：

```bash
python ra_lwlm/tools/smooth_cdf.py results/ra_lwlm_n4000/us/ra_lwlm_cdf.txt --n_grid 1000
# → results/ra_lwlm_n4000/us/ra_lwlm_cdf_smooth.txt
```

高斯核平滑（在 0 处反射，保证 F(0)=0、单调、末点为 1），默认按**纵轴等间隔**采样：
p = 0.001, 0.002, …, 1.000，横坐标由平滑 CDF 反解得到。`--bw` 是带宽倍数（>1 更平滑），
`--x_uniform` 切回横轴等间隔。这个脚本是纯 Python，登录节点可直接运行。

---

## 7. 参考结果（论文主配置：20 个训练场景，每场景 4000 个参考样本）

| 划分 | 均值 (m) | 中值 (m) | 90% (m) |
|---|---|---|---|
| SS（见过的场景） | 0.75 | 0.45 | 1.52 |
| US（未见场景，零样本） | 0.89 | 0.53 | 1.83 |

本仓库用已有的 checkpoint 复核过 US 一栏（10 个未见场景 × 1000 个 query = 10000 个样本）：

```
  group  |    mean  median     p90
  ALL    |   0.887   0.525   1.829
  LOS    |   0.552   0.372   1.163
  NLOS   |   1.283   0.826   2.538
```

NLOS 的误差约是 LOS 的 2.3 倍。复现时如果数字差得多，先按下面的清单查。

---

## 8. 常见问题

* **登录节点 import torch / sionna 报错**：架构不同，必须 `sbatch` 到 GPU 节点。
* **job 显示 COMPLETED 但没有结果**：job 脚本里要把 python 的退出码传出去
  （本仓库的脚本都写了 `RC=$?; exit $RC`）。
* **`#SBATCH` 不生效、日志跑到别的地方**：所有 `#SBATCH` 行必须在第一条可执行命令
  **之前**，写在 `module load` 后面会被完全忽略。
* **加载 checkpoint 时权重对不上**：`EnvPara` 必须与预训练时完全一致
  （`patch_t/patch_f`、`embed_dim`、`depth`、`num_heads`、`input_tdim`=天线数）。
* **跨场景误差异常大（几十米）**：先查 `config.json` 的配置向量是否落在训练时的归一化
  范围内。配置向量是 `[方位角(rad), (带宽−5MHz)/15MHz, (天线数−8)/24, (BS高度−15)/5]`，
  例如把 6 m 高的基站直接喂进去会得到 −1.8，远超训练范围，绝对回归类的读出会直接失效。
* **NLOS 组为空导致分位数报错**：全 LoS 的数据集（如 DeepMIMO）会出现这种情况，
  评测脚本里已经做了空组保护，返回 NaN 而不是崩掉。
* **GPU 显存不够**：调小 `--batch_size`；检索缓存按场景逐个构建，`--n_train` 越大
  启动阶段越久（4000 个参考样本的一遍编码大约几十秒）。
