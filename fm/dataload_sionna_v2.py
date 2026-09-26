"""
dataload_sionna_v2.py
=====================================================================
数据读取与预处理 / Data loading and preprocessing.

中文
----
数据布局：{dataset_dir}/scene_{scene_id:03d}/{ue_idx:05d}.h5py
  * channel 存成 (F, T)，读入后转置成 (T, F) = 天线 × 子载波
  * UElocation / distance / DoD_phi 存的是已归一化的值（坐标 = 真实米数 / 10）
  * 单基站（BS_Num = 1）

process_channel() 给出模型输入：
  aa  归一化 CSI 的 (角度, 时延) 域幅度图，shape (2, T, F) → 编码器输入
  ri  频域实/虚部
  da  时延-角度域实/虚部（DTI 预训练的重建目标）

EN
--
Layout: {dataset_dir}/scene_{scene_id:03d}/{ue_idx:05d}.h5py
  * `channel` is stored as (F, T) and transposed to (T, F) = antennas × subcarriers
  * UElocation / distance / DoD_phi are stored pre-normalised (position = metres / 10)
  * single base station (BS_Num = 1)

process_channel() returns the model inputs:
  aa  (angle, delay)-domain magnitude map of the normalised CSI, (2, T, F) → encoder
  ri  real/imaginary parts in the frequency domain
  da  real/imaginary parts in the delay-angle domain (the DTI reconstruction target)
"""

import numpy as np
import random
import h5py
import sys
import os

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(__file__))

# 默认路径来自仓库根目录的 paths.py / default paths come from paths.py at the repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATASET_DIR as _DEFAULT_DATASET_DIR, INDEX_DIR as _DEFAULT_INDEX_DIR


# ─────────────────────────────────────────────
# 工具函数（与原始 dataload.py 保持一致）
# ─────────────────────────────────────────────

def channel_normalization(H):
    """按总功率归一化，消除路径损耗带来的幅度差异。
    Normalise by total power so that path-loss amplitude differences are removed.
    H: complex array [T, F], T = antennas, F = subcarriers."""
    P_current = np.sum(np.abs(H) ** 2)
    alpha = np.sqrt(H.shape[1] / P_current)   # H.shape[1] = F (子载波数)
    return H * alpha


def freq_to_delay_angle_domain(H_freq):
    """频域 → 时延-角度域（天线轴 IFFT，子载波轴 FFT）。
    Frequency domain → delay-angle domain (IFFT along antennas, FFT along subcarriers).
    H_freq: [2, T, F] real/imag → complex delay-angle domain."""
    H_complex = H_freq[0] + 1j * H_freq[1]
    H_delay   = np.fft.ifft(H_complex, axis=0)
    H_da      = np.fft.fft(H_delay, axis=1) / np.sqrt(H_delay.shape[0])
    return H_da


def load_h5_sample(scene_id, ue_idx, dataset_dir=None):
    """读取单个 UE 的 H5 文件，返回复数信道和标签。

    Returns:
        channel:   complex (T, F) = (n_ant, n_subcarrier)
        ue_loc:    (2,) already /10
        bs_loc:    (2,) raw metres
        distance:  float, already /100
        dod_phi:   float, already /100
        bandwidth: float MHz（若 h5 中无此字段则默认 5e6）
    """
    root = dataset_dir if dataset_dir else _DEFAULT_DATASET_DIR
    path = os.path.join(root,
                        f"scene_{scene_id:03d}",
                        f"{ue_idx:05d}.h5py")
    with h5py.File(path, 'r') as f:
        cr = f["channel_real"][:]          # (F, T): subcarrier × antenna
        ci = f["channel_imag"][:]          # (F, T)
        ue_loc    = f["UElocation"][:]     # already /10 → shape (2,)
        bs_loc    = f["BSlocation"][:]     # raw (x,y) metres → shape (2,)
        distance  = float(f["distance"][()])   # already /100
        dod_phi   = float(f["DoD_phi"][()])    # already /100
        # bandwidth 字段存在于 RAG_dataset，旧数据集无此字段 → 默认 5e6
        bandwidth = float(f["bandwidth"][()]) if "bandwidth" in f else 5e6
    # 转置为 (T, F) = (antenna, subcarrier)
    channel = (cr + 1j * ci).T            # (T, F)
    return channel, ue_loc, bs_loc, distance, dod_phi, bandwidth


def process_channel(channel, input_fmap=2):
    """channel: complex (T, F) → aa [2,T,F], ri [2,T,F], delay-angle-ri [2,T,F]"""
    T, F = channel.shape
    channel = channel_normalization(channel)

    aa = np.zeros([input_fmap, T, F], dtype=np.float32)
    aa[0] = np.abs(channel)
    aa[1] = np.angle(channel)

    ri = np.zeros([input_fmap, T, F], dtype=np.float32)
    ri[0] = np.real(channel)
    ri[1] = np.imag(channel)

    da_complex = freq_to_delay_angle_domain(ri)
    da_ri = np.zeros([input_fmap, T, F], dtype=np.float32)
    da_ri[0] = np.real(da_complex)
    da_ri[1] = np.imag(da_complex)

    return aa, ri, da_ri


# ─────────────────────────────────────────────
# Dataset 类
# ─────────────────────────────────────────────

class SionnaDataset(Dataset):
    """
    用于 LWLM-RAG fine-tune 和 RAG 训练的数据集。

    参数：
        split:       'train' | 'in_scene_test' | 'gen_test'
        max_samples: 限制样本数（None=全部）
        input_fmap:  2（幅度+相位 或 实+虚）
        input_tdim:  天线数（默认 16）；加载样本时会校验实际 shape
        input_fdim:  子载波数（默认 128）；加载样本时会校验实际 shape
        dataset_dir: H5 文件根目录（空字符串→使用模块默认值）
        index_dir:   索引 npy 目录  （空字符串→使用模块默认值）
    """

    def __init__(self, split='train', max_samples=None,
                 input_fmap=2, input_tdim=16, input_fdim=128,
                 dataset_dir='', index_dir=''):
        self.input_fmap  = input_fmap
        self.input_tdim  = input_tdim
        self.input_fdim  = input_fdim
        self.dataset_dir = dataset_dir or _DEFAULT_DATASET_DIR
        self.index_dir   = index_dir   or _DEFAULT_INDEX_DIR

        # Use shuffled indices for train/in_scene_test so that max_samples
        # draws uniformly across all 80 scenes (not just scene 0).
        npy_map = {
            'train':          'train_shuffled.npy',
            'in_scene_test':  'in_scene_test_shuffled.npy',
            'gen_test':       'gen_test.npy',
        }
        assert split in npy_map, f"split must be one of {list(npy_map.keys())}"
        idx_path = os.path.join(self.index_dir, npy_map[split])
        self.index = np.load(idx_path)          # (N, 2): [scene_id, ue_idx]

        if max_samples is not None:
            self.index = self.index[:max_samples]

        print(f"[SionnaDataset] split={split}  samples={len(self.index)}  "
              f"antennas={input_tdim}  subcarriers={input_fdim}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        scene_id, ue_idx = int(self.index[idx, 0]), int(self.index[idx, 1])

        channel, ue_loc, bs_loc, distance, dod_phi, bandwidth = load_h5_sample(
            scene_id, ue_idx, dataset_dir=self.dataset_dir
        )

        # Validate shape matches configured dimensions
        T, F = channel.shape
        if T != self.input_tdim or F != self.input_fdim:
            raise ValueError(
                f"Channel shape mismatch at scene {scene_id} / ue {ue_idx}: "
                f"loaded ({T}, {F}) but expected ({self.input_tdim}, {self.input_fdim}). "
                f"Check --n_antennas / --n_subcarriers match the dataset."
            )

        aa, ri, da_ri = process_channel(channel, self.input_fmap)

        # 扩展到 [BS_Num=1, ...] 以兼容原始 LWLM 接口
        ch_aa   = aa  [None, ...]    # (1, 2, T, F)
        ch_ri   = ri  [None, ...]    # (1, 2, T, F)
        ch_da   = da_ri[None, ...]   # (1, 2, T, F)

        # 标签：UEloc已/10, distance已/100, phi已/100
        ue_label = np.zeros(4, dtype=np.float32)
        ue_label[:2] = ue_loc          # already /10
        ue_label[2]  = distance        # already /100
        ue_label[3]  = dod_phi         # already /100
        ue_label_all = ue_label[None, :]   # (1, 4)

        # BS 配置：[bs_x/100, bs_y/100, bw/10]
        # bandwidth 从 h5 读取（RAG_dataset 可变；旧数据集默认 5e6）
        bs_conf = np.zeros(3, dtype=np.float32)
        bs_conf[:2] = bs_loc / 100.0          # BS 位置归一化
        bs_conf[2]  = bandwidth / 1e7         # MHz → /10 归一化（5MHz→0.5, 10MHz→1.0, 20MHz→2.0）
        bs_conf_all = bs_conf[None, :]        # (1, 3)

        return (ch_aa.astype(np.float32),
                ch_ri.astype(np.float32),
                ch_da.astype(np.float32),
                ue_label_all.astype(np.float32),
                bs_conf_all.astype(np.float32),
                np.array([scene_id, ue_idx], dtype=np.int32))


# ─────────────────────────────────────────────
# 快速验证
# ─────────────────────────────────────────────
if __name__ == "__main__":
    ds = SionnaDataset(split='train', max_samples=100)
    aa, ri, da, ue, bsc, sid = ds[0]
    print("ch_aa shape:", aa.shape)      # (1, 2, 16, 128)
    print("ch_ri shape:", ri.shape)      # (1, 2, 16, 128)
    print("ue_label:   ", ue)            # [[x/10, y/10, d/100, phi/100]]
    print("bs_conf:    ", bsc)           # [[0, 0, 1.0]]
    print("scene/ue:   ", sid)

    loader = DataLoader(ds, batch_size=32, num_workers=4, shuffle=True)
    batch = next(iter(loader))
    print("batch ch_aa:", batch[0].shape)   # (32, 1, 2, 16, 128)
    print("batch ue:   ", batch[3].shape)   # (32, 1, 4)
