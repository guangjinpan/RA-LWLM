"""
gen_dataset.py
=====================================================================
用 Sionna 光线追踪生成定位数据集 / Generate the localisation dataset with Sionna RT.

中文
----
每个场景：随机生成 32×32 m 的建筑布局 + 一个角落基站，再批量算 n_ue 个 UE 的信道。
场景之间的无线配置是随机的（BS 高度 15~20 m、方位角 25°~65°、带宽 5/10/20 MHz、
天线数 8/16/32），这是"跨场景"实验的基础。

  --fixed_cfg 1  只固定基站几何（h=18 m, az=45°），带宽与天线数仍然随机采样。
                 未见场景测试集用它：几何固定让各测试场景之间可比，而带宽/天线数
                 保持变化，才能检验模型对未见无线配置的泛化。

输出 / output: {output_dir}/scene_{scene_id:03d}/{ue_idx:05d}.h5py
               {output_dir}/scene_{scene_id:03d}/config.json

EN
--
Per scene: sample a random 32×32 m building layout with one corner BS, then
compute the channels of n_ue UEs in batches. The radio configuration is randomised
across scenes (BS height 15–20 m, azimuth 25°–65°, bandwidth 5/10/20 MHz, 8/16/32
antennas), which is what makes the cross-scene experiments meaningful.

  --fixed_cfg 1  freeze only the BS geometry (h=18 m, az=45°) while still sampling the
                 bandwidth and the antenna count. The unseen-scene test set uses this:
                 fixed geometry keeps the test scenes comparable, while the varying
                 radio configuration is what probes generalisation to unseen setups.

用法 / usage:
    python gen_dataset.py --scene_start 0   --scene_end 20                 # 训练场景 / training scenes
    python gen_dataset.py --scene_start 100 --scene_end 110 --fixed_cfg 1  # 未见测试场景 / unseen test scenes
"""
import argparse
import os
import sys
import json
import time
import numpy as np
import h5py

# ── 命令行参数（先解析，避免 Sionna 的慢导入拖延报错）
# ── Parse the arguments first so a bad argument fails before the slow Sionna import
parser = argparse.ArgumentParser()
parser.add_argument('--scene_start', type=int, default=0,   help='起始场景 ID（含）')
parser.add_argument('--scene_end',   type=int, default=10,  help='结束场景 ID（不含）')
parser.add_argument('--n_ue',        type=int, default=10000, help='每个场景的目标 UE 数')
parser.add_argument('--seed',        type=int, default=42,  help='全局随机种子基准')
parser.add_argument('--batch_size',  type=int, default=50,  help='每次 PathSolver 调用的 UE 批量大小')
parser.add_argument('--output_dir',  type=str, default=None,
                    help='输出目录，默认取 paths.py 里的 DATASET_DIR / output directory, '
                         'defaults to DATASET_DIR from paths.py')
parser.add_argument('--synthetic_array', type=int, default=0,
                    help='PathSolver synthetic_array 参数（0=False, 1=True）/ PathSolver synthetic_array')
parser.add_argument('--fixed_cfg', type=int, default=0,
                    help='1=固定基站几何 h=18 m / az=45°（带宽与天线数仍随机），未见场景测试集用 / '
                         '1 = freeze the BS geometry to h=18 m, az=45 deg (bandwidth and antenna '
                         'count stay random); used for the unseen test set')
args = parser.parse_args()

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
from paths import DATASET_DIR
if args.output_dir is None:
    args.output_dir = DATASET_DIR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_builder import build_scene, sample_ue_positions, plot_topology

from sionna.rt import Receiver, PathSolver, subcarrier_frequencies

# ── 固定信道参数（带宽与天线数按场景随机，见主循环）
# ── Fixed channel parameters; bandwidth and antenna count vary per scene (see the main loop)
N_SUBCARRIERS = 128
CARRIER_FREQ  = 3.5e9
UE_HEIGHT     = 1.5
SYNTHETIC_ARRAY = bool(args.synthetic_array)

N_BATCH = args.batch_size   # 每批 UE 数（共享一次 PathSolver 调用）


# ─────────────────────────────────────────────
# 核心函数
# ─────────────────────────────────────────────

def setup_batch_receivers(scene, n_batch):
    """预先放好 n_batch 个占位接收机，之后只改坐标 —— 这样一次 PathSolver 调用
    可以同时解 n_batch 个 UE，比逐个 UE 快得多。
    Pre-create n_batch placeholder receivers and only move them afterwards, so one
    PathSolver call solves n_batch UEs at once instead of one per call."""
    for j in range(n_batch):
        name = f'rx_batch_{j:03d}'   # 零补3位，确保字典序=数字序
        if name not in scene.receivers:
            rx = Receiver(name=name, position=[1.0, 1.0, UE_HEIGHT])
            scene.add(rx)


def process_batch(scene, ue_positions, scene_id, freqs):
    """
    对一批 UE 调用一次 PathSolver，返回每个 UE 的 (h, phi_t, theta_t)；无路径的返回 None。
    Solve one batch of UEs with a single PathSolver call and return (h, phi_t, theta_t)
    per UE, or None where no path exists.

    ue_positions: (M, 2)，M <= N_BATCH
    freqs: 子载波频率数组（由该场景的带宽决定）/ subcarrier frequencies for this scene
    """
    M = len(ue_positions)

    # 更新前 M 个 receiver 位置，其余移到同一点（避免额外计算干扰）
    dummy = [float(ue_positions[0][0]), float(ue_positions[0][1]), UE_HEIGHT]
    for j in range(N_BATCH):
        if j < M:
            pos = [float(ue_positions[j][0]), float(ue_positions[j][1]), UE_HEIGHT]
        else:
            pos = dummy
        scene.receivers[f'rx_batch_{j:03d}'].position = pos

    solver = PathSolver()
    paths = solver(
        scene=scene,
        max_depth=5,
        max_num_paths_per_src=100000,
        samples_per_src=100000,
        synthetic_array=SYNTHETIC_ARRAY,
        los=True,
        specular_reflection=True,
        diffuse_reflection=True,
        refraction=True,
        diffraction=True,
        edge_diffraction=False,
        diffraction_lit_region=True,
        seed=scene_id,
    )

    # ── 提取功率（用于找最强路径）
    # paths.a = (a_real, a_imag)，shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    a_real = np.asarray(paths.a[0])
    a_imag = np.asarray(paths.a[1])
    # 对除 num_rx(0) 和 num_paths(-1) 之外的天线维度求和，得 [N_BATCH, P]
    reduce_axes = tuple(range(1, a_real.ndim - 1))
    power_all = np.sum(a_real**2 + a_imag**2, axis=reduce_axes)

    # ── 提取 DoD 角度
    # phi_t shape: [num_rx, num_tx, num_paths] = 3D (角度是路径属性，不依赖天线)
    phi_t_all   = np.asarray(paths.phi_t)
    theta_t_all = np.asarray(paths.theta_t)

    # ── 提取信道 CFR
    h_freq = paths.cfr(
        frequencies=freqs,
        normalize_delays=False,
        normalize=False,
        reverse_direction=True,
        out_type='numpy',
    )
    # cfr(reverse_direction=True) shape: (1, N_ANT, N_BATCH, 1, 1, N_SUBCAR)
    # 即 [num_rx=1_BS, num_rx_ant=32, num_tx=N_BATCH_UE, num_tx_ant=1, 1, num_freq]
    # 正确提取: [0, :, :, 0, 0, :] → (N_ANT, N_BATCH, N_SUBCAR) → transpose → (N_BATCH, N_ANT, N_SUBCAR)
    h_freq = h_freq[0, :, :, 0, 0, :].transpose(1, 0, 2)  # (N_BATCH, N_ANT, N_SUBCAR)

    results = []
    for j in range(M):
        pw = power_all[j]      # [P]
        if pw.size == 0 or pw.max() == 0:
            results.append(None)
            continue
        best = int(np.argmax(pw))
        # phi_t_all is 3D: [N_BATCH, num_tx, P]; theta_t_all same
        phi_t   = float(phi_t_all[j, 0, best])
        theta_t = float(theta_t_all[j, 0, best])
        h = h_freq[j, :, :].T   # [128, 32] complex
        results.append((h, phi_t, theta_t))

    return results


def save_ue_h5(h5_path, h, ue_pos, bs_pos, bs_azimuth_deg, phi_t, theta_t, scene_id,
               bandwidth, n_ant, max_retries=5, retry_delay=2.0):
    """写单个 UE 的 H5（带重试：网络文件系统偶发失败很常见）。
    Write one UE's H5 file, with retries — transient failures on a network file
    system are common."""
    import time
    bs_xy = np.array(bs_pos[:2], dtype=np.float32)
    dist  = float(np.linalg.norm(ue_pos - bs_xy))
    for attempt in range(max_retries):
        try:
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('channel_real', data=h.real.astype(np.float32))
                f.create_dataset('channel_imag', data=h.imag.astype(np.float32))
                f.create_dataset('UElocation',   data=(ue_pos / 10).astype(np.float32))
                f.create_dataset('BSlocation',   data=bs_xy)
                f.create_dataset('distance',     data=np.float32(dist / 100))
                f.create_dataset('DoD_phi',      data=np.float32(phi_t / 100))
                f.create_dataset('DoD_theta',    data=np.float32(theta_t / 100))
                f.create_dataset('bs_height',    data=np.float32(bs_pos[2]))
                f.create_dataset('bs_azimuth',   data=np.float32(np.deg2rad(bs_azimuth_deg)))
                f.create_dataset('scene_id',     data=np.int32(scene_id))
                f.create_dataset('bandwidth',    data=np.float32(bandwidth))
                f.create_dataset('n_ant',        data=np.int32(n_ant))
            return  # 成功，退出
        except OSError as e:
            if attempt < max_retries - 1:
                print(f"  [retry {attempt+1}/{max_retries}] h5 write error: {e}, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                raise


# ─────────────────────────────────────────────
# 主循环
# ─────────────────────────────────────────────

total_t0 = time.time()

for scene_id in range(args.scene_start, args.scene_end):
    scene_t0 = time.time()
    seed = args.seed + scene_id
    scene_dir = os.path.join(args.output_dir, f'scene_{scene_id:03d}')
    os.makedirs(scene_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Scene {scene_id}  [{args.scene_start}~{args.scene_end-1}]  seed={seed}")

    # ── 1. 采样本场景的无线参数（与建筑布局独立）
    #    Sample this scene's radio parameters (independent of the building layout)
    rng = np.random.default_rng(seed)
    if args.fixed_cfg:                                 # 未见场景测试集 / unseen test set
        bs_height, bs_az = 18.0, 45.0                  # 几何固定 / geometry frozen
    else:
        bs_height  = float(rng.uniform(15.0, 20.0))
        bs_az      = float(rng.uniform(25.0, 65.0))    # 中心 45°，±20° / 45 deg ± 20 deg
    # 带宽与天线数两种模式下都随机 / bandwidth and antenna count are random in both modes
    bandwidth  = float(rng.choice([5e6, 10e6, 20e6]))
    n_ant      = int(rng.choice([8, 16, 32]))

    print(f"  参数: BS height={bs_height:.1f}m  azimuth={bs_az:.1f}°  "
          f"bandwidth={bandwidth/1e6:.0f}MHz  n_ant={n_ant}")
    # ── 2. 构建场景（建筑布局由 seed 决定，无线参数由上方显式传入）
    scene, config = build_scene(
        seed=seed, scene_dir=scene_dir,
        bs_height=bs_height, bs_azimuth_deg=bs_az,
        bandwidth=bandwidth, n_ant=n_ant,
    )
    buildings = config['buildings']
    bs_pos    = config['bs_position']
    subcarrier_space = bandwidth / N_SUBCARRIERS
    freqs     = subcarrier_frequencies(N_SUBCARRIERS, subcarrier_space)

    print(f"  BS az={bs_az:.1f}°  h={bs_height:.1f}m  "
          f"BW={bandwidth/1e6:.0f}MHz  ant={n_ant}  |  {len(buildings)} 栋建筑")

    # ── 3. 采样候选 UE 位置（复用同一 rng，保证可复现性）
    n_candidates = min(args.n_ue * 3, 50000)
    ue_candidates = sample_ue_positions(rng, buildings, n_ue=n_candidates)
    print(f"  候选 UE 位置: {len(ue_candidates)}")

    # ── 4. 预先添加批量 Receiver
    setup_batch_receivers(scene, N_BATCH)

    # ── 5. 批量生成信道
    n_success = 0
    n_skip    = 0
    cand_idx  = 0
    saved_ue_pos = []

    while n_success < args.n_ue and cand_idx < len(ue_candidates):
        # 取一批候选位置
        batch_pos = ue_candidates[cand_idx: cand_idx + N_BATCH]
        cand_idx += N_BATCH
        if len(batch_pos) == 0:
            break

        results = process_batch(scene, batch_pos, scene_id, freqs)

        for j, res in enumerate(results):
            if n_success >= args.n_ue:
                break
            if res is None:
                n_skip += 1
                continue
            h, phi_t, theta_t = res
            h5_path = os.path.join(scene_dir, f'{n_success:05d}.h5py')
            save_ue_h5(h5_path, h, batch_pos[j], bs_pos, bs_az,
                        phi_t, theta_t, scene_id, bandwidth, n_ant)
            saved_ue_pos.append(batch_pos[j])
            n_success += 1

        # 进度打印（每 500 个 UE）
        if n_success > 0 and n_success % 500 == 0:
            elapsed = time.time() - scene_t0
            rate = n_success / elapsed
            eta  = (args.n_ue - n_success) / rate if rate > 0 else 0
            print(f"  [{scene_id}] {n_success}/{args.n_ue} UE  "
                  f"skip={n_skip}  {rate:.1f} UE/s  ETA {eta/60:.1f}min")

    # ── 6. 保存拓扑图
    if len(saved_ue_pos) > 0:
        vis_pos = np.array(saved_ue_pos[:500])
        plot_topology(buildings, bs_pos, bs_az,
                      ue_positions=vis_pos,
                      save_path=os.path.join(scene_dir, 'topology.png'))

    # ── 7. 更新 config 中的实际 UE 数
    config['n_ue'] = n_success
    with open(os.path.join(scene_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    scene_elapsed = time.time() - scene_t0
    print(f"  Scene {scene_id} 完成: {n_success} UE  "
          f"skip={n_skip}  耗时 {scene_elapsed/60:.1f}min")

    if n_success < args.n_ue:
        print(f"  [warn] 候选位置不足，仅生成 {n_success} 个 UE（目标 {args.n_ue}）")

total_elapsed = time.time() - total_t0
print(f"\n{'='*60}")
print(f"全部完成: scene {args.scene_start}~{args.scene_end-1}  "
      f"总耗时 {total_elapsed/60:.1f}min")
