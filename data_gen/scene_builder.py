"""
scene_builder.py
=====================================================================
随机场景生成 / Random scene generation.

中文：生成 32×32 m 的场景：地面 + 2~4 栋建筑（轴对齐长方体）+ 1 个基站
      （放在左下角 (0,0)，朝向随机），导出 PLY 网格与 Sionna 场景，并把几何
      和无线配置写进 config.json。建筑几何后续用于判定 LOS / NLOS。
EN:   Build a 32×32 m scene: ground plane + 2–4 axis-aligned building boxes + one
      BS at the (0,0) corner with a random boresight. Exports the PLY meshes and
      the Sionna scene, and records geometry plus radio config in config.json.
      The building boxes are later reused for the LOS / NLOS labelling.
"""
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ─────────────────────────────────────────────
# 1.  PLY 工具函数
# ─────────────────────────────────────────────

def _write_ply(filepath, vertices, faces):
    """写 ASCII PLY 文件（顶点 + 三角面）/ write an ASCII PLY file (vertices + triangles)."""
    with open(filepath, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def create_unit_box_ply(filepath):
    """单位 box PLY：顶点在 [-0.5, 0.5]^3，底面在 z=-0.5
    Unit-box PLY with vertices in [-0.5, 0.5]^3 and its base at z = -0.5."""
    v = np.array([
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
    ], dtype=np.float32)
    f = np.array([
        [0,2,1],[0,3,2],   # bottom  (outward normal: -z)
        [4,5,6],[4,6,7],   # top     (+z)
        [0,1,5],[0,5,4],   # front   (-y)
        [2,3,7],[2,7,6],   # back    (+y)
        [0,4,7],[0,7,3],   # left    (-x)
        [1,2,6],[1,6,5],   # right   (+x)
    ], dtype=np.int32)
    _write_ply(filepath, v, f)


def create_ground_ply(filepath, size=32.0):
    """正方形地面 PLY：z=0，[0, size]×[0, size]
    Square ground plane at z = 0 covering [0, size] x [0, size]."""
    v = np.array([
        [0,    0,    0],
        [size, 0,    0],
        [size, size, 0],
        [0,    size, 0],
    ], dtype=np.float32)
    f = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
    _write_ply(filepath, v, f)


# ─────────────────────────────────────────────
# 2.  建筑随机生成（rejection sampling）
# ─────────────────────────────────────────────

def _aabb(b):
    """建筑的轴对齐矩形 / axis-aligned footprint [x_min, x_max, y_min, y_max]."""
    return (b['cx'] - b['length']/2, b['cx'] + b['length']/2,
            b['cy'] - b['width']/2,  b['cy'] + b['width']/2)


def _overlap(b1, b2, margin=1.0):
    """两栋建筑是否重叠（含 margin 间距）/ do the two buildings overlap (with margin)?"""
    x0, x1, y0, y1 = _aabb(b1)
    x0 -= margin; x1 += margin; y0 -= margin; y1 += margin
    bx0, bx1, by0, by1 = _aabb(b2)
    return not (x1 < bx0 or bx1 < x0 or y1 < by0 or by1 < y0)


def _inside_building(x, y, buildings):
    """点 (x,y) 是否落在任意建筑内部 / is the point (x, y) inside any building?"""
    for b in buildings:
        x0, x1, y0, y1 = _aabb(b)
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def generate_buildings(rng, map_size=32.0, bs_exclusion_r=3.0,
                        len_range=(5, 16), wid_range=(5, 10), height=10.0,
                        n_range=(2, 4), margin=1.0, max_tries=500,
                        coverage_min=0.10, coverage_max=0.50,
                        max_scene_tries=20):
    """
    生成 n_range[0]~n_range[1] 栋互不重叠的建筑；总占地面积控制在
    [coverage_min, coverage_max] × map_size² 之间；BS 在 (0,0)，其周围
    bs_exclusion_r 米内不放建筑（否则基站会被自己的楼挡住）。

    Sample n_range[0]..n_range[1] non-overlapping buildings whose total footprint
    stays within [coverage_min, coverage_max] x map_size^2. The BS sits at (0,0) and
    no building is placed within bs_exclusion_r metres of it, otherwise the BS would
    be blocked by its own building.
    """
    map_area = map_size ** 2

    for scene_try in range(max_scene_tries):
        n = int(rng.integers(n_range[0], n_range[1] + 1))
        buildings = []

        for _ in range(n):
            placed = False
            for _ in range(max_tries):
                length = float(rng.uniform(*len_range))
                width  = float(rng.uniform(*wid_range))
                cx = float(rng.uniform(max(length/2 + 1, 5.0), min(map_size - length/2 - 1, 30.0)))
                cy = float(rng.uniform(max(width/2  + 1, 5.0), min(map_size - width/2  - 1, 30.0)))
                b = dict(cx=cx, cy=cy, length=length, width=width, height=height)
                if cx**2 + cy**2 < bs_exclusion_r**2:
                    continue
                if any(_overlap(b, existing, margin) for existing in buildings):
                    continue
                buildings.append(b)
                placed = True
                break
            if not placed:
                break  # 空间不足，进入覆盖率检查

        coverage = sum(b['length'] * b['width'] for b in buildings) / map_area
        if coverage_min <= coverage <= coverage_max:
            print(f"  建筑覆盖率: {coverage*100:.1f}%  ({len(buildings)} 栋)")
            return buildings

    # 超过最大重试次数，返回最后一次结果并警告
    coverage = sum(b['length'] * b['width'] for b in buildings) / map_area
    print(f"  [warn] 未能满足覆盖率约束，当前: {coverage*100:.1f}%，返回现有结果")
    return buildings


# ─────────────────────────────────────────────
# 3.  Sionna 场景 XML 生成
# ─────────────────────────────────────────────

def build_scene_xml(scene_dir, buildings, unit_box_relpath='meshes/unit_box.ply',
                    ground_relpath='meshes/ground.ply'):
    """生成 Sionna 场景 XML（PLY 路径都相对于 scene_dir）
    Build the Sionna scene XML; all PLY paths are relative to scene_dir."""
    lines = ['<scene version="2.1.0">',
             '',
             '<!-- Materials -->',
             '    <bsdf type="itu-radio-material" id="concrete">',
             '        <string name="type" value="concrete"/>',
             '        <float name="thickness" value="0.5"/>',
             '    </bsdf>',
             '',
             '<!-- Shapes -->',
             '',
             '    <!-- Ground -->',
             '    <shape type="ply" id="mesh-ground">',
             f'        <string name="filename" value="{ground_relpath}"/>',
             '        <boolean name="face_normals" value="true"/>',
             '        <ref id="concrete" name="bsdf"/>',
             '    </shape>',
             '']

    for i, b in enumerate(buildings):
        # Mitsuba transform：先 scale 到建筑尺寸，再 translate 到中心
        # 单位 box 底面在 z=-0.5 → translate z += height/2 使底面贴地
        sx, sy, sz = b['length'], b['width'], b['height']
        tx, ty, tz = b['cx'], b['cy'], b['height'] / 2
        lines += [
            f'    <!-- Building {i} -->',
            f'    <shape type="ply" id="mesh-building_{i}">',
            f'        <string name="filename" value="{unit_box_relpath}"/>',
            f'        <boolean name="face_normals" value="true"/>',
            f'        <transform name="to_world">',
            f'            <scale x="{sx:.4f}" y="{sy:.4f}" z="{sz:.4f}"/>',
            f'            <translate x="{tx:.4f}" y="{ty:.4f}" z="{tz:.4f}"/>',
            f'        </transform>',
            f'        <ref id="concrete" name="bsdf"/>',
            f'    </shape>',
            '',
        ]

    lines.append('</scene>')
    return '\n'.join(lines)


# ─────────────────────────────────────────────
# 4.  俯瞰拓扑图
# ─────────────────────────────────────────────

def plot_topology(buildings, bs_pos, bs_azimuth_deg, ue_positions=None,
                  map_size=32.0, save_path=None):
    with plt.rc_context({'font.family': 'serif',
                          'font.serif': ['Times New Roman', 'DejaVu Serif'],
                          'mathtext.fontset': 'stix'}):
        return _plot_topology_impl(buildings, bs_pos, bs_azimuth_deg,
                                    ue_positions, map_size, save_path)


def _plot_topology_impl(buildings, bs_pos, bs_azimuth_deg, ue_positions,
                         map_size, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1, map_size + 1)
    ax.set_ylim(-1, map_size + 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', fontsize=20)
    ax.set_ylabel('y (m)', fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True, alpha=0.3)

    # 地图边界
    ax.add_patch(patches.Rectangle((0, 0), map_size, map_size,
                                    linewidth=1.5, edgecolor='black',
                                    facecolor='lightyellow', zorder=0))
    # 建筑
    for b in buildings:
        x0 = b['cx'] - b['length'] / 2
        y0 = b['cy'] - b['width']  / 2
        ax.add_patch(patches.Rectangle((x0, y0), b['length'], b['width'],
                                        facecolor='gray', edgecolor='black',
                                        linewidth=0.8, zorder=2))
        ax.text(b['cx'], b['cy'], f"{b['length']:.1f}×{b['width']:.1f}",
                ha='center', va='center', fontsize=16, color='white', zorder=3)

    # UE 位置
    if ue_positions is not None and len(ue_positions) > 0:
        ax.scatter(ue_positions[:, 0], ue_positions[:, 1],
                   s=1, alpha=0.4, c='steelblue', zorder=1, label='UE')

    # BS 位置 + 朝向箭头
    bx, by = bs_pos[0], bs_pos[1]
    ax.plot(bx, by, 'r^', markersize=20, zorder=5, label='BS')
    arrow_len = 4.0
    az_rad = np.deg2rad(bs_azimuth_deg)
    ax.annotate('', xy=(bx + arrow_len * np.cos(az_rad),
                         by + arrow_len * np.sin(az_rad)),
                xytext=(bx, by),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


# ─────────────────────────────────────────────
# 5.  主函数：构建一个完整场景
# ─────────────────────────────────────────────

def build_scene(seed, scene_dir, bs_height, bs_azimuth_deg, bandwidth, n_ant,
                map_size=32.0, building_height=10.0):
    """
    构建确定性场景：建筑布局完全由 seed 决定，无线参数由调用方传入，
    因此同一个 seed 总能重现同一个场景。返回 (sionna_scene, config_dict)。

    Build a deterministic scene: the layout is fully determined by `seed` while the
    radio parameters are passed in, so the same seed always reproduces the same
    scene. Returns (sionna_scene, config_dict).

    参数 / arguments:
      bs_height      BS 天线高度 (m)
      bs_azimuth_deg BS 方位角 (°)
      bandwidth      系统带宽 (Hz)，如 5e6 / 10e6 / 20e6
      n_ant          BS 天线数，如 8 / 16 / 32
    """
    from sionna.rt import load_scene, Transmitter, PlanarArray

    os.makedirs(scene_dir, exist_ok=True)
    mesh_dir = os.path.join(scene_dir, 'meshes')
    os.makedirs(mesh_dir, exist_ok=True)

    # seed 仅用于建筑布局随机化
    rng = np.random.default_rng(seed)
    bs_pos = [0.0, 0.0, float(bs_height)]

    # ── 建筑
    buildings = generate_buildings(rng, map_size=map_size, height=building_height)

    # ── 写 PLY 文件
    unit_box_path = os.path.join(mesh_dir, 'unit_box.ply')
    ground_path   = os.path.join(mesh_dir, 'ground.ply')
    if not os.path.exists(unit_box_path):
        create_unit_box_ply(unit_box_path)
    create_ground_ply(ground_path, size=map_size)

    # ── 写 XML
    xml_str = build_scene_xml(scene_dir, buildings,
                               unit_box_relpath='meshes/unit_box.ply',
                               ground_relpath='meshes/ground.ply')
    xml_path = os.path.join(scene_dir, 'scene.xml')
    with open(xml_path, 'w') as f:
        f.write(xml_str)

    # ── 加载 Sionna 场景
    scene = load_scene(xml_path)
    scene.bandwidth = float(bandwidth)

    # ── 配置 BS
    scene.frequency = 3.5e9

    # Sionna PlanarArray spacing单位是波长倍数（默认0.5=λ/2），不是米
    scene.tx_array = PlanarArray(num_rows=1, num_cols=int(n_ant),
                                  horizontal_spacing=0.5,
                                  vertical_spacing=0.5,
                                  pattern="tr38901", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
                                  horizontal_spacing=0.5,
                                  vertical_spacing=0.5,
                                  pattern="iso", polarization="V")
    scene.synthetic_array = True

    # Sionna 朝向约定：orientation = [alpha, beta, gamma]
    # alpha: 绕 z 轴旋转（方位角），0 → +x 方向，π/2 → +y 方向（逆时针）
    # beta:  绕 y 轴旋转（俯仰角），0 → 不倾斜
    bs_orientation = [float(np.deg2rad(bs_azimuth_deg)), 0.0, 0.0]
    tx = Transmitter(name='tx0', position=[float(v) for v in bs_pos],
                     orientation=bs_orientation, power_dbm=30)
    scene.add(tx)

    # ── 保存 config
    config = {
        'scene_id': seed,
        'bs_position': bs_pos,
        'bs_azimuth_deg': float(bs_azimuth_deg),
        'bs_downtilt_deg': 0.0,
        'buildings': buildings,
        'n_ue': 0,
        'map_size': map_size,
        'bs_height': float(bs_height),
        'attenna_cols': int(n_ant),
        'bandwidth': float(bandwidth),
    }
    with open(os.path.join(scene_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    return scene, config


# ─────────────────────────────────────────────
# 6.  UE 采样（rejection sampling）
# ─────────────────────────────────────────────

def sample_ue_positions(rng, buildings, n_ue, map_size=32.0,
                         ue_margin=1.0, max_tries_factor=20):
    """在地图内均匀采样 UE 位置，落在建筑内部的点丢弃，返回 (N, 2)。
    Uniformly sample UE positions over the map, dropping those inside buildings."""
    positions = []
    max_tries = n_ue * max_tries_factor
    tries = 0
    while len(positions) < n_ue and tries < max_tries:
        x = float(rng.uniform(ue_margin, map_size - ue_margin))
        y = float(rng.uniform(ue_margin, map_size - ue_margin))
        if not _inside_building(x, y, buildings):
            positions.append([x, y])
        tries += 1
    if len(positions) < n_ue:
        print(f"  [warn] 只采样到 {len(positions)}/{n_ue} 个有效 UE 位置")
    return np.array(positions, dtype=np.float32)
