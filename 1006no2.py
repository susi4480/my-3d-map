# -*- coding: utf-8 -*-
"""
【機能】Ouster OS1-64風レイキャスト + ローカル座標化 + ぼかし + ランダム姿勢付与
------------------------------------------------------------------------------------
- 3D地図から擬似スキャンをレイキャストで生成（遮蔽裏は除外）
- 地図座標(ワールド) -> センサーローカル座標(+X=視線)へ変換
- ノイズ/ダウンサンプルで“ぼかし”（LI DARらしさを再現）
- ランダム回転・並進を付与して、地図と重ならない「クエリ点群」を出力
- Ground Truthの姿勢(R,t)と元のセンサー位置/ヨー角をメタデータで保存

出力:
  scan_sector_{CENTER_IDX:04d}_raycast_world.las   … レイキャスト結果（地図座標のまま）
  scan_sector_{CENTER_IDX:04d}_local.las           … センサーローカル座標（原点=LiDAR）
  scan_sector_{CENTER_IDX:04d}_query_world.las     … ランダム姿勢を与えた“未知姿勢”クエリ
  scan_sector_{CENTER_IDX:04d}_meta.txt            … 姿勢GTやパラメータ
"""
import os
import math
import json
import numpy as np
import laspy
import open3d as o3d
from datetime import datetime

# ==========================================================
# 入出力
# ==========================================================
INPUT_LAS   = "/output/0925_sita_merged_white.las"
OUTPUT_DIR  = "/output/1006no2_forward_scans_raycast"

# 中心線上のインデックス（LiDAR位置と視線推定用）
CENTER_IDX  = 2000
TARGET_IDX  = 2005

# 視野・分解能（Ouster OS1-64 近似）
FOV_H_DEG   = 90.0     # 水平視野角 ±45°
FOV_V_DEG   = 33.0     # 垂直視野角 ±16.5°
H_RES       = 0.35     # 水平方向分解能[°]
V_RES       = 0.5      # 垂直方向分解能[°]

# レイキャスト
MAX_RANGE   = 200.0    # 射程[m]
STEP_COUNT  = 1200     # レイ1本あたりのステップ数
HIT_THR     = 0.20     # 衝突判定距離[m]

# 地図点群の前処理
UKC         = -2.0     # 中心線高さ近傍[Z]
TOL_Z       = 0.2      # UKC許容
Z_MAX       = 10.0     # 地図として使う上限高さ[m]

# ぼかし（ノイズ/ダウンサンプル）
NOISE_STD   = 0.05     # [m] XYZガウスノイズ σ（例: 5cm）
VOXEL_SIZE  = 0.10     # [m] 体素間引き（解像度低下）

# ランダム姿勢（クエリ点群を地図とズラす）
RAND_YAW_DEG_RANGE   = (-8.0, 8.0)    # [deg]
RAND_PITCH_DEG_RANGE = (-3.0, 3.0)    # [deg]
RAND_ROLL_DEG_RANGE  = (-3.0, 3.0)    # [deg]
RAND_TRANS_RANGE_M   = { "x": (-1.5, 1.5), "y": (-1.5, 1.5), "z": (-0.3, 0.3) }

# 再現性（固定したい時は値を設定）
RANDOM_SEED = 42
# ==========================================================


# --------- ユーティリティ ---------
def l2(p, q):
    return math.hypot(q[0]-p[0], q[1]-p[1])

def write_las_xyz(path, xyz):
    if xyz.size == 0:
        print("⚠ 出力点なし:", path)
        return
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:,0], xyz[:,1], xyz[:,2]
    las.write(path)
    print(f"💾 保存: {path} ({len(xyz):,} 点)")

def rotz(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)

def roty(deg):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    return np.array([[ c,0, s],[0,1,0],[-s,0, c]], float)

def rotx(deg):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    return np.array([[1,0,0],[0, c,-s],[0, s, c]], float)

# --------- 中心線抽出（元コード）---------
def extract_centerline(X, Y, Z):
    BIN_X = 2.0
    MIN_PTS_PER_XBIN = 50
    GAP_DIST = 50.0
    SECTION_INTERVAL = 0.5

    x_min, x_max = X.min(), X.max()
    edges = np.arange(x_min, x_max + BIN_X, BIN_X)
    through = []
    for i in range(len(edges)-1):
        x0, x1 = edges[i], edges[i+1]
        m = (X >= x0) & (X < x1)
        if np.count_nonzero(m) < MIN_PTS_PER_XBIN:
            continue
        slab_xy = np.column_stack([X[m], Y[m]])
        slab_z  = Z[m]
        m_ukc = np.abs(slab_z - UKC) < TOL_Z
        if not np.any(m_ukc): continue
        slab_xy = slab_xy[m_ukc]
        order = np.argsort(slab_xy[:,1])
        left, right = slab_xy[order[0]], slab_xy[order[-1]]
        through.append(0.5*(left+right))
    through = np.asarray(through,float)
    if len(through) < 2:
        raise RuntimeError("中心線が作れません。")

    thinned = [through[0]]
    for p in through[1:]:
        if l2(thinned[-1],p) >= GAP_DIST:
            thinned.append(p)
    through = np.asarray(thinned,float)

    centers = []; tangents = []
    for i in range(len(through)-1):
        p,q = through[i], through[i+1]
        d = l2(p,q)
        if d < 1e-9: continue
        n_steps = int(d / SECTION_INTERVAL)
        t_hat = (q - p) / d
        for s_i in range(n_steps+1):
            s = min(s_i*SECTION_INTERVAL, d)
            t = s / d
            centers.append((1-t)*p + t*q)
            tangents.append(t_hat)
    return np.asarray(centers,float), np.asarray(tangents,float)

# --------- メイン ---------
def main():
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 地図読み込み
    las = laspy.read(INPUT_LAS)
    X, Y, Z = np.asarray(las.x,float), np.asarray(las.y,float), np.asarray(las.z,float)
    m_nav = (Z <= Z_MAX)
    xyz_map = np.column_stack([X[m_nav], Y[m_nav], Z[m_nav]])

    # 中心線生成 & LiDAR原点/視線
    centers, tangents = extract_centerline(X,Y,Z)
    origin_world = np.array([centers[CENTER_IDX,0], centers[CENTER_IDX,1], UKC], float)
    view_dir     = tangents[TARGET_IDX].astype(float)
    view_dir    /= np.linalg.norm(view_dir)

    print(f"📍 センサ位置(世界): {origin_world}, 視線方向(水平): {view_dir}")

    # KDTree
    pcd_map = o3d.geometry.PointCloud()
    pcd_map.points = o3d.utility.Vector3dVector(xyz_map)
    kdtree = o3d.geometry.KDTreeFlann(pcd_map)

    # レイ角
    num_h = int(FOV_H_DEG / H_RES) + 1
    num_v = int(FOV_V_DEG / V_RES) + 1
    h_angles = np.linspace(-FOV_H_DEG/2, FOV_H_DEG/2, num_h)
    v_angles = np.linspace(-FOV_V_DEG/2, FOV_V_DEG/2, num_v)
    print(f"🟢 水平レイ数: {num_h}, 垂直レイ数: {num_v}")

    # レイキャスト（世界座標のヒット点）
    hits_world = []
    for h in h_angles:
        for v in v_angles:
            theta = math.radians(h)
            phi   = math.radians(v)
            # 水平回転（視線ベクトルを基準にyaw回転）
            dir_h = np.array([
                view_dir[0]*math.cos(theta) - view_dir[1]*math.sin(theta),
                view_dir[0]*math.sin(theta) + view_dir[1]*math.cos(theta),
                0.0
            ], float)
            dir_h /= (np.linalg.norm(dir_h) + 1e-12)
            # 垂直成分を付与（簡易：Zはtanで傾ける）
            dir = dir_h.copy()
            dir[2] = math.tan(phi)
            dir /= (np.linalg.norm(dir) + 1e-12)

            for r in np.linspace(0, MAX_RANGE, STEP_COUNT):
                p = origin_world + dir * r
                _, idx, dist2 = kdtree.search_knn_vector_3d(p, 1)
                if len(idx) > 0 and math.sqrt(dist2[0]) < HIT_THR:
                    hits_world.append(np.asarray(pcd_map.points)[idx[0]])
                    break

    hits_world = np.asarray(hits_world, float)
    if hits_world.size == 0:
        print("⚠ レイキャスト結果なし")
        return

    # 出力(そのまま世界座標) — デバッグ用
    out_world = os.path.join(OUTPUT_DIR, f"scan_sector_{CENTER_IDX:04d}_raycast_world.las")
    write_las_xyz(out_world, hits_world)

    # ===== ローカル座標化 =====
    # 視線を+Xに合わせるためのヨー角（世界→ローカル）
    yaw_deg = math.degrees(math.atan2(view_dir[1], view_dir[0]))
    R_world_to_local = rotz(-yaw_deg)  # 世界での回転を逆に適用
    t_world_to_local = -origin_world

    # p_local = R*(p_world + t)
    hits_local = (R_world_to_local @ (hits_world + t_world_to_local).T).T

    # ===== ぼかし（ノイズ＋体素） =====
    if NOISE_STD > 0:
        hits_local = hits_local + np.random.normal(0.0, NOISE_STD, hits_local.shape)

    pcd_local = o3d.geometry.PointCloud()
    pcd_local.points = o3d.utility.Vector3dVector(hits_local)
    if VOXEL_SIZE > 0:
        pcd_local = pcd_local.voxel_down_sample(VOXEL_SIZE)
    hits_local = np.asarray(pcd_local.points)

    out_local = os.path.join(OUTPUT_DIR, f"scan_sector_{CENTER_IDX:04d}_local.las")
    write_las_xyz(out_local, hits_local)

    # ===== ランダム姿勢を与えて“未知姿勢”クエリ点群を世界座標に配置 =====
    ry = np.random.uniform(*RAND_PITCH_DEG_RANGE)  # pitch
    rx = np.random.uniform(*RAND_ROLL_DEG_RANGE)   # roll
    rz = np.random.uniform(*RAND_YAW_DEG_RANGE)    # yaw

    R_noise = rotz(rz) @ roty(ry) @ rotx(rx)

    tx = np.random.uniform(*RAND_TRANS_RANGE_M["x"])
    ty = np.random.uniform(*RAND_TRANS_RANGE_M["y"])
    tz = np.random.uniform(*RAND_TRANS_RANGE_M["z"])
    t_noise = np.array([tx, ty, tz], float)

    # ローカル→世界（元の向きへ戻す回転）
    R_local_to_world = rotz(yaw_deg)

    # クエリ点群（世界座標）
    # 1) ローカル点をノイズ姿勢で回す
    # 2) 世界向きへ戻す
    # 3) 元の原点に平行移動
    # 4) さらにランダム並進
    hits_query_world = (R_local_to_world @ (R_noise @ hits_local.T)).T + origin_world + t_noise

    out_query = os.path.join(OUTPUT_DIR, f"scan_sector_{CENTER_IDX:04d}_query_world.las")
    write_las_xyz(out_query, hits_query_world)

    # ===== メタデータ保存（Ground Truthなど） =====
    meta = {
        "timestamp": datetime.now().isoformat(),
        "input_las": INPUT_LAS,
        "center_idx": CENTER_IDX,
        "target_idx": TARGET_IDX,
        "origin_world": origin_world.tolist(),
        "view_dir_world": view_dir.tolist(),
        "yaw_deg_world_forward": yaw_deg,
        "fov_h_deg": FOV_H_DEG,
        "fov_v_deg": FOV_V_DEG,
        "h_res_deg": H_RES,
        "v_res_deg": V_RES,
        "max_range_m": MAX_RANGE,
        "hit_threshold_m": HIT_THR,
        "blur_noise_std_m": NOISE_STD,
        "voxel_size_m": VOXEL_SIZE,
        "random_pose_deg": {"roll": rx, "pitch": ry, "yaw": rz},
        "random_trans_m": {"x": tx, "y": ty, "z": tz},
        "files": {
            "raycast_world": out_world,
            "local": out_local,
            "query_world": out_query
        }
    }
    meta_path = os.path.join(OUTPUT_DIR, f"scan_sector_{CENTER_IDX:04d}_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"📝 メタ保存: {meta_path}")

    print("✅ 完了: ")
    print("  - 地図座標のレイキャスト        →", out_world)
    print("  - センサーローカル座標           →", out_local)
    print("  - 地図とズレたクエリ(未知姿勢)   →", out_query)

if __name__ == "__main__":
    main()
