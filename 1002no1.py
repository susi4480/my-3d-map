# -*- coding: utf-8 -*-
"""
【機能】高密度 Ouster OS1-64 風レイキャスト（360° / 64ch / 高分解能）
-------------------------------------------------------------------
- 中心線からLiDAR位置と向きを決定（CENTER_IDX / TARGET_IDX）
- 水平360°、0.35°刻み・垂直±16.5°で64ビーム（等間隔近似）
- 1レイあたり3000ステップ、HIT_THR=0.40mで最初の命中点のみ採用
- 遮蔽物裏は除外（ファーストリターン）
- 出力：scan_sector_{CENTER_IDX}_os1_64_dense.las（白色）
-------------------------------------------------------------------
必要: numpy, laspy, open3d
"""

import os
import math
import numpy as np
import laspy
import open3d as o3d
from pyproj import CRS

# ============== 入出力・基本設定 ==============
INPUT_LAS   = "/output/0925_sita_merged_white.las"
OUTPUT_DIR  = "/output/forward_scans_raycast"

CENTER_IDX  = 2000   # LiDAR位置（中心線上のインデックス）
TARGET_IDX  = 2005   # 視線方向参照点（接線ベクトルの取得用）

# 中心線抽出時の高さ基準
UKC         = -2.0
TOL_Z       = 0.2
Z_MAX       = 10.0

# Ouster OS1-64 風設定
FOV_H_DEG   = 360.0      # 水平視野（フルスキャン）
H_RES_DEG   = 0.35       # 水平角度分解能（°）
V_FOV_DEG   = 33.0       # 垂直視野（合計） ≒ ±16.5°
V_CHANNELS  = 64         # ビーム数

# レイ・ヒット判定
MAX_RANGE   = 200.0      # 最大射程[m]
STEP_COUNT  = 3000       # 1レイのステップ数（約6.7cm刻み）
HIT_THR     = 0.40       # 命中判定距離[m]

# 出力
DOWNSAMPLE  = False      # Trueなら軽量化
VOXEL_SIZE  = 0.05       # 軽量化時のボクセルサイズ[m]

# ============== ユーティリティ ==============
def l2(p, q):
    return math.hypot(q[0]-p[0], q[1]-p[1])

def write_las_xyz_rgb(path, xyz, rgb=None, epsg=32654):
    if xyz.size == 0:
        print("⚠ 出力点なし")
        return
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_crs(CRS.from_epsg(epsg))
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgb is not None:
        las.red, las.green, las.blue = rgb[:,0], rgb[:,1], rgb[:,2]
    las.write(path)
    print(f"💾 保存: {path} ({len(xyz):,} 点)")

def extract_centerline(X, Y, Z, ukc=-2.0, tol_z=0.2):
    """X方向ビニングからUKC付近の左右端の中点を通る簡易中心線を生成"""
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
        m_ukc = np.abs(slab_z - ukc) < tol_z
        if not np.any(m_ukc):
            continue
        slab_xy = slab_xy[m_ukc]
        order = np.argsort(slab_xy[:, 1])
        left, right = slab_xy[order[0]], slab_xy[order[-1]]
        through.append(0.5 * (left + right))
    through = np.asarray(through, float)
    if len(through) < 2:
        raise RuntimeError("中心線が作れません。")

    thinned = [through[0]]
    for p in through[1:]:
        if l2(thinned[-1], p) >= GAP_DIST:
            thinned.append(p)
    through = np.asarray(thinned, float)

    centers, tangents = [], []
    for i in range(len(through)-1):
        p, q = through[i], through[i+1]
        d = l2(p, q)
        if d < 1e-9:
            continue
        n_steps = int(d / SECTION_INTERVAL)
        t_hat = (q - p) / d
        for s_i in range(n_steps+1):
            s = min(s_i * SECTION_INTERVAL, d)
            t = s / d
            centers.append((1-t)*p + t*q)
            tangents.append(t_hat)
    return np.asarray(centers, float), np.asarray(tangents, float)

def build_local_basis(forward):
    """forward（水平面の接線）から right, up（Z軸）を作るローカル基底"""
    f = np.array([forward[0], forward[1], 0.0], float)
    nf = np.linalg.norm(f)
    if nf < 1e-9:
        f = np.array([1.0, 0.0, 0.0])
    else:
        f /= nf
    up = np.array([0.0, 0.0, 1.0], float)
    right = np.cross(f, up)
    nr = np.linalg.norm(right)
    if nr < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= nr
    # 再直交化
    f = np.cross(up, right)
    f /= np.linalg.norm(f)
    return f, right, up

# ============== メイン ==============
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 入力読み込み
    las = laspy.read(INPUT_LAS)
    X = np.asarray(las.x, float)
    Y = np.asarray(las.y, float)
    Z = np.asarray(las.z, float)

    # 対象点（高さ上限）
    m_nav = (Z <= Z_MAX)
    xyz = np.column_stack([X[m_nav], Y[m_nav], Z[m_nav]])

    # 中心線と向き
    centers, tangents = extract_centerline(X, Y, Z, UKC, TOL_Z)
    origin = np.array([centers[CENTER_IDX, 0], centers[CENTER_IDX, 1], UKC], float)
    fwd2d  = tangents[TARGET_IDX]    # XY平面の接線方向
    forward, right, up = build_local_basis(fwd2d)

    print(f"📍 センサ位置: {origin}")
    print(f"🎯 基底 | forward: {forward}, right: {right}, up: {up}")

    # KDTree 構築
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # 角度セット
    num_h = int(round(FOV_H_DEG / H_RES_DEG)) + 1
    yaw_list = np.linspace(-FOV_H_DEG/2.0, +FOV_H_DEG/2.0, num_h)  # -180°～+180°
    v_min = -V_FOV_DEG/2.0
    v_max = +V_FOV_DEG/2.0
    if V_CHANNELS <= 1:
        pitch_list = np.array([0.0])
    else:
        pitch_list = np.linspace(v_min, v_max, V_CHANNELS)  # 等間隔近似

    print(f"🟢 水平レイ数: {len(yaw_list)}（{FOV_H_DEG}° / {H_RES_DEG}°）")
    print(f"🔵 垂直レイ数: {len(pitch_list)}（{V_CHANNELS} ch, ±{V_FOV_DEG/2.0}°）")

    # ステップ距離
    r_list = np.linspace(0.0, MAX_RANGE, STEP_COUNT)

    hits = []

    # レイキャスト（外側ループはピッチ→ヨーでキャッシュ局所性多少改善）
    for pitch_deg in pitch_list:
        pitch = math.radians(pitch_deg)
        cos_p = math.cos(pitch)
        sin_p = math.sin(pitch)

        for yaw_deg in yaw_list:
            yaw = math.radians(yaw_deg)
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)

            # ローカル基底での方向ベクトル:
            # dir = cos(pitch)*(cos(yaw)*forward + sin(yaw)*right) + sin(pitch)*up
            dir_vec = cos_p * (cos_y * forward + sin_y * right) + sin_p * up
            dir_vec /= np.linalg.norm(dir_vec) + 1e-12

            # 前方のみ探索（遮蔽裏は最初のヒットでbreak）
            hit_found = False
            for r in r_list:
                p = origin + dir_vec * r
                _, idxs, d2 = kdtree.search_knn_vector_3d(p, 1)
                if len(idxs) > 0:
                    dist = math.sqrt(d2[0])
                    if dist < HIT_THR:
                        hits.append(np.asarray(pcd.points)[idxs[0]])
                        hit_found = True
                        break
            # 見つからない場合はスキップ（そのレイは未ヒット）

    hits = np.asarray(hits, float)
    if hits.size == 0:
        print("⚠ レイキャスト結果なし")
        return

    # 必要なら軽量化
    if DOWNSAMPLE:
        pcd_out = o3d.geometry.PointCloud()
        pcd_out.points = o3d.utility.Vector3dVector(hits)
        pcd_out = pcd_out.voxel_down_sample(VOXEL_SIZE)
        hits = np.asarray(pcd_out.points)

    # 全点白色で出力（Intensityは本処理では保持しない）
    rgb = np.full((len(hits), 3), 65535, dtype=np.uint16)
    out_path = os.path.join(OUTPUT_DIR, f"scan_sector_{CENTER_IDX:04d}_os1_64_dense.las")
    write_las_xyz_rgb(out_path, hits, rgb=rgb, epsg=32654)
    print(f"🎉 高密度レイキャスト完了: {len(hits):,} 点")


if __name__ == "__main__":
    main()
