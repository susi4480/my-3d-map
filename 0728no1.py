# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルを0.5m間隔でY-Z断面スライス（±0.1m厚）
- 各スライスに対して：
    - Z > 3.2 除去＋ノイズ除去（空でも続行）
    - Y方向に5分割した5点からRayを出す（空ならスキップ）
    - 航行可能空間（緑点）を生成
    - 除去した点（Z>3.2やノイズ）も戻して全体出力
"""

import os
import math
import numpy as np
import laspy
from scipy.spatial import cKDTree, ConvexHull, QhullError
from sklearn.neighbors import LocalOutlierFactor

# === 入出力設定 ===
input_las = "/data/0704_suidoubasi_ue.las"
output_dir = "/output/slice_area_navigation_final/"
os.makedirs(output_dir, exist_ok=True)

# === スライスパラメータ ===
slice_thickness = 0.2  # ±0.1m
slice_interval = 0.5   # 中心の間隔（0.5mごと）

# === 航行空間抽出パラメータ ===
Z_CUTOFF = 3.2
WATER_LEVEL = 3.5
CLEARANCE = 1.0
VOXEL_SIZE = 0.2
N_RAYS = 720
RAY_LENGTH = 60.0   # ← 3倍にしたRay長
STEP = 0.05
DIST_THRESH = 0.1
SAFETY_DIST = 1.0

# === Raycast関数（origin複数）===
def run_raycast_multi(pts, origins):
    if len(pts) < 3 or len(origins) == 0:
        return np.empty((0, 3))

    tree = cKDTree(pts[:,1:3])
    angles = np.linspace(0, 2*np.pi, N_RAYS, endpoint=False)
    dirs = np.vstack((np.cos(angles), np.sin(angles))).T
    steps = np.arange(0, RAY_LENGTH+STEP, STEP)

    all_hits = []
    for origin in origins:
        grid = origin + steps[:,None,None]*dirs[None,:,:]
        flat = grid.reshape(-1,2)
        dists, _ = tree.query(flat)
        dists = dists.reshape(grid.shape[:2])

        for j in range(N_RAYS):
            col = dists[:,j]
            idx = np.where(col < DIST_THRESH)[0]
            if idx.size == 0:
                continue
            i = idx[0]
            d_safe = max(steps[i] - SAFETY_DIST, 0)
            dy, dz = dirs[j]
            y_s, z_s = origin + np.array([dy, dz]) * d_safe
            _, ii = tree.query([y_s, z_s])
            x_s = pts[ii, 0]
            all_hits.append([x_s, y_s, z_s])

    if not all_hits:
        return np.empty((0,3))

    hits = np.array(all_hits)
    minb = pts.min(axis=0)
    ijk = np.floor((hits - minb) / VOXEL_SIZE).astype(int)
    uidx = np.unique(ijk, axis=0)
    centers = minb + (uidx + 0.5) * VOXEL_SIZE
    z_lim = WATER_LEVEL + CLEARANCE
    return centers[centers[:,2] <= z_lim]

# === メイン処理 ===
print("📥 LAS読み込み中...")
las = laspy.read(input_las)
pts_all = np.vstack([las.x, las.y, las.z]).T
cols_all = np.vstack([las.red, las.green, las.blue]).T

x_min, x_max = np.floor(pts_all[:,0].min()), np.ceil(pts_all[:,0].max())
x_centers = np.arange(x_min, x_max + slice_interval, slice_interval)

for i, x_center in enumerate(x_centers):
    x_low = x_center - slice_thickness/2
    x_high = x_center + slice_thickness/2
    mask = (pts_all[:,0] >= x_low) & (pts_all[:,0] <= x_high)
    if not np.any(mask): continue

    pts_slice = pts_all[mask]
    cols_slice = cols_all[mask]

    # === Z > 3.2 除去 & ノイズ除去（スキップなし） ===
    z_mask = pts_slice[:,2] <= Z_CUTOFF
    pts_zcut = pts_slice[z_mask]
    cols_zcut = cols_slice[z_mask]

    inlier_mask = np.ones(len(pts_zcut), dtype=bool)
    if len(pts_zcut) >= 10:
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
        try:
            inlier_mask = lof.fit_predict(pts_zcut[:, :3]) == 1
        except:
            pass
    pts_clean = pts_zcut[inlier_mask]

    # === Ray origin をY方向5分割で決定 ===
    if len(pts_clean) >= 3:
        yz = pts_clean[:,1:3]
        y_min, y_max = yz[:,0].min(), yz[:,0].max()
        y_splits = np.linspace(y_min, y_max, 6)
        y_centers = (y_splits[:-1] + y_splits[1:]) / 2
        z_median = np.median(yz[:,1])
        origins = np.array([[y, z_median] for y in y_centers])
    else:
        origins = np.empty((0,2))

    # === Raycastで緑点生成（0件でも進行） ===
    ray_pts = run_raycast_multi(pts_clean, origins)
    ray_cols = np.tile([0, 65535, 0], (len(ray_pts), 1))  # 緑

    # === 統合出力 ===
    pts_out = np.vstack([pts_slice, ray_pts])
    cols_out = np.vstack([cols_slice, ray_cols])

    out_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    out_las.header = las.header
    out_las.x, out_las.y, out_las.z = pts_out.T
    out_las.red, out_las.green, out_las.blue = cols_out.T.astype(np.uint16)
    if hasattr(las.header, 'crs') and las.header.crs:
        out_las.header.crs = las.header.crs

    out_path = os.path.join(output_dir, f"slice_x_{x_center:.2f}m_navigable.las")
    out_las.write(out_path)
    print(f"✅ [{i+1}/{len(x_centers)}] 出力: {out_path}（点数: {len(pts_out)}）")

print("🎉 全スライスの処理が完了しました。")
