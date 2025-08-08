# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルをX方向に60cm幅スライス（±30cm）
- スライス間隔は50cm（前後10cmオーバーラップ）
- 各スライス内の X=+0.2, +0.4m 位置でY方向5分割 → Rayを計10点から放出
- Z > 3.2 除去＋ノイズ除去（LOF）後にRay処理
- 除去点を戻して、航行可能点（緑）を含む全点を出力
"""

import os
import math
import numpy as np
import laspy
from scipy.spatial import cKDTree
from sklearn.neighbors import LocalOutlierFactor

# === 入出力設定 ===
input_las = "/data/0704_suidoubasi_ue.las"
output_dir = "/output/slice_area_navigation_overlap/"
os.makedirs(output_dir, exist_ok=True)

# === スライスパラメータ ===
slice_width = 0.6     # スライス幅（60cm）
slice_step = 0.5      # スライス間隔（50cm）

# === Raycastパラメータ ===
Z_CUTOFF = 3.2
WATER_LEVEL = 3.5
CLEARANCE = 1.0
VOXEL_SIZE = 0.2
N_RAYS = 720
RAY_LENGTH = 20.0
STEP = 0.05
DIST_THRESH = 0.1
SAFETY_DIST = 1.0

def run_raycast_multi(pts, origins):
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
            d_hit = steps[i]
            d_safe = max(d_hit - SAFETY_DIST, 0)
            dy, dz = dirs[j]
            y_s, z_s = origin[1:] + np.array([dy, dz]) * d_safe
            _, ii = tree.query([origin[0], y_s, z_s][1:])
            x_s = origin[0]
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

# === LAS読み込み ===
print("📥 LAS読み込み中...")
las = laspy.read(input_las)
pts_all = np.vstack([las.x, las.y, las.z]).T
cols_all = np.vstack([las.red, las.green, las.blue]).T

x_min, x_max = np.floor(pts_all[:,0].min()), np.ceil(pts_all[:,0].max())
x_centers = np.arange(x_min, x_max + slice_step, slice_step)

for i, x_center in enumerate(x_centers):
    x_low = x_center - slice_width/2
    x_high = x_center + slice_width/2
    mask = (pts_all[:,0] >= x_low) & (pts_all[:,0] <= x_high)
    if not np.any(mask): continue

    pts_slice = pts_all[mask]
    cols_slice = cols_all[mask]

    z_mask = pts_slice[:,2] <= Z_CUTOFF
    pts_zcut = pts_slice[z_mask]
    cols_zcut = cols_slice[z_mask]

    if len(pts_zcut) < 20:
        print(f"⏭ スライス {i} 点数不足")
        pts_out = np.vstack([pts_slice])
        cols_out = np.vstack([cols_slice])
    else:
        lof = LocalOutlierFactor(n_neighbors=min(20, len(pts_zcut)-1), contamination=0.02)
        inlier_mask = lof.fit_predict(pts_zcut[:, :3]) == 1
        pts_clean = pts_zcut[inlier_mask]

        if len(pts_clean) < 20:
            print(f"⏭ スライス {i} ノイズ除去後点数不足")
            pts_out = np.vstack([pts_slice])
            cols_out = np.vstack([cols_slice])
        else:
            ray_origins = []
            for offset in [0.2, 0.4]:
                x_r = x_low + offset
                y_min, y_max = pts_clean[:,1].min(), pts_clean[:,1].max()
                y_splits = np.linspace(y_min, y_max, 6)
                y_centers = (y_splits[:-1] + y_splits[1:]) / 2
                z_median = np.median(pts_clean[:,2])
                for y in y_centers:
                    ray_origins.append([x_r, y, z_median])

            ray_pts = run_raycast_multi(pts_clean, np.array(ray_origins))
            ray_cols = np.tile([0, 65535, 0], (len(ray_pts), 1))
            pts_out = np.vstack([pts_slice, ray_pts])
            cols_out = np.vstack([cols_slice, ray_cols])

    out_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    out_las.header = las.header
    out_las.x, out_las.y, out_las.z = pts_out.T
    out_las.red, out_las.green, out_las.blue = cols_out.T.astype(np.uint16)
    if hasattr(las.header, 'crs') and las.header.crs:
        out_las.header.crs = las.header.crs

    out_path = os.path.join(output_dir, f"slice_x_{x_center:.2f}m_overlap.las")
    out_las.write(out_path)
    print(f"✅ [{i+1}/{len(x_centers)}] 出力: {out_path}（点数: {len(pts_out)}）")

print("🎉 全スライス（オーバーラップ）の処理が完了しました。")
