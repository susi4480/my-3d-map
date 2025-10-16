# -*- coding: utf-8 -*-
"""
0630no1.py（オフセット自動設定版）
【機能】
- 川底の点群を2Dに変換しMorphology補間
- 最近傍Zを使って3D復元
- UTM座標を保持してLAS保存
- LASヘッダーのoffsetを点群の最小値に設定
"""

import os
import glob
import numpy as np
from pyproj import Transformer, CRS
from skimage.morphology import binary_closing, disk
import laspy
from scipy.spatial import cKDTree

# === 設定 ===
input_dir = "/workspace/fulldata/floor_sita_xyz/"
output_las = "/workspace/output/0916_suidoubasi_floor_sita_offset.las"
voxel_size = 0.25
z_upper_limit = 3.0
morph_radius = 20

# === XYZ読み込み（NaN・不正行を除去）===
def load_xyz_files(directory):
    all_points = []
    files = glob.glob(os.path.join(directory, "*.xyz"))
    for f in files:
        try:
            data = np.genfromtxt(f, dtype=float)
            if data.ndim == 1 and data.size == 3:
                data = data.reshape(1, 3)
            elif data.ndim != 2 or data.shape[1] != 3:
                print(f"⚠ 無効なファイル形式: {f}（shape={data.shape}）")
                continue
            data = data[~np.isnan(data).any(axis=1)]  # NaN除去
            all_points.append(data)
        except Exception as e:
            print(f"⚠ 読み込み失敗: {f} → {e}")
    if not all_points:
        raise RuntimeError("❌ 有効な .xyz ファイルが読み込めませんでした")
    return np.vstack(all_points)

def main():
    # === [1] 点群読み込み ===
    floor_points = load_xyz_files(input_dir)
    print(f"✅ 元の点数: {len(floor_points):,}")

    # === [2] 緯度経度 → UTMに変換 ===
    transformer = Transformer.from_crs("epsg:4326", "epsg:32654", always_xy=True)
    x_utm, y_utm = transformer.transform(floor_points[:, 1], floor_points[:, 0])
    points_utm = np.column_stack((x_utm, y_utm, floor_points[:, 2]))

    # === [3] Z < z_upper_limit の点だけ抽出して 2Dグリッド化 ===
    mask = points_utm[:, 2] <= z_upper_limit
    grid_points = points_utm[mask]

    min_x, min_y = grid_points[:, 0].min(), grid_points[:, 1].min()
    ix = np.floor((grid_points[:, 0] - min_x) / voxel_size).astype(int)
    iy = np.floor((grid_points[:, 1] - min_y) / voxel_size).astype(int)

    if np.any(ix < 0) or np.any(iy < 0):
        raise RuntimeError("❌ グリッドインデックスに無効な値があります")

    grid_shape = (ix.max() + 1, iy.max() + 1)
    grid = np.zeros(grid_shape, dtype=bool)
    grid[ix, iy] = True

    # === [4] Morphology補間（クロージング）===
    grid_closed = binary_closing(grid, footprint=disk(morph_radius))

    # === [5] 新たに追加された点を抽出 ===
    new_mask = (grid_closed & ~grid)
    new_ix, new_iy = np.where(new_mask)
    new_x = new_ix * voxel_size + min_x
    new_y = new_iy * voxel_size + min_y
    new_xy = np.column_stack((new_x, new_y))

    # === [6] 最近傍からZを補完 ===
    tree = cKDTree(grid_points[:, :2])
    dists, idxs = tree.query(new_xy, k=1)
    new_z = grid_points[idxs, 2]
    new_points = np.column_stack((new_x, new_y, new_z))
    print(f"✅ 補間点数（2D）: {len(new_points):,}")

    # === [7] 元の点群とマージ ===
    merged_points = np.vstack([points_utm, new_points])
    print(f"📦 合計点数: {len(merged_points):,}")

    # === [8] LAS保存（CRS付き, オフセット自動計算）===
    header = laspy.LasHeader(point_format=3, version="1.2")

    # 🔹 オフセットを点群の最小値に設定
    header.offsets = merged_points.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])  # 1mm精度
    header.add_crs(CRS.from_epsg(32654))

    las = laspy.LasData(header)
    las.x = merged_points[:, 0]
    las.y = merged_points[:, 1]
    las.z = merged_points[:, 2]
    las.write(output_las)

    print(f"🎉 LAS出力完了: {output_las}")

if __name__ == "__main__":
    main()
