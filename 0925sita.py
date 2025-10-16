# -*- coding: utf-8 -*-
"""
【機能】
LiDAR（上下構造）LASを統合して地図PLYを作成
---------------------------------------------------------
- 入力: lidar_sita_las / lidar_ue_las フォルダ
- 統合 → ダウンサンプリング → 法線推定（向きそろえなし）
- 出力: /workspace/output/1013_lidar_map.ply
---------------------------------------------------------
ICP地図用。orient_normals_consistent_tangent_plane() は使用しない。
"""

import os
import glob
import numpy as np
import laspy
import open3d as o3d
from pyproj import CRS

# ===== 入出力 =====
lidar_dir_sita = r"/workspace/data/fulldata/lidar_sita_las/"
lidar_dir_ue   = r"/workspace/data/fulldata/lidar_ue_las/"
OUTPUT_PLY     = r"/workspace/output/1013_lidar_map.ply"

# ===== パラメータ =====
VOXEL_SIZE = 0.15        # ダウンサンプリング解像度
NORMAL_RADIUS = 1.0      # 法線推定半径
NORMAL_NN = 100           # 法線推定近傍点数

# === LASフォルダ読込 ===
def load_las_folder(folder):
    files = glob.glob(os.path.join(folder, "*.las"))
    if not files:
        print(f"⚠ {folder} に LAS ファイルが見つかりません")
        return np.empty((0, 3)), np.empty((0,))
    all_points, all_intensity = [], []
    for f in files:
        las = laspy.read(f)
        pts = np.vstack([las.x, las.y, las.z]).T
        inten = np.array(las.intensity, dtype=np.float32)
        all_points.append(pts)
        all_intensity.append(inten)
        print(f"📂 読込: {os.path.basename(f)} ({len(pts):,} 点)")
    return np.vstack(all_points), np.hstack(all_intensity)

# === メイン ===
def main():
    # LiDAR上下統合
    pts_sita, inten_sita = load_las_folder(lidar_dir_sita)
    pts_ue, inten_ue = load_las_folder(lidar_dir_ue)

    if pts_sita.size == 0 and pts_ue.size == 0:
        print("❌ 有効なLASがありません。処理終了。")
        return

    all_pts = np.vstack([pts_sita, pts_ue])
    all_inten = np.hstack([inten_sita, inten_ue])
    print(f"✅ 統合完了: {len(all_pts):,} 点")

    # ダウンサンプリング
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"📉 ダウンサンプリング後: {len(pcd.points):,} 点")

    # 法線推定（向きそろえなし）
    print("🧭 法線推定中...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS, max_nn=NORMAL_NN
        )
    )
    print("✅ 法線推定完了（向き統一なし）")

    # 出力
    o3d.io.write_point_cloud(OUTPUT_PLY, pcd)
    print(f"🎉 出力完了: {OUTPUT_PLY}")

if __name__ == "__main__":
    main()
