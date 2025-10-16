# -*- coding: utf-8 -*-
"""
【機能】
上下LiDAR点群(LAS)を統合し、法線付きPLYを出力
------------------------------------------------------
- 入力:
    /data/0821_merged_lidar_sita.las
    /data/0821_merged_lidar_ue.las
- 出力:
    /workspace/output/1016_merged_lidar_uesita.ply
- 法線推定あり（Open3D）
- 法線の向き統一（orient_normals_*）は行わない
------------------------------------------------------
必要:
    pip install laspy open3d numpy
"""

import laspy
import numpy as np
import open3d as o3d
import os

# ===== 入出力パス =====
INPUT_SITA = r"/workspace/data/0821_merged_lidar_sita.las"
INPUT_UE   = r"/workspace/data/0821_merged_lidar_ue.las"
OUTPUT_PLY = r"/workspace/output/1016_merged_lidar_uesita.ply"

# ===== LAS読込関数 =====
def load_las_points(path):
    las = laspy.read(path)
    pts = np.vstack((las.x, las.y, las.z)).T
    print(f"✅ 読み込み完了: {os.path.basename(path)} 点数={len(pts):,}")
    return pts

# ===== 点群読み込み =====
pts_sita = load_las_points(INPUT_SITA)
pts_ue   = load_las_points(INPUT_UE)

# ===== 統合 =====
merged_points = np.vstack([pts_sita, pts_ue])
print(f"🔗 統合点数: {len(merged_points):,}")

# ===== Open3D 点群作成 =====
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(merged_points)

# ===== 法線推定 =====
print("🧭 法線推定中...")
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=1.0,  # 法線推定半径（必要に応じ調整可）
        max_nn=100
    )
)
print("✅ 法線推定完了（向き統一なし）")

# ===== 出力ディレクトリ作成 =====
os.makedirs(os.path.dirname(OUTPUT_PLY), exist_ok=True)

# ===== PLY出力 =====
o3d.io.write_point_cloud(OUTPUT_PLY, pcd, write_ascii=False)
print(f"🎉 PLY出力完了: {OUTPUT_PLY}")
