# -*- coding: utf-8 -*-
"""
【機能】ICPによる自己位置推定（raycastデータを用いたスキャンマッチング）
----------------------------------------------------------------------
- 地図LAS（基準点群）と各クエリLASを読み込み
- ICPで最適な剛体変換(R, t)を推定
- 初期姿勢はidentity（乱数姿勢からのズレを回収）
- 推定姿勢・対応スコアを出力
----------------------------------------------------------------------
出力:
  /output/1006_icp_results/
      icp_result_0000.txt（姿勢行列・スコア）
      aligned_0000.ply（整列結果の可視化用）
"""

import os
import numpy as np
import open3d as o3d
import laspy

# ====== 入出力 ======
MAP_LAS_PATH = "/output/0925_sita_merged_white.las"
QUERY_DIR    = "/output/1006_seq_query_world"
OUT_DIR      = "/output/1006_icp_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ====== ICPパラメータ ======
VOXEL_SIZE = 0.1
MAX_ITER   = 200
THRESHOLD  = 1.0  # 最近傍距離の最大許容[m]

# ====== LAS読み込み関数 ======
def read_las_to_o3d(path):
    las = laspy.read(path)
    pts = np.vstack([las.x, las.y, las.z]).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

# ====== 地図点群読み込み ======
print("📥 地図点群読み込み中...")
map_pcd = read_las_to_o3d(MAP_LAS_PATH)
map_pcd = map_pcd.voxel_down_sample(VOXEL_SIZE)
map_pcd.estimate_normals()
print(f"✅ 地図点数: {len(map_pcd.points):,}")

# ====== クエリ群を処理 ======
query_files = sorted([f for f in os.listdir(QUERY_DIR) if f.endswith(".las")])
print(f"📂 クエリ数: {len(query_files)}")

for i, fname in enumerate(query_files):
    print(f"\n🔹 [{i+1}/{len(query_files)}] {fname}")
    qpath = os.path.join(QUERY_DIR, fname)
    query_pcd = read_las_to_o3d(qpath)
    query_pcd = query_pcd.voxel_down_sample(VOXEL_SIZE)
    query_pcd.estimate_normals()

    # === ICP ===
    result = o3d.pipelines.registration.registration_icp(
        query_pcd, map_pcd, THRESHOLD,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=MAX_ITER)
    )

    print(f"  📈 Fitness: {result.fitness:.3f}, RMSE: {result.inlier_rmse:.3f}")
    print("  R|t:\n", result.transformation)

    # === 結果保存 ===
    np.savetxt(os.path.join(OUT_DIR, f"icp_result_{i:04d}.txt"), result.transformation, fmt="%.6f")

    aligned = query_pcd.transform(result.transformation)
    o3d.io.write_point_cloud(os.path.join(OUT_DIR, f"aligned_{i:04d}.ply"), aligned)

print("\n✅ すべてのICPマッチング完了！")
