# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルを読み込み
- Z ≤ 3.0m の点群から最大連結クラスタ（航行空間）を抽出
- 航行空間クラスタの色を緑に変更
- 元の点群と結合して LAS 形式で出力
"""

import numpy as np
import open3d as o3d
import laspy

# === 入出力設定 ===
INPUT_LAS  = r"/output/0725_suidoubasi_ue.las"
OUTPUT_LAS = r"/output/0728_navi.las"
Z_LIMIT    = 3.0  # 航行空間の上限

# === LAS読み込み ===
print("📥 LAS読み込み中...")
las = laspy.read(INPUT_LAS)
xyz = np.vstack([las.x, las.y, las.z]).T
rgb = np.vstack([las.red, las.green, las.blue]).T / 65535.0

# === Open3D 点群生成（全体点群） ===
full_pcd = o3d.geometry.PointCloud()
full_pcd.points = o3d.utility.Vector3dVector(xyz)
full_pcd.colors = o3d.utility.Vector3dVector(rgb)

# === Z ≤ 3.0 の点を抽出（航行空間候補） ===
mask = xyz[:, 2] <= Z_LIMIT
filtered_xyz = xyz[mask]
filtered_rgb = rgb[mask]

filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_xyz)
filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_rgb)

# === 法線推定（必要）===
filtered_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
filtered_pcd.orient_normals_consistent_tangent_plane(50)

# === DBSCANクラスタリング（最大クラスタ）===
print("🔎 クラスタリング中...")
labels = np.array(filtered_pcd.cluster_dbscan(eps=0.6, min_points=100, print_progress=True))
valid = labels >= 0
if np.sum(valid) == 0:
    raise RuntimeError("❌ クラスタが見つかりませんでした。パラメータを調整してください。")
largest_label = np.bincount(labels[valid]).argmax()
navi_idx = np.where(labels == largest_label)[0]

# === 航行クラスタに緑色を設定 ===
green = np.tile(np.array([[0.0, 1.0, 0.0]]), (len(navi_idx), 1))
navi_pts = np.asarray(filtered_pcd.points)[navi_idx]

navi_pcd = o3d.geometry.PointCloud()
navi_pcd.points = o3d.utility.Vector3dVector(navi_pts)
navi_pcd.colors = o3d.utility.Vector3dVector(green)

# === 統合（元の点群 + 航行空間）===
combined_pcd = full_pcd + navi_pcd

# === LAS出力の準備 ===
print(f"💾 LAS出力準備中: {OUTPUT_LAS}")
combined_xyz = np.asarray(combined_pcd.points)
combined_rgb = (np.asarray(combined_pcd.colors) * 65535).astype(np.uint16)

header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = combined_xyz.min(axis=0)
header.scales = np.array([0.001, 0.001, 0.001])

out_las = laspy.LasData(header)
out_las.x, out_las.y, out_las.z = combined_xyz[:, 0], combined_xyz[:, 1], combined_xyz[:, 2]
out_las.red, out_las.green, out_las.blue = combined_rgb[:, 0], combined_rgb[:, 1], combined_rgb[:, 2]

# === 書き出し ===
out_las.write(OUTPUT_LAS)
print(f"✅ 出力完了: {OUTPUT_LAS}（点数: {len(combined_xyz)}）")
