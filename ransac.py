# -*- coding: utf-8 -*-
"""
【機能】
/output/0821_suidoubasi_ue.las に対して RANSAC による平面検出を繰り返し実行し、
- 法線Z > 0.85 かつ Z ≤ 1.1 → 床（青）
- 法線Z < 0.3 かつ Z ≤ 3.2 → 壁（赤）
- その他 → 灰色
として分類しPLY出力する。
"""

import laspy
import numpy as np
import open3d as o3d

# === 入出力設定 ===
input_las_path = "/output/0821_suidoubasi_sita_no_color.las"
output_ply_path = "/output/0821_sita_ransac_wall_floor_color.ply"

# === RANSACパラメータ ===
distance_threshold = 0.1
ransac_n = 3
num_iterations = 1000
min_inliers = 5000
horizontal_thresh = 0.85
vertical_thresh = 0.3
floor_z_max = 1.1
wall_z_max = 3.2

# === LAS読み込み ===
print("📥 LASファイル読み込み中...")
las = laspy.read(input_las_path)
points = np.vstack([las.x, las.y, las.z]).T
print(f"✅ 点数: {len(points):,}")

# === Open3Dに変換 ===
pcd_all = o3d.geometry.PointCloud()
pcd_all.points = o3d.utility.Vector3dVector(points)

# === 色初期化（灰色）===
colors = np.full((len(points), 3), 0.5)  # 全体を灰色で初期化
remaining = pcd_all
processed_mask = np.zeros(len(points), dtype=bool)

# === 平面を繰り返し検出 ===
print("📐 RANSACによる平面検出中...")
while True:
    plane_model, inliers = remaining.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    if len(inliers) < min_inliers:
        print(f"ℹ️ インライア数が少ないため終了: {len(inliers)}点")
        break

    [a, b, c, d] = plane_model
    normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    print(f"→ 平面法線: ({a:.3f}, {b:.3f}, {c:.3f}) | 点数: {len(inliers)}")

    # インライアのインデックスを取得（全体点群に対して）
    remaining_indices = np.asarray(remaining.points)
    full_indices = np.where(~processed_mask)[0]
    inlier_indices = full_indices[inliers]
    inlier_points = points[inlier_indices]

    # === 分類条件 ===
    if abs(normal[2]) > horizontal_thresh:
        z_mask = inlier_points[:, 2] <= floor_z_max
        if np.sum(z_mask) > 0:
            colors[inlier_indices[z_mask]] = [0, 0, 1]  # 青：床
            print(f"🟦 水平面（Z ≤ {floor_z_max}）として分類: {np.sum(z_mask)}点")
        else:
            print("🔹 水平だがZ条件を満たさないため分類せず")
    elif abs(normal[2]) < vertical_thresh:
        z_mask = inlier_points[:, 2] <= wall_z_max
        if np.sum(z_mask) > 0:
            colors[inlier_indices[z_mask]] = [1, 0, 0]  # 赤：壁
            print(f"🟥 垂直面（Z ≤ {wall_z_max}）として分類: {np.sum(z_mask)}点")
        else:
            print("🔹 垂直だがZ条件を満たさないため分類せず")
    else:
        print("⚪ その他の面（分類せず）")

    # 処理済みマスク更新
    processed_mask[inlier_indices] = True
    remaining = remaining.select_by_index(inliers, invert=True)

    if len(remaining.points) < min_inliers:
        print("✅ 残り点数が少ないため終了")
        break

# === 出力 ===
print("💾 PLY出力中...")
pcd_all.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(output_ply_path, pcd_all)
print(f"🎉 完了: {output_ply_path}")
