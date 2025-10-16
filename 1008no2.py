# -*- coding: utf-8 -*-
"""
【統合処理】floor補間 → 分類（法線付きPLY出力）→ Ouster GPU-ICP整合
-----------------------------------------------------------------------
環境: DGX (CUDA 12.4, Python 3.10, Open3D-GPU 0.18.0)
手順:
  1. /output/0925_floor_sita_merged.las を読み込み Morphology補間
  2. /output/0925_lidar_sita_merged.las と統合し法線推定＋分類
  3. /output/1009_sita_classified_with_normals.ply を保存
  4. /data/realdata/*.pcap + .json を読み込み
  5. GPU Point-to-Plane ICPによる自己位置推定
-----------------------------------------------------------------------
必要ライブラリ:
 pip install numpy==1.26.4 scipy==1.13.1 laspy==2.5.4 open3d-gpu==0.18.0 ouster-sdk==0.15.1
"""

import os
import numpy as np
import cv2
import open3d as o3d
import laspy
from scipy.spatial import cKDTree
from ouster.sdk.open_source import open_source
from ouster.sdk.core import XYZLut

# ==========================================================
# 🔧 パス設定
# ==========================================================
FLOOR_LAS = "/workspace/output/0925_floor_sita_merged.las"
LIDAR_LAS = "/workspace/output/0925_lidar_sita_merged.las"
FLOOR_PLY = "/workspace/output/1009_floor_interp_only.ply"
CLASSIFIED_PLY = "/workspace/output/1009_sita_classified_with_normals.ply"
ICP_OUTPUT_DIR = "/workspace/output/icp_result"

PCAP_PATH = "/workspace/data/realdata/2022-07-06-17-32-45_OS-2-128-992048000507-1024x10-002.pcap"
JSON_PATH = "/workspace/data/realdata/2022-07-06-17-32-45_OS-2-128-992048000507-1024x10.json"

# ==========================================================
# ⚙️ パラメータ設定
# ==========================================================
voxel_size_interp = 0.05
morph_radius = 100
down_voxel_size = 0.2

normal_wall_z_max = 3.2
floor_z_max = 1.1
horizontal_threshold = 0.6
search_radius_normals = 1.0
max_neighbors_normals = 500

search_radius_z = 5.0
max_neighbors_z = 50

# ICPパラメータ
VOXEL_SIZE_ICP = 0.10
MAX_CORR_DIST = 0.5
FRAME_IDX = 500


# ==========================================================
# 🧩 1. LAS読み込み関数
# ==========================================================
def load_las(path):
    las = laspy.read(path)
    pts = np.vstack([las.x, las.y, las.z]).T
    if np.all(las.intensity == 0):
        intensity = None
        print(f"⚠ {path} の Intensity は全て0 → 無視します")
    else:
        intensity = np.array(las.intensity, dtype=np.float32)
    return pts, intensity


# ==========================================================
# 🧩 2. Morphology補間（Z中央値）
# ==========================================================
def morphology_interpolation_median(base_points, mask_fn):
    target = base_points[mask_fn(base_points)]
    if target.size == 0:
        print("⚠ 補間対象なし → スキップ")
        return np.empty((0, 3))

    min_x, min_y = target[:, 0].min(), target[:, 1].min()
    ix = np.floor((target[:, 0] - min_x) / voxel_size_interp).astype(int)
    iy = np.floor((target[:, 1] - min_y) / voxel_size_interp).astype(int)
    grid = np.zeros((ix.max() + 1, iy.max() + 1), dtype=bool)
    grid[ix, iy] = True

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * morph_radius + 1, 2 * morph_radius + 1))
    grid_closed = cv2.morphologyEx(grid.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    new_ix, new_iy = np.where(grid_closed & ~grid)
    if len(new_ix) == 0:
        print("⚠ 新規セルなし → スキップ")
        return np.empty((0, 3))

    new_xy = np.column_stack([new_ix * voxel_size_interp + min_x, new_iy * voxel_size_interp + min_y])
    tree = cKDTree(target[:, :2])
    dists, idxs = tree.query(new_xy, k=max_neighbors_z, distance_upper_bound=search_radius_z)

    new_z = np.full(len(new_xy), np.nan)
    for i in range(len(new_xy)):
        valid = np.isfinite(dists[i]) & (dists[i] < np.inf)
        if not np.any(valid):
            continue
        neighbor_z = target[idxs[i, valid], 2]
        new_z[i] = np.median(neighbor_z)

    valid = ~np.isnan(new_z)
    print(f"✅ 補間点生成: {np.count_nonzero(valid):,} 点")
    return np.column_stack([new_xy[valid], new_z[valid]])


# ==========================================================
# 🧩 3. floor補間＋分類（法線付きPLY出力）
# ==========================================================
def process_floor_and_classify():
    print("\n=== [1] floor LAS読み込み中 ===")
    floor_points, _ = load_las(FLOOR_LAS)
    interp_floor = morphology_interpolation_median(floor_points, lambda pts: pts[:, 2] <= 3.0)
    floor_completed = np.vstack([floor_points, interp_floor])

    # --- floor補間PLY出力 ---
    pcd_floor = o3d.geometry.PointCloud()
    pcd_floor.points = o3d.utility.Vector3dVector(floor_completed)
    pcd_floor.paint_uniform_color([0.0, 0.0, 1.0])
    o3d.io.write_point_cloud(FLOOR_PLY, pcd_floor)
    print(f"💾 補間後floor出力: {FLOOR_PLY} ({len(floor_completed):,} 点)")

    print("\n=== [2] lidar LAS読み込み ===")
    lidar_points, _ = load_las(LIDAR_LAS)
    all_points_final = np.vstack([floor_completed, lidar_points])

    # --- ダウンサンプリング ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points_final)
    pcd = pcd.voxel_down_sample(voxel_size=down_voxel_size)
    points = np.asarray(pcd.points)

    # --- 法線推定 ---
    print("📐 法線推定中 (GPU非対応処理)...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=search_radius_normals, max_nn=max_neighbors_normals))
    pcd.orient_normals_consistent_tangent_plane(30)
    normals = np.asarray(pcd.normals)

    # --- 分類 ---
    colors = np.zeros((len(points), 3))
    colors[:] = [1.0, 1.0, 1.0]
    colors[(normals[:, 2] < 0.6) & (points[:, 2] < normal_wall_z_max)] = [1.0, 0.0, 0.0]  # 壁
    colors[(normals[:, 2] > horizontal_threshold) & (points[:, 2] < floor_z_max)] = [0.0, 0.0, 1.0]  # 床
    colors[points[:, 2] >= normal_wall_z_max] = [1.0, 1.0, 0.0]  # ビル
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(CLASSIFIED_PLY, pcd)
    print(f"🎉 分類PLY出力完了: {CLASSIFIED_PLY} ({len(points):,} 点, 法線付き)")
    return CLASSIFIED_PLY


# ==========================================================
# 🧩 4. Ousterフレーム読込
# ==========================================================
def read_ouster_frame(pcap_path, json_path, frame_idx=0):
    print("\n=== [3] Ousterフレーム読込 ===")
    source = open_source(pcap_path, meta=[json_path])
    sensor_info = getattr(source, "sensor_info", None) or getattr(source, "metadata", None)
    if isinstance(sensor_info, list):
        sensor_info = sensor_info[0]
    xyzlut = XYZLut(sensor_info)
    scans = list(source)
    scan = scans[frame_idx][0] if isinstance(scans[frame_idx], list) else scans[frame_idx]
    xyz = xyzlut(scan)
    points = np.asarray(xyz).reshape(-1, 3)
    print(f"✅ フレーム {frame_idx} 点数: {points.shape[0]:,}")
    return points


# ==========================================================
# 🧩 5. GPU ICP実行
# ==========================================================
def run_icp_gpu(scan_pcd, map_pcd):
    print("\n=== [4] GPU ICP整合開始 ===")
    device = o3d.core.Device("CUDA:0")
    scan_tensor = o3d.t.geometry.PointCloud.from_legacy(scan_pcd, device=device)
    map_tensor = o3d.t.geometry.PointCloud.from_legacy(map_pcd, device=device)

    result = o3d.t.pipelines.registration.icp(
        source=scan_tensor,
        target=map_tensor,
        max_correspondence_distance=MAX_CORR_DIST,
        estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
    )

    T = result.transformation.cpu().numpy()
    print("✅ GPU ICP完了. 変換行列:\n", T)
    return T


# ==========================================================
# 🧩 メイン処理
# ==========================================================
def main():
    # --- Floor分類処理 ---
    map_ply = process_floor_and_classify()

    # --- Ousterフレーム読込 ---
    xyz = read_ouster_frame(PCAP_PATH, JSON_PATH, FRAME_IDX)
    os.makedirs(ICP_OUTPUT_DIR, exist_ok=True)
    scan_ply_path = os.path.join(ICP_OUTPUT_DIR, f"ouster_frame_{FRAME_IDX:04d}.ply")
    o3d.io.write_point_cloud(scan_ply_path, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz)))
    print(f"💾 スキャン点群保存: {scan_ply_path}")

    # --- 地図読み込み ---
    map_pcd = o3d.io.read_point_cloud(map_ply)
    scan_pcd = o3d.io.read_point_cloud(scan_ply_path)

    # --- GPU ICP ---
    T = run_icp_gpu(scan_pcd, map_pcd)

    # --- 結果可視化 ---
    scan_pcd.transform(T)
    map_pcd.paint_uniform_color([0.6, 0.6, 0.6])
    scan_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    o3d.visualization.draw_geometries(
        [map_pcd, scan_pcd],
        window_name="ICP Self-Localization (GPU)",
        width=1280, height=720
    )


if __name__ == "__main__":
    main()
