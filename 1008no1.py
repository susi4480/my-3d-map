# -*- coding: utf-8 -*-
"""
【機能】Ouster OS-2（.pcap＋.json）→ 法線付きPLY地図にICP整合（自己位置推定）
-----------------------------------------------------------------------
・地図は PLY（法線付き）を使用（例：/output/1008_sita_classified_with_normals.ply）
・Ouster SDK v0.15.1 対応（sensor_info / listスキャン対応）
・Point-to-Plane ICP（地図側の法線を使用）
-----------------------------------------------------------------------
必要: pip install ouster-sdk open3d laspy numpy
"""

import os
import numpy as np
import open3d as o3d
from ouster.sdk.open_source import open_source
from ouster.sdk.core import XYZLut


# ==========================================================
# 🔧 入出力設定
# ==========================================================
PCAP_PATH = r"/data/realdata/2022-07-06-17-32-45_OS-2-128-992048000507-1024x10-002.pcap"
JSON_PATH = r"/data/realdata/2022-07-06-17-32-45_OS-2-128-992048000507-1024x10.json"
MAP_PLY   = r"/output/1008_sita_classified_with_normals.ply"  # ← 法線付きPLY地図
OUTPUT_DIR = r"/lab/output"

# ICPパラメータ
VOXEL_SIZE = 0.10          # ダウンサンプリング解像度[m]
MAX_CORR_DIST = 0.5        # ICP対応点距離[m]
FRAME_IDX = 500            # 取得フレーム番号（0=最初のスキャン）
# ==========================================================


# === [1] Ousterから1フレーム点群を生成 ===
def read_ouster_frame(pcap_path, json_path, frame_idx=0):
    """Ouster .pcap + .json から指定フレームの点群(XYZ)を生成（v0.15.1対応）"""
    print("📥 Ousterデータ読み込み中...")
    # meta は list[str] で渡す
    source = open_source(pcap_path, meta=[json_path])

    # sensor_info を取得（metadataは将来削除予定のため）
    sensor_info = getattr(source, "sensor_info", None)
    if sensor_info is None:
        sensor_info = getattr(source, "metadata", None)
    if isinstance(sensor_info, list):
        sensor_info = sensor_info[0]

    xyzlut = XYZLut(sensor_info)

    # open_source() 自体が iterable
    scans = list(source)
    if not scans:
        raise RuntimeError("❌ スキャンデータが空です。PCAPとJSONの対応を確認してください。")
    if frame_idx >= len(scans):
        raise ValueError(f"指定フレーム {frame_idx} は存在しません（最大 {len(scans)-1}）")

    scan = scans[frame_idx]
    if isinstance(scan, list):  # 複数センサ対応
        scan = scan[0]

    # XYZ生成
    xyz = xyzlut(scan)
    points = np.asarray(xyz).reshape(-1, 3)
    print(f"✅ フレーム {frame_idx} 読み込み完了: {points.shape[0]:,} 点")
    return points


# === [2] スキャン点群を保存（確認用） ===
def save_scan_ply(xyz, output_dir, frame_idx):
    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, f"ouster_frame_{frame_idx:04d}.ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"💾 保存: {ply_path} ({len(xyz):,} 点)")
    return pcd


# === [3] 地図（PLY）を読み込み ===
def read_map_ply(ply_path):
    """PLY地図をOpen3D点群として読み込み（法線付き推奨）"""
    print("🗺 地図読込中...")
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"🗺 地図読込完了: {ply_path} ({len(pcd.points):,} 点)")
    if not pcd.has_normals():
        # 念のため：法線が無い場合は推定（Point-to-Plane用）
        print("⚠ 法線が含まれていません → 推定します（radius=1.0, max_nn=50）")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=50)
        )
    return pcd


# === [4] ICP（Point-to-Plane） ===
def run_icp(scan_pcd, map_pcd, voxel_size, max_corr_dist):
    """点群ICP(Point-to-Plane)で位置・姿勢を推定"""
    # ダウンサンプリング
    scan_ds = scan_pcd.voxel_down_sample(voxel_size)
    map_ds  = map_pcd.voxel_down_sample(voxel_size)

    # 安定化のため、scan側にも法線推定（推奨）
    if not scan_ds.has_normals():
        scan_ds.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
        )

    print("🧭 ICP整合中...")
    result = o3d.pipelines.registration.registration_icp(
        scan_ds, map_ds,
        max_correspondence_distance=max_corr_dist,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    print("✅ ICP完了")
    print("推定変換行列（LiDAR→Map）:")
    print(result.transformation)
    return result.transformation


# === [5] メイン処理 ===
def main():
    # 1) LiDAR 1フレーム読み込み
    xyz = read_ouster_frame(PCAP_PATH, JSON_PATH, FRAME_IDX)
    scan_pcd = save_scan_ply(xyz, OUTPUT_DIR, FRAME_IDX)

    # 2) 地図PLY読込（法線付き）
    map_pcd = read_map_ply(MAP_PLY)

    # 3) ICP
    T = run_icp(scan_pcd, map_pcd, VOXEL_SIZE, MAX_CORR_DIST)

    # 4) 可視化（地図=灰、LiDAR=赤）
    scan_pcd.transform(T)
    map_pcd.paint_uniform_color([0.6, 0.6, 0.6])
    scan_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    o3d.visualization.draw_geometries(
        [map_pcd, scan_pcd],
        window_name="ICP Self-Localization (Map=PLY+Normals)",
        width=1280, height=720
    )


if __name__ == "__main__":
    main()
