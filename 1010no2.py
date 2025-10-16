# -*- coding: utf-8 -*-
"""
【黒影対応ICP（GPU最適化：Cupoch / フォールバック：Open3D CPU）】
Ouster OS-2（.pcap + .json）→ 欠測(黒)除外 + FOVクロップ + 地図に逐次ICP整合（自己位置推定）
- Cupoch が利用できれば GPU Point-to-Plane ICP（高速）
- Cupoch が無ければ Open3D CPU ICP に自動フォールバック
- 地図に法線が無い場合は自動で推定
- 地図外（重なり弱）期間は SKIP、初めて地図内に入ったら自動開始（再ロック）
- PLY保存（一定間隔）、姿勢CSV出力
"""

import os
import csv
import math
import numpy as np
import open3d as o3d
from ouster.sdk.open_source import open_source
from ouster.sdk.core import XYZLut, ChanField, SensorInfo

# ==========================================================
# 🔧 入出力設定（ここを書き換えて使う）
# ==========================================================
PCAP_PATH  = r"/workspace/data/realdata/2022-07-06-17-32-45_OS-2-128-992048000507-1024x10-002.pcap"
JSON_PATH  = r"/workspace/data/realdata/2022-07-06-17-32-45_OS-2-128-992048000507-1024x10.json"
MAP_PATH   = r"/workspace/output/1010_sita_classified_normals_type2_free.ply"
OUTPUT_DIR = r"/workspace/output/icp_run"

# ==========================================================
# 🔧 パラメータ（必要に応じて調整）
# ==========================================================
VOXEL_SIZE     = 0.15     # ダウンサンプリング解像度[m]
MAX_CORR_DIST  = 0.50     # ICP対応点距離[m]
MAX_ITER       = 60       # ICP反復回数
MIN_RANGE      = 1.0      # 距離フィルタ最小[m]
MAX_RANGE      = 120.0    # 距離フィルタ最大[m]（0で無効）
FOV_H_DEG      = 70.0     # 水平FOVの半角[deg]（±FOV_H_DEG）
FOV_V_DEG      = 20.0     # 垂直FOVの半角[deg]（±FOV_V_DEG）
MIN_FITNESS    = 0.30     # 地図内とみなす最小一致度
MAX_RMSE       = 0.25     # 許容RMSE上限
SAVE_EVERY     = 10       # 何フレームごとに整合後PLYを保存するか
USE_GPU        = True     # 可能ならGPU（Cupoch）を使う
MAP_NORMAL_RAD = 1.0      # 地図の法線推定半径[m]
MAP_NORMAL_NN  = 50       # 地図の法線推定 近傍点数
# ==========================================================


# =============== 共通ユーティリティ ===============
def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0],[sy, cy, 0],[0, 0, 1]])
    Ry = np.array([[cp, 0, sp],[0, 1, 0],[-sp, 0, cp]])
    Rx = np.array([[1, 0, 0],[0, cr, -sr],[0, sr, cr]])
    return Rz @ Ry @ Rx

def matrix_to_quaternion(R):
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    else:
        S = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
    return (qx, qy, qz, qw)

def scan_xyz_with_masks(xyz_raw, rng_raw):
    """欠測(黒=range=0)除外 + 距離 + FOVクロップ"""
    valid = (rng_raw > 0)
    xyz = np.asarray(xyz_raw)
    pts = xyz.reshape(-1, 3)
    valid = valid.reshape(-1)

    # 距離フィルタ
    d = np.linalg.norm(pts, axis=1)
    valid &= (d > MIN_RANGE)
    if MAX_RANGE > 0:
        valid &= (d < MAX_RANGE)

    pts = pts[valid]
    if pts.shape[0] == 0:
        return pts

    # FOV（水平・垂直）
    az = np.arctan2(pts[:,1], pts[:,0])
    el = np.arctan2(pts[:,2], np.sqrt(pts[:,0]**2 + pts[:,1]**2))
    h_ok = np.abs(az) <= np.deg2rad(FOV_H_DEG)
    v_ok = np.abs(el) <= np.deg2rad(FOV_V_DEG)
    return pts[h_ok & v_ok]


# =============== ICP実装：Open3D CPU（フォールバック） ===============
def cpu_icp_o3d(scan_ds, map_ds, init_T):
    result = o3d.pipelines.registration.registration_icp(
        scan_ds, map_ds,
        max_correspondence_distance=MAX_CORR_DIST,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=MAX_ITER)
    )
    return result.transformation, float(result.fitness), float(result.inlier_rmse)


# =============== ICP実装：Cupoch GPU（高速） ===============
_HAS_CUPOCH = False
try:
    if USE_GPU:
        import cupoch as cph
        _HAS_CUPOCH = True
except Exception:
    _HAS_CUPOCH = False

def o3d_to_cupoch_pcd(o3d_pcd, estimate_normals_if_missing=True):
    """Open3D PointCloud → Cupoch PointCloud（GPU化）"""
    import cupoch as cph
    cpcd = cph.geometry.PointCloud()
    pts = np.asarray(o3d_pcd.points)
    cpcd.points = cph.utility.Vector3fVector(pts.astype(np.float32))
    if o3d_pcd.has_normals():
        nrm = np.asarray(o3d_pcd.normals)
        cpcd.normals = cph.utility.Vector3fVector(nrm.astype(np.float32))
    elif estimate_normals_if_missing:
        # Cupoch側で推定
        cpcd.estimate_normals(cph.geometry.KDTreeSearchParamHybrid(radius=MAP_NORMAL_RAD, max_nn=MAP_NORMAL_NN))
    return cpcd

def try_gpu_icp_cupoch(scan_ds, map_ds, init_T):
    """CupochによるGPU Point-to-Plane ICP"""
    import cupoch as cph
    # Open3D → Cupoch 変換（GPUに載せ替え）
    scan_t = o3d_to_cupoch_pcd(scan_ds, estimate_normals_if_missing=False)
    map_t  = o3d_to_cupoch_pcd(map_ds,  estimate_normals_if_missing=True)

    reg = cph.registration.registration_icp(
        scan_t, map_t,
        MAX_CORR_DIST, init_T.astype(np.float32),
        cph.registration.TransformationEstimationPointToPlane(),
        cph.registration.ICPConvergenceCriteria(max_iteration=MAX_ITER)
    )
    return reg.transformation.astype(np.float64), float(reg.fitness), float(reg.inlier_rmse)


# =============== メイン処理 ===============
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, "icp_trajectory.csv")
    out_ply = os.path.join(OUTPUT_DIR, "icp_registered_frames")
    os.makedirs(out_ply, exist_ok=True)

    # 地図読み込み（法線が無ければ推定）
    print("🗺 地図読み込み中...")
    map_pcd = o3d.io.read_point_cloud(MAP_PATH)
    if not map_pcd.has_normals():
        print("⚠ 地図に法線なし → 推定します")
        map_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=MAP_NORMAL_RAD, max_nn=MAP_NORMAL_NN))
    print(f"✅ 地図点数: {len(map_pcd.points):,}")

    # LiDARメタ
    print("📄 LiDARメタ情報読み込み中...")
    with open(JSON_PATH, "r") as f:
        sensor_info = SensorInfo(f.read())
    xyzlut = XYZLut(sensor_info, use_extrinsics=False)

    # PCAP
    print("📡 PCAPストリーム初期化中...")
    source = open_source(PCAP_PATH)
    print("✅ open_source OK")

    if _HAS_CUPOCH:
        print("🚀 Cupoch(GPU) 利用可能 → 高速ICPで実行します")
    else:
        print("⚠ Cupoch(GPU) 利用不可 → Open3D(CPU) にフォールバックします")

    # 姿勢推定ループ
    T_global = np.eye(4)
    frame_idx = 0
    icp_started = False  # 地図内突入済みか

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame","status","fitness","rmse","tx","ty","tz","qx","qy","qz","qw"])

    for scans in source:
        # 複数センサに対応（通常は1）
        for scan in (scans if isinstance(scans, list) else [scans]):
            xyz_raw = xyzlut(scan)                      # (H,W,3)
            rng_raw = scan.field(ChanField.RANGE)       # (H,W)
            pts = scan_xyz_with_masks(xyz_raw, rng_raw) # 欠測/距離/FOV 除外
            if pts.shape[0] == 0:
                continue

            # Open3D点群 & ダウンサンプリング
            scan_pcd = o3d.geometry.PointCloud()
            scan_pcd.points = o3d.utility.Vector3dVector(pts)
            scan_ds = scan_pcd.voxel_down_sample(VOXEL_SIZE)

            # === ICP ===
            if _HAS_CUPOCH:
                try:
                    T_new, fitness, rmse = try_gpu_icp_cupoch(scan_ds, map_pcd, T_global)
                except Exception as e:
                    print(f"⚠ Cupoch ICP失敗（{e}）→ CPUにフォールバック")
                    T_new, fitness, rmse = cpu_icp_o3d(scan_ds, map_pcd, T_global)
            else:
                T_new, fitness, rmse = cpu_icp_o3d(scan_ds, map_pcd, T_global)

            # === スキップ or 始動判定 ===
            if fitness < MIN_FITNESS or rmse > MAX_RMSE:
                if not icp_started:
                    print(f"Frame {frame_idx:04d}: 地図外 (skip)")
                else:
                    print(f"Frame {frame_idx:04d}: ICP失敗 → 前回姿勢維持")
                with open(out_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([frame_idx, "SKIP", fitness, rmse,
                                *T_global[:3,3], *matrix_to_quaternion(T_global[:3,:3])])
                frame_idx += 1
                continue

            if not icp_started:
                print(f"🧭 地図内進入検出（frame {frame_idx}） → ICP追跡開始")
                icp_started = True

            # === 姿勢更新 & ロギング ===
            T_global = T_new
            R, t = T_global[:3,:3], T_global[:3,3]
            q = matrix_to_quaternion(R)
            print(f"Frame {frame_idx:04d}: fitness={fitness:.3f}, rmse={rmse:.3f}")

            with open(out_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([frame_idx, "OK", fitness, rmse, t[0], t[1], t[2], *q])

            # === 整合後PLYの保存（間引き）
            if frame_idx % SAVE_EVERY == 0:
                scan_save = o3d.geometry.PointCloud(scan_ds)  # コピー
                scan_save.transform(T_global)
                ply_path = os.path.join(out_ply, f"frame_{frame_idx:05d}_registered.ply")
                o3d.io.write_point_cloud(ply_path, scan_save)

            frame_idx += 1

    print("✅ 全フレーム処理完了")
    print(f"📄 出力CSV: {out_csv}")
    print(f"📂 整合済みPLY: {out_ply}")


if __name__ == "__main__":
    main()
