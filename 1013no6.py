# -*- coding: utf-8 -*-
"""
【黒影対応ICP：初期位置自動合わせ(2段構え) + 逐次追従】
- 初期合わせ①: FPFH + Fast Global Registration (FGR)
- 初期合わせ②: ヨー角スイープ + 粗ICP（FGR失敗時のフォールバック）
- Cupoch(GPU) があればGPU Point-to-Plane ICP、無ければOpen3D(CPU)
- 欠測(黒=range=0)除外 + 距離フィルタのみ（FOV制限なし）
- 地図に法線が無ければ自動推定
- 地図外はSKIP、ロスト時に初期合わせを再度試行
- 整合後PLYを間引き保存、姿勢CSV出力

入力:
  PCAP_PATH, JSON_PATH, MAP_PATH
出力:
  OUTPUT_DIR / icp_trajectory.csv
  OUTPUT_DIR / icp_registered_frames / frame_XXXXX_registered.ply
"""

import os
import csv
import math
import numpy as np
import open3d as o3d
from ouster.sdk.open_source import open_source
from ouster.sdk.core import XYZLut, ChanField, SensorInfo

# ==========================================================
# 🔧 入出力設定
# ==========================================================
PCAP_PATH  = r"/workspace/data/realdata/2022-07-06-17-32-45_OS-2-128-992048000507-1024x10-002.pcap"
JSON_PATH  = r"/workspace/data/realdata/2022-07-06-17-32-45_OS-2-128-992048000507-1024x10.json"
MAP_PATH   = r"/workspace/output/1013_lidar_map.ply"   # 法線(nx,ny,nz)付きの地図PLYを推奨
OUTPUT_DIR = r"/workspace/output/icp_run"              # 出力先フォルダ

# ==========================================================
# ⚙️ 逐次ICP（追従）パラメータ
# ==========================================================
VOXEL_SIZE     = 0.15     # ダウンサンプリング解像度[m]（追従用）
MAX_CORR_DIST  = 0.50     # ICP対応点距離[m]
MAX_ITER       = 60       # ICP反復回数
MIN_RANGE      = 1.0      # 距離フィルタ最小[m]
MAX_RANGE      = 120.0    # 距離フィルタ最大[m]（0で無効）
MIN_FITNESS    = 0.30     # 地図内とみなす最小一致度
MAX_RMSE       = 0.25     # 許容RMSE上限
SAVE_EVERY     = 10       # 何フレームごとに整合後PLYを保存するか
USE_GPU        = True     # 可能ならGPU（Cupoch）を使う
MAP_NORMAL_RAD = 1.0      # 地図の法線推定半径[m]
MAP_NORMAL_NN  = 50       # 地図の法線推定 近傍点数

# ==========================================================
# 🌎 初期合わせ（グローバル）パラメータ
# ==========================================================
# 1) FGR（特徴ベース）
FGR_VOXEL              = 0.60   # FGR用の粗ダウンサンプル[m]
FGR_MAX_CORR_DIST      = FGR_VOXEL * 2.5
FGR_ITER               = 1000
FGR_ACCEPT_FITNESS_MIN = 0.10   # FGRを採用する最小Fitness（環境により要調整）

# 2) ヨー角スイープ + 粗ICP
YAW_SWEEP_STEP_DEG     = 15     # 0..360 の刻み[deg]
COARSE_VOXEL           = 0.40   # 粗ICP用ダウンサンプル[m]
COARSE_CORR_DIST       = 2.0    # 対応距離[m]
COARSE_MAX_ITER        = 15     # 反復回数
SWEEP_ACCEPT_FIT_MIN   = 0.12   # 採用する最小Fitness（環境により要調整）

# 3) いつグローバル初期合わせを走らせるか
SKIP_BEFORE_GLOBAL_INIT = 3     # 連続SKIPがこの回数に達したら初期合わせを試行
RETRY_GLOBAL_ON_RELOST  = True  # 追従中にロストしたら再び初期合わせを試みる


# ==========================================================
# 🧩 ユーティリティ
# ==========================================================
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
    """欠測(黒=range=0)除外 + 距離フィルタのみ（FOV制限なし）"""
    valid = (rng_raw > 0)
    xyz = np.asarray(xyz_raw)
    pts = xyz.reshape(-1, 3)
    valid = valid.reshape(-1)

    # 距離フィルタ
    d = np.linalg.norm(pts, axis=1)
    valid &= (d > MIN_RANGE)
    if MAX_RANGE > 0:
        valid &= (d < MAX_RANGE)

    return pts[valid]

# ==========================================================
# 💻 逐次ICP：Open3D CPU
# ==========================================================
def cpu_icp_o3d(scan_ds, map_ds, init_T):
    result = o3d.pipelines.registration.registration_icp(
        scan_ds, map_ds,
        max_correspondence_distance=MAX_CORR_DIST,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=MAX_ITER)
    )
    return result.transformation, float(result.fitness), float(result.inlier_rmse)

# ==========================================================
# ⚡ 逐次ICP：Cupoch GPU（自動判定）
# ==========================================================
_HAS_CUPOCH = False
try:
    if USE_GPU:
        import cupoch as cph
        _HAS_CUPOCH = True
except Exception:
    _HAS_CUPOCH = False

def o3d_to_cupoch_pcd(o3d_pcd, estimate_normals_if_missing=True):
    import cupoch as cph
    cpcd = cph.geometry.PointCloud()
    pts = np.asarray(o3d_pcd.points)
    cpcd.points = cph.utility.Vector3fVector(pts.astype(np.float32))
    if o3d_pcd.has_normals():
        nrm = np.asarray(o3d_pcd.normals)
        cpcd.normals = cph.utility.Vector3fVector(nrm.astype(np.float32))
    elif estimate_normals_if_missing:
        cpcd.estimate_normals(cph.geometry.KDTreeSearchParamHybrid(radius=MAP_NORMAL_RAD, max_nn=MAP_NORMAL_NN))
    return cpcd

def try_gpu_icp_cupoch(scan_ds, map_ds, init_T):
    import cupoch as cph
    scan_t = o3d_to_cupoch_pcd(scan_ds, estimate_normals_if_missing=False)
    map_t  = o3d_to_cupoch_pcd(map_ds,  estimate_normals_if_missing=True)
    reg = cph.registration.registration_icp(
        scan_t, map_t,
        MAX_CORR_DIST, init_T.astype(np.float32),
        cph.registration.TransformationEstimationPointToPlane(),
        cph.registration.ICPConvergenceCriteria(max_iteration=MAX_ITER)
    )
    return reg.transformation.astype(np.float64), float(reg.fitness), float(reg.inlier_rmse)

# ==========================================================
# 🧭 初期合わせ①：FGR（FPFH特徴）
# ==========================================================
def preprocess_for_feats(pcd, voxel):
    pcd_ds = pcd.voxel_down_sample(voxel)
    pcd_ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3.0, max_nn=60)
    )
    radius_feat = voxel * 5.0
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_ds,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feat, max_nn=100)
    )
    return pcd_ds, fpfh

def try_global_init_fgr(scan_pcd, map_pcd, voxel=FGR_VOXEL):
    scan_ds, scan_fpfh = preprocess_for_feats(scan_pcd, voxel)
    map_ds,  map_fpfh  = preprocess_for_feats(map_pcd,  voxel)
    option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=FGR_MAX_CORR_DIST,
        iteration_number=FGR_ITER
    )
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        scan_ds, map_ds, scan_fpfh, map_fpfh, option
    )
    return result.transformation, float(result.fitness), float(result.inlier_rmse)

# ==========================================================
# 🧭 初期合わせ②：ヨー角スイープ＋粗ICP
# ==========================================================
def yaw_matrix(yaw_rad):
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    Rz = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    T  = np.eye(4); T[:3,:3] = Rz
    return T

def coarse_icp_once(scan_pcd, map_pcd, init_T, corr=COARSE_CORR_DIST, iters=COARSE_MAX_ITER):
    res = o3d.pipelines.registration.registration_icp(
        scan_pcd, map_pcd, corr, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters)
    )
    return res.transformation, float(res.fitness), float(res.inlier_rmse)

def try_yaw_sweep_init(scan_pcd, map_pcd,
                       yaw_step_deg=YAW_SWEEP_STEP_DEG,
                       coarse_voxel=COARSE_VOXEL,
                       corr=COARSE_CORR_DIST,
                       iters=COARSE_MAX_ITER):
    # 速くするため粗ダウンサンプル
    scan_c = scan_pcd.voxel_down_sample(coarse_voxel)
    map_c  = map_pcd.voxel_down_sample(coarse_voxel)
    best_T, best_fit, best_rmse = None, -1.0, 1e9
    for deg in range(0, 360, yaw_step_deg):
        Tinit = yaw_matrix(np.deg2rad(deg))
        T, fit, rmse = coarse_icp_once(scan_c, map_c, Tinit, corr=corr, iters=iters)
        if fit > best_fit or (fit == best_fit and rmse < best_rmse):
            best_T, best_fit, best_rmse = T, fit, rmse
    return best_T, best_fit, best_rmse

# ==========================================================
# 🚀 メイン
# ==========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, "icp_trajectory.csv")
    out_ply = os.path.join(OUTPUT_DIR, "icp_registered_frames")
    os.makedirs(out_ply, exist_ok=True)

    # 地図読み込み
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

    # PCAPストリーム
    print("📡 PCAPストリーム初期化中...")
    source = open_source(PCAP_PATH)
    print("✅ open_source OK")

    if _HAS_CUPOCH:
        print("🚀 Cupoch(GPU) 利用可能 → 高速ICPで追従します")
    else:
        print("⚠ Cupoch(GPU) 利用不可 → Open3D(CPU) にフォールバックします")

    # 状態
    T_global = np.eye(4)
    frame_idx = 0
    icp_started = False
    skip_count = 0

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame","status","fitness","rmse","tx","ty","tz","qx","qy","qz","qw"])

    for scans in source:
        for scan in (scans if isinstance(scans, list) else [scans]):
            xyz_raw = xyzlut(scan)                      # (H,W,3)
            rng_raw = scan.field(ChanField.RANGE)       # (H,W)
            pts = scan_xyz_with_masks(xyz_raw, rng_raw) # 欠測/距離フィルタのみ
            if pts.shape[0] == 0:
                continue

            # O3D点群 & 追従用ダウンサンプル
            scan_pcd = o3d.geometry.PointCloud()
            scan_pcd.points = o3d.utility.Vector3dVector(pts)
            scan_ds = scan_pcd.voxel_down_sample(VOXEL_SIZE)

            # ---- まだ地図に入れていない？ → 初期合わせを試す ----
            tried_global = False
            if not icp_started and skip_count >= SKIP_BEFORE_GLOBAL_INIT:
                tried_global = True
                print("🧭 初期合わせ: FGRを試行中...")
                T0, fit0, rmse0 = try_global_init_fgr(scan_pcd, map_pcd, voxel=FGR_VOXEL)
                if fit0 >= FGR_ACCEPT_FITNESS_MIN:
                    T_global = T0
                    print(f"✅ FGR成功: fitness={fit0:.3f}, rmse={rmse0:.3f}")
                    skip_count = 0
                else:
                    print("⚠ FGR失敗 → ヨー角スイープ+粗ICPへ")
                    T1, fit1, rmse1 = try_yaw_sweep_init(scan_pcd, map_pcd)
                    if T1 is not None and fit1 >= SWEEP_ACCEPT_FIT_MIN:
                        T_global = T1
                        print(f"✅ YawSweep成功: fitness={fit1:.3f}, rmse={rmse1:.3f}")
                        skip_count = 0
                    else:
                        print("❌ 初期合わせに失敗（次フレームで再試行）")

            # ---- 逐次ICP（追従） ----
            if _HAS_CUPOCH:
                try:
                    T_new, fitness, rmse = try_gpu_icp_cupoch(scan_ds, map_pcd, T_global)
                except Exception as e:
                    print(f"⚠ Cupoch ICP失敗（{e}）→ CPUにフォールバック")
                    T_new, fitness, rmse = cpu_icp_o3d(scan_ds, map_pcd, T_global)
            else:
                T_new, fitness, rmse = cpu_icp_o3d(scan_ds, map_pcd, T_global)

            # ---- 成否判定 ----
            if fitness < MIN_FITNESS or rmse > MAX_RMSE:
                status = "SKIP" if not icp_started else "REJECT"
                print(f"Frame {frame_idx:04d}: {status} (fit={fitness:.3f}, rmse={rmse:.3f})")
                with open(out_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([frame_idx, status, fitness, rmse,
                                *T_global[:3,3], *matrix_to_quaternion(T_global[:3,:3])])
                frame_idx += 1
                skip_count += 1

                # ロスト時の再初期化
                if RETRY_GLOBAL_ON_RELOST and icp_started and skip_count >= SKIP_BEFORE_GLOBAL_INIT and not tried_global:
                    print("🆘 ロスト検知 → 初期合わせを再試行予定")
                continue

            # ---- 始動フラグ ----
            if not icp_started:
                print(f"🧭 地図内進入検出（frame {frame_idx}） → ICP追跡開始")
                icp_started = True
                skip_count = 0

            # ---- 姿勢更新 & ログ ----
            T_global = T_new
            R, t = T_global[:3,:3], T_global[:3,3]
            q = matrix_to_quaternion(R)
            print(f"Frame {frame_idx:04d}: fit={fitness:.3f}, rmse={rmse:.3f}")

            with open(out_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([frame_idx, "OK", fitness, rmse, t[0], t[1], t[2], *q])

            # ---- 整合後PLY出力（間引き）----
            if frame_idx % SAVE_EVERY == 0:
                scan_save = o3d.geometry.PointCloud(scan_ds)
                scan_save.transform(T_global)
                ply_path = os.path.join(out_ply, f"frame_{frame_idx:05d}_registered.ply")
                o3d.io.write_point_cloud(ply_path, scan_save)

            frame_idx += 1

    print("✅ 全フレーム処理完了")
    print(f"📄 出力CSV: {out_csv}")
    print(f"📂 整合済みPLY: {out_ply}")

if __name__ == "__main__":
    main()
