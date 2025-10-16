# -*- coding: utf-8 -*-
"""
【GPU対応・法線自動推定版】+ FOV×path.json 初期位置合わせ
黒影対応ICP（サブマップブートストラップ）
---------------------------------------------------------------
- LiDAR航行データ（.pcap + .json）で自己位置推定
- 初期位置は未知（GNSS・手動補正なし）
- ★ 追加: path_resampled.json 上の視点で FOV×NCC による初期位置合わせ
- 一致後は GPU(Cupoch) による高速ICP追従
- 欠測(range=0)点除外 + 距離フィルタあり
- すべてのICPで法線を自動推定
"""

import os, csv, math, json
import numpy as np
import open3d as o3d
import cv2
from collections import deque
from ouster.sdk.open_source import open_source
from ouster.sdk.core import XYZLut, ChanField, SensorInfo

# ==========================================================
# 🔧 入出力設定
# ==========================================================
PCAP_PATH  = r"/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10-001.pcap"
JSON_PATH  = r"/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10.json"
MAP_PATH   = r"/workspace/data/1016_merged_lidar_uesita.ply"
PATH_JSON  = r"/workspace/data/path_resampled.json"
OUTPUT_DIR = r"/workspace/output/icp_no2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# ⚙️ 一般パラメータ
# ==========================================================
VOXEL_SIZE = 0.15
MAX_CORR_DIST = 0.5
MAX_ITER = 60
MIN_RANGE = 1.0
MAX_RANGE = 120.0
MIN_FITNESS = 0.30
MAX_RMSE = 0.25
SAVE_EVERY = 10
USE_GPU = True

# ---- サブマップ構築 ----
BUILD_SUBMAP_FRAMES = 80
ODOM_VOXEL = 0.20
ODOM_CORR_DIST = 1.0
ODOM_MAX_ITER = 30
SUBMAP_KEEP = 150

BOOTSTRAP_TRY_INTERVAL = 50
MAX_BOOTSTRAP_FRAMES = 5000

# ==========================================================
# 🌄 FOV×path 初期合わせパラメータ（★追加）
# ==========================================================
# FOV（センサー視野を角度で表す）
H_FOV_DEG = 120.0     # 水平視野（±60°）
V_FOV_DEG = 30.0      # 垂直視野（±15°）
H_RES_DEG = 0.2       # 水平方向角度分解能
V_RES_DEG = 0.2       # 垂直方向角度分解能
NCC_MIN_COMMON = 200  # NCCに使う共通有効画素の下限
NCC_THRESHOLD = 0.24  # 一致とみなすNCC閾値（0.22〜0.30で調整）

# パスが逆向きの可能性に備えてYaw反転オプション
REVERSE_PATH_DIRECTION = True  # Trueなら yaw += 180°

# ==========================================================
# 🧩 関数群（共通）
# ==========================================================
def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp, cp*sr, cp*cr]])

def matrix_to_quaternion(R):
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1]-R[1,2]) / S
        qy = (R[0,2]-R[2,0]) / S
        qz = (R[1,0]-R[0,1]) / S
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            S = math.sqrt(1.0 + R[0,0]-R[1,1]-R[2,2]) * 2
            qw = (R[2,1]-R[1,2]) / S; qx = 0.25*S
            qy = (R[0,1]+R[1,0]) / S; qz = (R[0,2]+R[2,0]) / S
        elif i == 1:
            S = math.sqrt(1.0 + R[1,1]-R[0,0]-R[2,2]) * 2
            qw = (R[0,2]-R[2,0]) / S; qx = (R[0,1]+R[1,0]) / S
            qy = 0.25*S; qz = (R[1,2]+R[2,1]) / S
        else:
            S = math.sqrt(1.0 + R[2,2]-R[0,0]-R[1,1]) * 2
            qw = (R[1,0]-R[0,1]) / S; qx = (R[0,2]+R[2,0]) / S
            qy = (R[1,2]+R[2,1]) / S; qz = 0.25*S
    return (qx,qy,qz,qw)

def scan_xyz_with_masks(xyz_raw, rng_raw):
    valid = (rng_raw > 0)
    xyz = np.asarray(xyz_raw).reshape(-1,3)
    rng = np.asarray(rng_raw).reshape(-1)
    valid = valid.reshape(-1)
    d = np.linalg.norm(xyz, axis=1)
    valid &= (d > MIN_RANGE)
    if MAX_RANGE > 0:
        valid &= (d < MAX_RANGE)
    return xyz[valid]

def yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3,:3] = [[c,-s,0],[s,c,0],[0,0,1]]
    return T

# ==========================================================
# 🧱 Scan-to-Scan ICP（法線自動推定付き）
# ==========================================================
def scan2scan_icp(src, tgt):
    """Scan-to-Scan ICP（Point-to-Plane + 法線自動推定）"""
    for p in (src, tgt):
        if not p.has_normals():
            p.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=ODOM_VOXEL*3, max_nn=30)
            )
    res = o3d.pipelines.registration.registration_icp(
        src, tgt, ODOM_CORR_DIST, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ODOM_MAX_ITER)
    )
    return res.transformation, float(res.fitness), float(res.inlier_rmse)

# ==========================================================
# 🧩 Submapクラス
# ==========================================================
class Submap:
    def __init__(self, keep=SUBMAP_KEEP, voxel=ODOM_VOXEL):
        self.keep = keep
        self.voxel = voxel
        self.buff = deque()
        self.pcd = o3d.geometry.PointCloud()
    def add(self, pcd):
        self.buff.append(pcd)
        while len(self.buff) > self.keep:
            self.buff.popleft()
        merged = o3d.geometry.PointCloud()
        for p in self.buff:
            merged += p
        self.pcd = merged.voxel_down_sample(self.voxel)

# ==========================================================
# 🌍 初期合わせ（従来のFGR / Yawスイープ）※バックアップ用
# ==========================================================
FGR_VOXEL = 0.60
FGR_MAX_CORR_DIST = FGR_VOXEL * 2.5
FGR_ITER = 1000
FGR_ACCEPT_FITNESS_MIN = 0.08

YAW_SWEEP_STEP_DEG = 15
COARSE_VOXEL = 0.40
COARSE_CORR_DIST = 2.0
COARSE_MAX_ITER = 15
SWEEP_ACCEPT_FIT_MIN = 0.08

MAP_NORMAL_RAD = 1.0
MAP_NORMAL_NN = 50

def preprocess_for_feats(pcd, voxel):
    pcd_ds = pcd.voxel_down_sample(voxel)
    pcd_ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3, max_nn=60)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_ds, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5, max_nn=100)
    )
    return pcd_ds, fpfh

def try_global_init_fgr(scan, map_pcd, voxel=FGR_VOXEL):
    s_ds, s_feat = preprocess_for_feats(scan, voxel)
    m_ds, m_feat = preprocess_for_feats(map_pcd, voxel)
    opt = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=FGR_MAX_CORR_DIST,
        iteration_number=FGR_ITER)
    res = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        s_ds, m_ds, s_feat, m_feat, opt)
    return res.transformation, float(res.fitness), float(res.inlier_rmse)

def coarse_icp_once(scan,map_pcd,init):
    res = o3d.pipelines.registration.registration_icp(
        scan,map_pcd,COARSE_CORR_DIST,init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=COARSE_MAX_ITER))
    return res.transformation,float(res.fitness),float(res.inlier_rmse)

def try_yaw_sweep_init(scan,map_pcd):
    scan_c = scan.voxel_down_sample(COARSE_VOXEL)
    map_c  = map_pcd.voxel_down_sample(COARSE_VOXEL)
    best_T,best_fit,best_rmse=None,-1,1e9
    for deg in range(0,360,YAW_SWEEP_STEP_DEG):
        T0=yaw_matrix(np.deg2rad(deg))
        T,fit,rmse=coarse_icp_once(scan_c,map_c,T0)
        if fit>best_fit or (fit==best_fit and rmse<best_rmse):
            best_T,best_fit,best_rmse=T,fit,rmse
    return best_T,best_fit,best_rmse

# ==========================================================
# ⚡ GPU (Cupoch)
# ==========================================================
_HAS_CUPOCH=False
try:
    if USE_GPU:
        import cupoch as cph; _HAS_CUPOCH=True
        print("🚀 GPU有効 (Cupoch)")
except Exception:
    print("⚠ GPUなし → CPUモード")

def o3d_to_cupoch(p):
    cpcd=cph.geometry.PointCloud()
    pts=np.asarray(p.points).astype(np.float32)
    cpcd.points=cph.utility.Vector3fVector(pts)
    if not p.has_normals():
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=MAP_NORMAL_RAD, max_nn=MAP_NORMAL_NN))
    n=np.asarray(p.normals).astype(np.float32)
    cpcd.normals=cph.utility.Vector3fVector(n)
    return cpcd

def try_gpu_icp(scan,map_pcd,init_T):
    s=o3d_to_cupoch(scan); m=o3d_to_cupoch(map_pcd)
    reg=cph.registration.registration_icp(
        s,m,MAX_CORR_DIST,init_T.astype(np.float32),
        cph.registration.TransformationEstimationPointToPlane(),
        cph.registration.ICPConvergenceCriteria(max_iteration=MAX_ITER))
    return reg.transformation.astype(np.float64),float(reg.fitness),float(reg.inlier_rmse)

# ==========================================================
# 🧭 FOV×path.json 初期合わせ（★追加）
# ==========================================================
def rotate_z(points: np.ndarray, yaw_deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    R = np.array([[ c, -s, 0.0],
                  [ s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return points @ R.T

def points_to_fov_depth(points: np.ndarray) -> np.ndarray:
    """点群をFOV(±H/2, ±V/2)に投影して最短距離の深度画像を作る"""
    h_bins = int(round(H_FOV_DEG / H_RES_DEG))
    v_bins = int(round(V_FOV_DEG / V_RES_DEG))
    depth = np.full((v_bins, h_bins), np.inf, np.float32)
    if points.size == 0:
        depth[:] = 0
        return depth
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x*x + y*y + z*z)
    yaw   = np.degrees(np.arctan2(y, x))
    pitch = np.degrees(np.arctan2(z, np.sqrt(x*x + y*y)))
    m = (np.abs(yaw) <= H_FOV_DEG/2) & (np.abs(pitch) <= V_FOV_DEG/2) & (r > 0)
    if not np.any(m):
        depth[:] = 0
        return depth
    yaw, pitch, r = yaw[m], pitch[m], r[m]
    u = ((yaw + H_FOV_DEG/2) / H_RES_DEG).astype(np.int32)
    v = ((V_FOV_DEG/2 - pitch) / V_RES_DEG).astype(np.int32)
    h_bins = int(round(H_FOV_DEG / H_RES_DEG))
    v_bins = int(round(V_FOV_DEG / V_RES_DEG))
    valid = (u >= 0) & (u < h_bins) & (v >= 0) & (v < v_bins)
    u, v, r = u[valid], v[valid], r[valid]
    lin = v * h_bins + u
    buf = depth.reshape(-1)
    np.minimum.at(buf, lin, r.astype(np.float32))
    depth = buf.reshape(v_bins, h_bins)
    depth[np.isinf(depth)] = 0
    return depth

def depth_to_u8(depth: np.ndarray) -> np.ndarray:
    d = depth.copy()
    mask = d > 0
    if np.any(mask):
        dmin, dmax = float(d[mask].min()), float(d[mask].max())
        if dmax > dmin:
            d[mask] = 255.0 * (d[mask] - dmin) / (dmax - dmin)
    return d.astype(np.uint8)

def ncc_on_valid(a_u8: np.ndarray, b_u8: np.ndarray, min_common: int) -> float:
    m = (a_u8 > 0) & (b_u8 > 0)
    n = int(m.sum())
    if n < min_common:
        return -1e9
    A = a_u8[m].astype(np.float32)
    B = b_u8[m].astype(np.float32)
    A -= A.mean(); B -= B.mean()
    denom = np.sqrt((A*A).sum() * (B*B).sum()) + 1e-6
    return float((A*B).sum() / denom)

def map_to_fov_depth(map_pts: np.ndarray, view_xy, yaw_deg: float) -> np.ndarray:
    """地図点群を観測視点(view_xy, yaw_deg)でセンサー系に変換してFOV深度に投影"""
    vx, vy = view_xy
    pts = map_pts - np.array([vx, vy, 0.0], dtype=np.float64)
    pts_s = rotate_z(pts, -yaw_deg)  # センサー前方+Xに合わせる
    return points_to_fov_depth(pts_s)

def try_path_fov_init(map_pts: np.ndarray, path_xy: np.ndarray, scan_pts: np.ndarray):
    """path上の各視点(進行方向yaw)で地図FOVを作り、scanのFOVとNCC比較して最良を返す"""
    # スキャン側（センサー座標系そのまま）
    scan_depth = points_to_fov_depth(scan_pts)
    scan_u8 = depth_to_u8(scan_depth)

    best = dict(score=-1e9, x=None, y=None, yaw=None, ref_u8=None)
    for i in range(len(path_xy) - 1):
        x0, y0 = path_xy[i]
        x1, y1 = path_xy[i+1]
        yaw_deg = math.degrees(math.atan2(y1 - y0, x1 - x0))
        if REVERSE_PATH_DIRECTION:
            yaw_deg = (yaw_deg + 180.0) % 360.0

        ref_depth = map_to_fov_depth(map_pts, (x0, y0), yaw_deg)
        ref_u8 = depth_to_u8(ref_depth)

        # 共通画素チェック
        if ((ref_u8 > 0) & (scan_u8 > 0)).sum() < NCC_MIN_COMMON:
            continue

        s = ncc_on_valid(ref_u8, scan_u8, NCC_MIN_COMMON)
        if s > best["score"]:
            best.update(score=s, x=float(x0), y=float(y0), yaw=float(yaw_deg), ref_u8=ref_u8)

    return best, scan_u8

# ==========================================================
# 🚀 メイン
# ==========================================================
def main():
    print("🗺 地図読み込み中...")
    map_pcd=o3d.io.read_point_cloud(MAP_PATH)
    if not map_pcd.has_normals():
        print("⚠ 法線なし→推定")
        map_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=MAP_NORMAL_RAD,max_nn=MAP_NORMAL_NN))
    map_pts = np.asarray(map_pcd.points).astype(np.float64)
    print(f"✅ 地図点数: {len(map_pts):,}")

    print("📄 path_resampled.json 読み込み中...")
    with open(PATH_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    path_xy = np.array(data["path"], dtype=np.float64)
    print(f"✅ path点数: {len(path_xy)}")

    with open(JSON_PATH) as f: sensor_info=SensorInfo(f.read())
    xyzlut=XYZLut(sensor_info,use_extrinsics=False)
    source=open_source(PCAP_PATH)

    submap=Submap()
    prev=None; T_local=np.eye(4); T_global=np.eye(4)
    boot=True; frame_idx=0

    out_csv=os.path.join(OUTPUT_DIR,"trajectory.csv")
    with open(out_csv,"w",newline="") as f:
        csv.writer(f).writerow(["frame","status","fitness","rmse","tx","ty","tz","qx","qy","qz","qw"])

    print("📡 ストリーム開始 (GPU:",_HAS_CUPOCH,")")

    for scans in source:
        for scan in (scans if isinstance(scans,list) else [scans]):
            xyz_raw=xyzlut(scan); rng_raw=scan.field(ChanField.RANGE)
            pts=scan_xyz_with_masks(xyz_raw,rng_raw)
            if len(pts)==0: 
                frame_idx+=1
                continue

            pcd=o3d.geometry.PointCloud(); pcd.points=o3d.utility.Vector3dVector(pts)
            ds=pcd.voxel_down_sample(VOXEL_SIZE)

            # 法線自動推定
            if not ds.has_normals():
                ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=ODOM_VOXEL*3,max_nn=30))

            # --- 初期ブート（地図外） ---
            if boot:
                if prev is None:
                    prev=ds; submap.add(ds); frame_idx+=1; continue

                # ローカルオドメトリ（submap用蓄積）
                T_inc,fit,rmse=scan2scan_icp(ds,prev)
                T_local=T_local@np.linalg.inv(T_inc)
                cur=o3d.geometry.PointCloud(ds); cur.transform(T_local)
                submap.add(cur); prev=ds
                frame_idx+=1

                # 一定間隔で FOV×path 初期合わせを試みる
                if frame_idx % BOOTSTRAP_TRY_INTERVAL == 0:
                    print(f"🧭 {frame_idx}F FOV×path で初期合わせ試行...")
                    best, scan_u8 = try_path_fov_init(map_pts, path_xy, pts)
                    print(f"   → best NCC={best['score']:.3f} @ (x={best['x']}, y={best['y']}, yaw={best['yaw']})")
                    if best["score"] > NCC_THRESHOLD and best["x"] is not None:
                        # 初期姿勢（センサー→地図）を (x,y,yaw) から構成
                        yaw_rad = math.radians(best["yaw"])
                        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
                        T0 = np.eye(4)
                        T0[:3,:3] = [[c,-s,0],[s,c,0],[0,0,1]]
                        T0[:3,3] = [best["x"], best["y"], 0.0]
                        T_global = T0
                        boot = False
                        # デバッグ出力（一致画像）
                        ref_c = cv2.applyColorMap(best["ref_u8"], cv2.COLORMAP_JET)
                        qry_c = cv2.applyColorMap(scan_u8, cv2.COLORMAP_JET)
                        overlay = cv2.addWeighted(ref_c, 0.5, qry_c, 0.5, 0)
                        cv2.imwrite(os.path.join(OUTPUT_DIR, f"bootstrap_F{frame_idx:05d}.png"), overlay)
                        print("✅ 初期整合成功 (FOV×path)")
                        continue

                # バックアップ: 旧来のFGR / YawSweep も必要なら残す（コメント外して使用）
                # if frame_idx%BOOTSTRAP_TRY_INTERVAL==0 and len(submap.pcd.points)>3000:
                #     print(f"🧭 {frame_idx}F FGR/YawSweep 試行中...")
                #     T1,fit1,rmse1=try_yaw_sweep_init(submap.pcd,map_pcd)
                #     if T1 is not None and fit1>=SWEEP_ACCEPT_FIT_MIN:
                #         boot=False; T_global=T1; print(f"✅ 初期整合成功(YawSweep):{fit1:.3f}")
                #         continue

                if frame_idx>=MAX_BOOTSTRAP_FRAMES:
                    print("🆘 地図に入らず終了"); return

                continue  # 次フレームへ

            # --- 地図内フェーズ（ICP追従） ---
            if _HAS_CUPOCH:
                try:
                    T_new,fit,rmse=try_gpu_icp(ds,map_pcd,T_global)
                except Exception:
                    T_new,fit,rmse=scan2scan_icp(ds,map_pcd)
            else:
                T_new,fit,rmse=scan2scan_icp(ds,map_pcd)

            status="OK" if (fit>=MIN_FITNESS and rmse<=MAX_RMSE) else "SKIP"
            if status=="OK": 
                T_global=T_new

            R,t=T_global[:3,:3],T_global[:3,3]; q=matrix_to_quaternion(R)
            with open(out_csv,"a",newline="") as f:
                csv.writer(f).writerow([frame_idx,status,fit,rmse,*t,*q])

            if frame_idx%SAVE_EVERY==0 and status=="OK":
                s_save=o3d.geometry.PointCloud(ds)
                s_save.transform(T_global)
                o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR,f"frame_{frame_idx:05d}.ply"),s_save)

            print(f"Frame {frame_idx:04d}: {status} fit={fit:.3f} rmse={rmse:.3f}")
            frame_idx+=1

    print("✅ 全フレーム完了")
    print(f"📄 出力CSV: {out_csv}")

if __name__=="__main__":
    main()
