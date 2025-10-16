# -*- coding: utf-8 -*-
"""
【黒影対応ICP（GNSSなし・自動サブマップブートストラップ）】
---------------------------------------------------------------
- LiDAR航行データ（.pcap + .json）を使用し、自己位置推定を行う
- 初期位置は未知（GNSS・手動補正なし）
- scan-to-scanでローカルサブマップを構築し、地図との一致を自動探索
- 一致後はGPU(Cupoch)による高速ICP追従
- 欠測(黒=range=0)除外 + 距離フィルタ有効
"""

import os, csv, math
import numpy as np
import open3d as o3d
from collections import deque
from ouster.sdk.open_source import open_source
from ouster.sdk.core import XYZLut, ChanField, SensorInfo

# ==========================================================
# 入出力設定
# ==========================================================
PCAP_PATH  = r"/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10-001.pcap"
JSON_PATH  = r"/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10.json"
MAP_PATH   = r"/workspace/data/1013_lidar_map.ply"
OUTPUT_DIR = r"/workspace/output/1014no4_icp_run_auto"

# ==========================================================
# パラメータ
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

# ---- 初期合わせ ----
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

# ==========================================================
# 関数群
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

# ---- scan-to-scan ICP ----
def scan2scan_icp(src, tgt):
    # === 法線がなければ推定（Point-to-Plane対応） ===
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

# ---- Submap ----
class Submap:
    def __init__(self, keep=SUBMAP_KEEP, voxel=ODOM_VOXEL):
        self.keep = keep; self.voxel = voxel; self.buff = deque()
        self.pcd = o3d.geometry.PointCloud()
    def add(self, pcd):
        self.buff.append(pcd)
        while len(self.buff) > self.keep:
            self.buff.popleft()
        merged = o3d.geometry.PointCloud()
        for p in self.buff: merged += p
        self.pcd = merged.voxel_down_sample(self.voxel)

# ---- 初期合わせ ----
def preprocess_for_feats(pcd, voxel):
    pcd_ds = pcd.voxel_down_sample(voxel)
    pcd_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3, max_nn=60))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_ds,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5, max_nn=100)
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

def yaw_matrix(yaw):
    c,s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4); T[:3,:3] = [[c,-s,0],[s,c,0],[0,0,1]]; return T

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

# ---- Cupoch ----
_HAS_CUPOCH=False
try:
    if USE_GPU:
        import cupoch as cph; _HAS_CUPOCH=True
except Exception: _HAS_CUPOCH=False

def o3d_to_cupoch(p):
    cpcd=cph.geometry.PointCloud()
    pts=np.asarray(p.points).astype(np.float32)
    cpcd.points=cph.utility.Vector3fVector(pts)
    if p.has_normals():
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
# メイン
# ==========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv=os.path.join(OUTPUT_DIR,"icp_trajectory.csv")
    out_ply=os.path.join(OUTPUT_DIR,"icp_registered_frames")
    os.makedirs(out_ply,exist_ok=True)

    print("🗺 地図読み込み中...")
    map_pcd=o3d.io.read_point_cloud(MAP_PATH)
    if not map_pcd.has_normals():
        print("⚠ 法線なし→推定します")
        map_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=MAP_NORMAL_RAD,max_nn=MAP_NORMAL_NN))
    print(f"✅ 地図点数: {len(map_pcd.points):,}")

    with open(JSON_PATH) as f: sensor_info=SensorInfo(f.read())
    xyzlut=XYZLut(sensor_info,use_extrinsics=False)
    source=open_source(PCAP_PATH)

    print("📡 PCAP読込開始 (Cupoch GPU:",_HAS_CUPOCH,")")

    submap=Submap()
    prev=None; T_local=np.eye(4); T_global=np.eye(4)
    boot=True; frame_idx=0; skip=0

    with open(out_csv,"w",newline="") as f:
        csv.writer(f).writerow(["frame","status","fitness","rmse","tx","ty","tz","qx","qy","qz","qw"])

    for scans in source:
        for scan in (scans if isinstance(scans,list) else [scans]):
            xyz_raw=xyzlut(scan); rng_raw=scan.field(ChanField.RANGE)
            pts=scan_xyz_with_masks(xyz_raw,rng_raw)
            if len(pts)==0: continue
            pcd=o3d.geometry.PointCloud(); pcd.points=o3d.utility.Vector3dVector(pts)
            ds=pcd.voxel_down_sample(VOXEL_SIZE)

            if boot:
                if prev is None: prev=ds; submap.add(ds); frame_idx+=1; continue
                T_inc,fit,rmse=scan2scan_icp(ds,prev)
                T_local=T_local@np.linalg.inv(T_inc)
                cur=o3d.geometry.PointCloud(ds); cur.transform(T_local)
                submap.add(cur); prev=ds
                frame_idx+=1

                if frame_idx%BOOTSTRAP_TRY_INTERVAL==0 and len(submap.pcd.points)>3000:
                    print(f"🧭 {frame_idx}F サブマップ試行")
                    T0,fit0,rmse0=try_global_init_fgr(submap.pcd,map_pcd)
                    if fit0>=FGR_ACCEPT_FITNESS_MIN:
                        boot=False; T_global=T0; print(f"✅ 初期整合(FGR):{fit0:.3f}")
                        continue
                    T1,fit1,rmse1=try_yaw_sweep_init(submap.pcd,map_pcd)
                    if T1 is not None and fit1>=SWEEP_ACCEPT_FIT_MIN:
                        boot=False; T_global=T1; print(f"✅ 初期整合(Yaw):{fit1:.3f}")
                        continue
                    print("❌ まだ一致せず")
                if frame_idx>=MAX_BOOTSTRAP_FRAMES:
                    print("🆘 地図内に入らず→終了"); return
                continue

            # === 地図内フェーズ ===
            if _HAS_CUPOCH:
                try: T_new,fit,rmse=try_gpu_icp(ds,map_pcd,T_global)
                except: T_new,fit,rmse=scan2scan_icp(ds,map_pcd)
            else:
                T_new,fit,rmse=scan2scan_icp(ds,map_pcd)

            if fit<MIN_FITNESS or rmse>MAX_RMSE:
                status="SKIP"
            else:
                status="OK"; T_global=T_new

            R,t=T_global[:3,:3],T_global[:3,3]; q=matrix_to_quaternion(R)
            with open(out_csv,"a",newline="") as f:
                csv.writer(f).writerow([frame_idx,status,fit,rmse,*t,*q])

            if frame_idx%SAVE_EVERY==0 and status=="OK":
                s_save=o3d.geometry.PointCloud(ds)
                s_save.transform(T_global)
                o3d.io.write_point_cloud(os.path.join(out_ply,f"frame_{frame_idx:05d}.ply"),s_save)

            print(f"Frame{frame_idx:04d}: {status} fit={fit:.3f} rmse={rmse:.3f}")
            frame_idx+=1

    print("✅ 全フレーム完了")
    print("📄 出力:",out_csv)

if __name__=="__main__":
    main()
