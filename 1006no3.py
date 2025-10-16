# -*- coding: utf-8 -*-
"""
【機能】Ouster OS1-64風レイキャスト連続生成（Z=0固定, 出力2種のみ, Z制限なし）
-----------------------------------------------------------------------
- 地図LASから中心線を抽出
- 中心線に沿って一定間隔でLiDARを配置（Z=0）
- 各位置で:
    1. raycast_world（Z=0からのレイキャスト結果）
    2. query_world（ランダム姿勢付き疑似観測）
- テキスト出力なし / 高さ制限なし（建物上部まで含む）
-----------------------------------------------------------------------
出力:
  /output/1006_seq_raycast_world/scan_sector_0000_raycast_world.las
  /output/1006_seq_query_world/  scan_sector_0000_query_world.las
"""

import os
import math
import numpy as np
import laspy
import open3d as o3d

# ==========================================================
# 入出力設定
# ==========================================================
INPUT_LAS   = "/output/0925_sita_merged_white.las"
OUT_DIR_RAY = "/output/1006_seq_raycast_world"
OUT_DIR_QRY = "/output/1006_seq_query_world"

# レイキャスト間隔（中心線に沿って何点おきにスキャンするか：点は約0.5m刻みで生成される想定）
SECTION_STEP = 50   # 例: 50なら約25m〜程度（中心線生成間隔に依存）

# ==========================================================
# LiDARパラメータ（OS1-64準拠に近似）
# ==========================================================
FOV_H_DEG   = 360.0     # 水平視野角
FOV_V_DEG   = 42.4      # 垂直視野角（±21.2°）
H_RES       = 0.18      # 水平方向分解能[°]
V_RES       = 0.33      # 垂直方向分解能[°]
MAX_RANGE   = 170.0     # 最大射程[m]
STEP_COUNT  = 1400      # レイ1本あたりのステップ数
HIT_THR     = 0.20      # 衝突判定距離[m]

# ==========================================================
# 中心線抽出パラメータ（抽出のためだけに使用）
# ==========================================================
UKC         = -2.0      # 中心線抽出の基準Z（川底近辺）
TOL_Z       = 0.2       # 許容範囲[m]

# ==========================================================
# ノイズ＆姿勢ずらし
# ==========================================================
NOISE_STD   = 0.05
VOXEL_SIZE  = 0.10
RAND_YAW_DEG_RANGE   = (-8.0, 8.0)
RAND_PITCH_DEG_RANGE = (-3.0, 3.0)
RAND_ROLL_DEG_RANGE  = (-3.0, 3.0)
RAND_TRANS_RANGE_M   = { "x": (-1.5, 1.5), "y": (-1.5, 1.5), "z": (-0.3, 0.3) }

RANDOM_SEED = 42
# ==========================================================

# ------------------ 基本関数群 ------------------
def rotz(deg):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)
def roty(deg):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    return np.array([[ c,0, s],[0,1,0],[-s,0, c]], float)
def rotx(deg):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    return np.array([[1,0,0],[0, c,-s],[0, s, c]], float)
def l2(p,q): return math.hypot(q[0]-p[0], q[1]-p[1])

def write_las(path, xyz):
    if xyz.size == 0:
        print("⚠ 出力点なし:", path)
        return
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:,0], xyz[:,1], xyz[:,2]
    las.write(path)
    print(f"💾 {os.path.basename(path)} ({len(xyz):,} 点)")

# ------------------ 中心線抽出 ------------------
def extract_centerline(X,Y,Z):
    BIN_X=2.0; MIN_PTS=50; GAP=50.0; INTERVAL=0.5
    x_min, x_max = X.min(), X.max()
    edges = np.arange(x_min, x_max+BIN_X, BIN_X)
    pts=[]
    for i in range(len(edges)-1):
        m = (X>=edges[i])&(X<edges[i+1])
        if np.count_nonzero(m)<MIN_PTS: continue
        slab_xy = np.column_stack([X[m],Y[m]])
        slab_z  = Z[m]
        mz = np.abs(slab_z-UKC)<TOL_Z
        if not np.any(mz): continue
        slab_xy = slab_xy[mz]
        order = np.argsort(slab_xy[:,1])
        left, right = slab_xy[order[0]], slab_xy[order[-1]]
        pts.append(0.5*(left+right))
    pts = np.asarray(pts)
    if len(pts)<2: raise RuntimeError("中心線抽出失敗")

    # 間引き
    thin=[pts[0]]
    for p in pts[1:]:
        if l2(thin[-1],p)>=GAP: thin.append(p)
    pts=np.asarray(thin)
    centers=[]; tang=[]
    for i in range(len(pts)-1):
        p,q=pts[i],pts[i+1]; d=l2(p,q)
        if d<1e-9: continue
        n=int(d/INTERVAL); t_hat=(q-p)/d
        for s_i in range(n+1):
            s=min(s_i*INTERVAL,d); t=s/d
            centers.append((1-t)*p+t*q)
            tang.append(t_hat)
    return np.asarray(centers), np.asarray(tang)

# ------------------ レイキャスト（修正版：pcd_map から点を取得） ------------------
def raycast(origin_world, view_dir, pcd_map, kdtree):
    xyz_map = np.asarray(pcd_map.points)  # ← ここが重要
    num_h=int(FOV_H_DEG/H_RES)+1
    num_v=int(FOV_V_DEG/V_RES)+1
    h_angles=np.linspace(-FOV_H_DEG/2,FOV_H_DEG/2,num_h)
    v_angles=np.linspace(-FOV_V_DEG/2,FOV_V_DEG/2,num_v)
    hits=[]
    for h in h_angles:
        for v in v_angles:
            theta=math.radians(h); phi=math.radians(v)
            dir_h=np.array([
                view_dir[0]*math.cos(theta)-view_dir[1]*math.sin(theta),
                view_dir[0]*math.sin(theta)+view_dir[1]*math.cos(theta),
                0.0
            ])
            dir_h/=np.linalg.norm(dir_h)+1e-12
            dir=dir_h.copy(); dir[2]=math.tan(phi)
            dir/=np.linalg.norm(dir)+1e-12
            for r in np.linspace(0,MAX_RANGE,STEP_COUNT):
                p=origin_world+dir*r
                _,idx,dist2=kdtree.search_knn_vector_3d(p,1)
                if len(idx)>0 and math.sqrt(dist2[0])<HIT_THR:
                    hits.append(xyz_map[idx[0]])  # ← kdtree.data ではなく pcd_map から
                    break
    return np.asarray(hits)

# ------------------ メイン ------------------
def main():
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)
    os.makedirs(OUT_DIR_RAY, exist_ok=True)
    os.makedirs(OUT_DIR_QRY, exist_ok=True)

    las=laspy.read(INPUT_LAS)
    X,Y,Z=np.asarray(las.x,float),np.asarray(las.y,float),np.asarray(las.z,float)
    xyz_map=np.column_stack([X,Y,Z])  # Z制限撤廃

    print("📍 中心線抽出中...")
    centers,tangents=extract_centerline(X,Y,Z)
    total=len(centers)
    print(f"✅ 中心線点数: {total}")

    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(xyz_map)
    kdtree=o3d.geometry.KDTreeFlann(pcd)

    frame_idx=0
    for i in range(0,total,SECTION_STEP):
        origin=np.array([centers[i,0], centers[i,1], 0.0],float)   # Z=0で発射
        view=tangents[i]/np.linalg.norm(tangents[i])

        print(f"\n🚀 FRAME {frame_idx:04d} @ {origin}")

        # === (2) レイキャスト ===
        hits_world=raycast(origin, view, pcd, kdtree)  # ← pcd を渡す
        if hits_world.size==0:
            frame_idx+=1
            continue
        out_world=os.path.join(OUT_DIR_RAY,f"scan_sector_{frame_idx:04d}_raycast_world.las")
        write_las(out_world,hits_world)

        # === (3) ランダム姿勢クエリ ===
        yaw_deg=math.degrees(math.atan2(view[1],view[0]))
        R_w2l=rotz(-yaw_deg)
        hits_local=(R_w2l@(hits_world-origin).T).T
        hits_local+=np.random.normal(0.0,NOISE_STD,hits_local.shape)
        pcd_local=o3d.geometry.PointCloud()
        pcd_local.points=o3d.utility.Vector3dVector(hits_local)
        pcd_local=pcd_local.voxel_down_sample(VOXEL_SIZE)
        hits_local=np.asarray(pcd_local.points)

        ry=np.random.uniform(*RAND_PITCH_DEG_RANGE)
        rx=np.random.uniform(*RAND_ROLL_DEG_RANGE)
        rz=np.random.uniform(*RAND_YAW_DEG_RANGE)
        R_noise=rotz(rz)@roty(ry)@rotx(rx)
        tx=np.random.uniform(*RAND_TRANS_RANGE_M["x"])
        ty=np.random.uniform(*RAND_TRANS_RANGE_M["y"])
        tz=np.random.uniform(*RAND_TRANS_RANGE_M["z"])
        t_noise=np.array([tx,ty,tz])
        R_l2w=rotz(yaw_deg)
        hits_query=(R_l2w@(R_noise@hits_local.T)).T+origin+t_noise

        out_query=os.path.join(OUT_DIR_QRY,f"scan_sector_{frame_idx:04d}_query_world.las")
        write_las(out_query,hits_query)

        frame_idx+=1

    print("\n✅ 全フレーム生成完了:", frame_idx, "個")

if __name__=="__main__":
    main()
