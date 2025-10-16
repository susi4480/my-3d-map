# -*- coding: utf-8 -*-
"""
【機能】UKC=0 の高さを基準に中心線を生成し、レイキャスト風に擬似スキャンを切り出し
- 中心線は「Z=UKC付近の点」の左右端の中点を連ねて構築
- 指定した中心点インデックスをセンサ原点とし、扇形視野でレイキャスト
- 扇形条件（半径 + 視野角）で点群をフィルタ
- CloudCompareで確認できるよう LAS保存

必要ライブラリ: numpy, laspy, open3d
"""

import os
import math
import numpy as np
import laspy
import open3d as o3d

# ===== 入出力 =====
INPUT_LAS  = "/data/0925_sita_classified.las"  # 地図LAS
OUTPUT_DIR = "/output/forward_scans"           # 出力先

# ===== 中心線抽出パラメータ =====
UKC = 0.0             # 水面基準の高さ
TOL_Z = 0.2           # UKC=0 近傍の許容範囲
BIN_X = 2.0
MIN_PTS_PER_XBIN = 50
GAP_DIST = 50.0
SECTION_INTERVAL = 0.5

# ===== レイキャスト風スキャンパラメータ =====
FORWARD_LEN   = 60.0      # 半径 [m]
FOV_DEG       = 120.0     # 視野角（度, 例: 120° = ±60°）
VOXEL_SIZE    = 0.30      # ダウンサンプリング
Z_MAX_FOR_NAV = 3.0       # 高さ制限
SELECT_I      = 2000      # どの中心点でスキャンを作るか (None=全点)

# ==== ユーティリティ ====
def l2(p, q):
    return math.hypot(q[0]-p[0], q[1]-p[1])

def write_las_xyz(path, xyz):
    if xyz.size == 0:
        return
    header = laspy.LasHeader(point_format=3, version="1.2")
    las_out = laspy.LasData(header)
    las_out.x, las_out.y, las_out.z = xyz[:,0], xyz[:,1], xyz[:,2]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    las_out.write(path)

# === 中心線抽出 (UKC=0 基準) ===
def extract_centerline(X, Y, Z):
    x_min, x_max = X.min(), X.max()
    edges = np.arange(x_min, x_max + BIN_X, BIN_X)
    through=[]
    for i in range(len(edges)-1):
        x0, x1 = edges[i], edges[i+1]
        m = (X >= x0) & (X < x1)
        if np.count_nonzero(m) < MIN_PTS_PER_XBIN:
            continue

        slab_xy = np.column_stack([X[m], Y[m]])
        slab_z  = Z[m]

        # ★ Z=UKC近傍の点だけ
        m_ukc = np.abs(slab_z - UKC) < TOL_Z
        if not np.any(m_ukc):
            continue

        slab_xy = slab_xy[m_ukc]
        order = np.argsort(slab_xy[:,1])
        left, right = slab_xy[order[0]], slab_xy[order[-1]]

        through.append(0.5*(left+right))

    through=np.asarray(through,float)
    if len(through)<2:
        raise RuntimeError("中心線が作れません。")

    # 点間隔で間引き
    thinned=[through[0]]
    for p in through[1:]:
        if l2(thinned[-1],p) >= GAP_DIST:
            thinned.append(p)
    through=np.asarray(thinned,float)

    # セクション内挿
    centers=[]; tangents=[]
    for i in range(len(through)-1):
        p,q=through[i],through[i+1]
        d=l2(p,q)
        if d<1e-9: continue
        n_steps=int(d/SECTION_INTERVAL)
        t_hat=(q-p)/d
        for s_i in range(n_steps+1):
            s=min(s_i*SECTION_INTERVAL,d)
            t=s/d
            centers.append((1-t)*p+t*q)
            tangents.append(t_hat)
    centers=np.asarray(centers,float)
    tangents=np.asarray(tangents,float)

    return centers, tangents

# === メイン処理 ===
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === LAS読み込み ===
    las = laspy.read(INPUT_LAS)
    X, Y, Z = np.asarray(las.x,float), np.asarray(las.y,float), np.asarray(las.z,float)
    xyz = np.column_stack([X,Y,Z])

    # 高さ制限
    m_nav = (Z <= Z_MAX_FOR_NAV)
    xyz_nav = xyz[m_nav]
    if len(xyz_nav) == 0:
        raise RuntimeError("Z_MAX_FOR_NAV で点が残りません。")

    # === 中心線抽出 ===
    centers, tangents = extract_centerline(X, Y, Z)

    # === レイキャスト風スキャン生成 ===
    indices = range(len(centers)) if SELECT_I is None else [SELECT_I]
    cos_fov = np.cos(np.deg2rad(FOV_DEG/2))
    out_count = 0

    for i in indices:
        if i >= len(centers): continue
        c = centers[i]      # センサ位置（UKC=0 の左右端中心）
        t_hat = tangents[i] # 航路方向ベクトル

        rel = xyz_nav[:,:2] - c
        dist = np.linalg.norm(rel, axis=1)
        rel_norm = rel / np.maximum(dist[:,None],1e-9)

        forward = rel @ t_hat
        mask = (dist < FORWARD_LEN) & (forward > 0) & ((rel_norm @ t_hat) > cos_fov)

        scan = xyz_nav[mask]
        if len(scan)==0:
            print(f"⚠ 中心点{i}: スキャン点なし")
            continue

        # ダウンサンプル
        pcd_scan=o3d.geometry.PointCloud()
        pcd_scan.points=o3d.utility.Vector3dVector(scan)
        pcd_scan=pcd_scan.voxel_down_sample(VOXEL_SIZE)

        out_las=os.path.join(OUTPUT_DIR,f"scan_sector_{i:04d}.las")
        write_las_xyz(out_las, np.asarray(pcd_scan.points))
        print(f"✅ 中心点{i}: {len(pcd_scan.points)} 点 → {out_las}")
        out_count+=1

    if out_count==0:
        print("⚠ 出力なし。パラメータ(FOV,FORWARD_LEN,SELECT_I)を見直してください。")
    else:
        print(f"🎉 完了: {out_count} 件の擬似スキャンを保存しました。")

if __name__=="__main__":
    main()
