# -*- coding: utf-8 -*-
"""
【機能】M0 on M5風スライスを生成し、各スライスの点群（v–z断面）をそのままLASファイルで出力
- 長方形抽出やoccupancy処理は行わない
- 中心線に沿ったスライス構造を構成し、各断面に含まれる点群を保存
"""

import os
import math
import numpy as np
import laspy

# ===== 入出力 =====
INPUT_LAS = "/workspace/output/0828_01_500_suidoubasi_ue.las"
OUTPUT_DIR = "/workspace/output/slices_m0style"
os.makedirs(output_dir, exist_ok=True)
# ===== パラメータ =====
UKC = -1.0
BIN_X = 2.0
MIN_PTS_PER_XBIN = 50
GAP_DIST = 50.0
SECTION_INTERVAL = 0.5
LINE_LENGTH = 60.0
SLICE_THICKNESS = 0.20
Z_MAX = 10.0  # 任意の上限Z値（制限したい場合）

# ==== ユーティリティ ====
def copy_header_with_metadata(src_header):
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales = src_header.scales
    header.offsets = src_header.offsets
    if getattr(src_header, "srs", None): header.srs = src_header.srs
    if getattr(src_header, "vlrs", None): header.vlrs.extend(src_header.vlrs)
    if getattr(src_header, "evlrs", None): header.evlrs.extend(src_header.evlrs)
    return header

def l2(p, q):
    return math.hypot(q[0]-p[0], q[1]-p[1])

def write_las_points(path, header_src, pts_xyz):
    header = copy_header_with_metadata(header_src)
    las_out = laspy.LasData(header)
    N = len(pts_xyz)
    if N == 0:
        return
    try:
        las_out.points = laspy.ScaleAwarePointRecord.zeros(N, header=header)
    except AttributeError:
        las_out.points = laspy.PointRecord.zeros(N, header=header)
    pts_xyz = np.asarray(pts_xyz, float)
    las_out.x = pts_xyz[:,0]; las_out.y = pts_xyz[:,1]; las_out.z = pts_xyz[:,2]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    las_out.write(path)

# ========= メイン処理 =========
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    las = laspy.read(INPUT_LAS)
    X,Y,Z = np.asarray(las.x,float), np.asarray(las.y,float), np.asarray(las.z,float)
    xy = np.column_stack([X,Y])

    # --- 中心線（M5風） ---
    x_min, x_max = xy[:,0].min(), xy[:,0].max()
    edges = np.arange(x_min, x_max+BIN_X, BIN_X)
    through=[]
    for i in range(len(edges)-1):
        x0,x1 = edges[i], edges[i+1]
        m = (xy[:,0]>=x0)&(xy[:,0]<x1)
        if np.count_nonzero(m) < MIN_PTS_PER_XBIN: continue
        slab_xy, slab_z = xy[m], Z[m]
        order = np.argsort(slab_xy[:,1])
        slab_xy, slab_z = slab_xy[order], slab_z[order]
        under = slab_z <= UKC
        if not np.any(under): continue
        idx = np.where(under)[0]
        left, right = slab_xy[idx[0]], slab_xy[idx[-1]]
        through.append(0.5*(left+right))
    through = np.asarray(through,float)
    if len(through)<2: raise RuntimeError("中心線が作れません")

    # --- 間引き ---
    thinned=[through[0]]
    for p in through[1:]:
        if l2(thinned[-1],p)>=GAP_DIST: thinned.append(p)
    through=np.asarray(thinned,float)

    # --- 内挿 ---
    centers=[]
    for i in range(len(through)-1):
        p,q=through[i],through[i+1]
        d=l2(p,q)
        if d<1e-9: continue
        n_steps=int(d/SECTION_INTERVAL)
        for s_i in range(n_steps+1):
            s=min(s_i*SECTION_INTERVAL,d)
            t=s/d
            centers.append((1-t)*p+t*q)
    centers=np.asarray(centers,float)

    # --- スライスして保存 ---
    half_len=LINE_LENGTH*0.5
    half_th=SLICE_THICKNESS*0.5
    for i in range(len(centers)-1):
        c=centers[i]; cn=centers[i+1]
        t_vec=cn-c; norm=np.linalg.norm(t_vec)
        if norm<1e-9: continue
        t_hat=t_vec/norm; n_hat=np.array([-t_hat[1],t_hat[0]],float)
        dxy=xy - c
        u = dxy @ t_hat
        v = dxy @ n_hat
        m_band = (np.abs(u)<=half_th)&(np.abs(v)<=half_len)&(Z <= Z_MAX)
        pts_xyz = np.column_stack([X[m_band], Y[m_band], Z[m_band]])

        out_path = os.path.join(OUTPUT_DIR, f"slice_{i:04d}.las")
        write_las_points(out_path, las.header, pts_xyz)
        print(f"✅ slice {i:04d} 出力: {len(pts_xyz)} 点")

if __name__=="__main__":
    main()
