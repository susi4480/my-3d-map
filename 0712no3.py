# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルを読み込み、元点群とカラー情報、CRSを保持
- 赤色点群から L 字型補間線（白点）を生成（垂直線＋赤点→垂直線上端）
- レイキャストによる航行可能空間外殻（緑点）を生成（SORノイズ除去なし、補間点含む）
- 白点・緑点を元点群と結合し、LASで保存（PLY出力はなし）
"""

import os
import math
import logging
import numpy as np
import laspy
import open3d as o3d
from scipy.spatial import cKDTree, ConvexHull

# === 設定 ===
INPUT_LAS   = r"C:\Users\user\Documents\lab\output_ply\slice_area\slice_x_387183.00m.las"
OUTPUT_LAS  = r"C:\Users\user\Documents\lab\output_ply\slice_area_navigation\combined_output.las"

# レイキャスト関連
WATER_LEVEL = 3.5
CLEARANCE   = 1.0
VOXEL_SIZE  = 0.2
N_RAYS      = 720
RAY_LENGTH  = 20.0
STEP        = 0.05
DIST_THRESH = 0.1
SAFETY_DIST = 1.0

# 補間線パラメータ
Z_VERT   = 0.01
Z_HORIZ  = 1.3
Z_TOL    = 0.35
N_INTERP = 50

def interpolate_red_l_shape(pts, cols):
    mask_r = (cols[:,0] > 0.9) & (cols[:,1] < 0.1) & (cols[:,2] < 0.1)
    red = pts[mask_r]
    if red.size == 0:
        logging.error("赤点が見つかりません")
        return np.empty((0,3))
    y_med = np.median(red[:,1])

    v = red[np.abs(red[:,2]-Z_VERT) < Z_TOL]
    left_v  = v[v[:,1] < y_med]
    right_v = v[v[:,1] >= y_med]
    if left_v.size == 0 or right_v.size == 0:
        logging.error("左右の下端赤点が見つかりません")
        return np.empty((0,3))
    pL = left_v[np.argmin(left_v[:,2])]
    pR = right_v[np.argmin(right_v[:,2])]

    h = red[np.abs(red[:,2]-Z_HORIZ) < Z_TOL]
    left_h  = h[h[:,1] < y_med]
    right_h = h[h[:,1] >= y_med]
    if left_h.size == 0 or right_h.size == 0:
        logging.error("左右の水平赤点が見つかりません")
        return np.empty((0,3))
    pHL = left_h[np.argmin(left_h[:,1])]
    pHR = right_h[np.argmax(right_h[:,1])]

    pL_top = np.array([pL[0], pL[1], pHL[2]])
    pR_top = np.array([pR[0], pR[1], pHR[2]])

    lineL  = np.linspace(pL,  pL_top,  N_INTERP)
    lineR  = np.linspace(pR,  pR_top,  N_INTERP)
    lineHL = np.linspace(pHL, pL_top,  N_INTERP)
    lineHR = np.linspace(pHR, pR_top,  N_INTERP)
    return np.vstack([lineL, lineR, lineHL, lineHR])

def run_raycast_hull(pts):
    yz = pts[:,1:3]
    origin = yz[ConvexHull(yz).vertices].mean(axis=0)
    tree = cKDTree(yz)
    angles = np.linspace(0, 2*math.pi, N_RAYS, endpoint=False)
    dirs = np.vstack((np.cos(angles), np.sin(angles))).T
    steps = np.arange(0, RAY_LENGTH+STEP, STEP)
    grid = origin + steps[:,None,None]*dirs[None,:,:]
    flat = grid.reshape(-1,2)
    dists, _ = tree.query(flat)
    dists = dists.reshape(grid.shape[:2])
    hit_pts = []
    for j in range(N_RAYS):
        col = dists[:,j]
        idx = np.where(col < DIST_THRESH)[0]
        if idx.size == 0:
            continue
        i = idx[0]
        d_hit = steps[i]
        d_safe = max(d_hit - SAFETY_DIST, 0)
        dy, dz = dirs[j]
        y_s, z_s = origin + np.array([dy, dz]) * d_safe
        _, ii = tree.query([y_s, z_s])
        x_s = pts[ii,0]
        hit_pts.append([x_s, y_s, z_s])
    hits = np.array(hit_pts)
    minb = pts.min(axis=0)
    ijk = np.floor((hits-minb)/VOXEL_SIZE).astype(int)
    uidx = np.unique(ijk,axis=0)
    centers = minb + (uidx+0.5)*VOXEL_SIZE
    z_lim = WATER_LEVEL + CLEARANCE
    return centers[centers[:,2] <= z_lim]

def main():
    logging.basicConfig(level=logging.INFO)
    las = laspy.read(INPUT_LAS)
    pts = np.vstack([las.x, las.y, las.z]).T
    cols = np.vstack([las.red, las.green, las.blue]).T.astype(np.float64) / 65535.0

    # 補間点（白）
    interp = interpolate_red_l_shape(pts, cols)
    interp_color = np.tile([1.0, 1.0, 1.0], (len(interp), 1))  # 白色
    logging.info(f"補間点数: {len(interp)}")

    # レイキャスト（補間点も含めて実行）
    pts_with_interp = np.vstack([pts, interp])
    hull_pts = run_raycast_hull(pts_with_interp)
    hull_color = np.tile([0.0, 1.0, 0.0], (len(hull_pts), 1))  # 緑色
    logging.info(f"緑点数: {len(hull_pts)}")

    # 統合
    all_pts  = np.vstack([pts, interp, hull_pts])
    all_cols = np.vstack([cols, interp_color, hull_color])
    red_all   = (all_cols[:,0] * 65535).astype(np.uint16)
    green_all = (all_cols[:,1] * 65535).astype(np.uint16)
    blue_all  = (all_cols[:,2] * 65535).astype(np.uint16)

    # LAS出力
    out = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    out.header = las.header
    out.x, out.y, out.z = all_pts.T
    out.red, out.green, out.blue = red_all, green_all, blue_all
    if hasattr(las.header,'crs') and las.header.crs:
        out.header.crs = las.header.crs
    os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)
    out.write(OUTPUT_LAS)
    logging.info(f"✅ LAS出力完了: {OUTPUT_LAS}")

if __name__ == '__main__':
    main()
