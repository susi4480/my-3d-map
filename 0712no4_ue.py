# -*- coding: utf-8 -*-
"""
【機能】
- 指定フォルダ内のLASファイルを全て処理
- 赤色点群から L 字型補間線（白点）を生成（垂直線＋赤点→垂直線上端）
- レイキャストによる航行可能空間外殻（緑点）を生成（SORノイズ除去なし、補間点含む）
- 白点・緑点を元点群と結合し、LASで保存（PLY出力なし、出力スキップなし）
"""

import os
import math
import logging
import numpy as np
import laspy
from scipy.spatial import cKDTree, ConvexHull, QhullError

# === 入出力設定 ===
INPUT_DIR   = "/output/slice_area/"
OUTPUT_DIR  = "/output/slice_area_navigation_ue/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === パラメータ ===
WATER_LEVEL = 3.5
CLEARANCE   = 1.0
VOXEL_SIZE  = 0.2
N_RAYS      = 720
RAY_LENGTH  = 20.0
STEP        = 0.05
DIST_THRESH = 0.1
SAFETY_DIST = 1.0

Z_VERT   = 0.01
Z_HORIZ  = 1.3
Z_TOL    = 0.35
N_INTERP = 50

def interpolate_red_l_shape(pts, cols):
    mask_r = (cols[:,0] > 0.9) & (cols[:,1] < 0.1) & (cols[:,2] < 0.1)
    red = pts[mask_r]
    if red.size == 0:
        logging.warning("❗ 赤点が見つかりません")
        return np.empty((0,3))
    y_med = np.median(red[:,1])

    v = red[np.abs(red[:,2]-Z_VERT) < Z_TOL]
    left_v  = v[v[:,1] < y_med]
    right_v = v[v[:,1] >= y_med]
    if left_v.size == 0 or right_v.size == 0:
        logging.warning("❗ 垂直線用の赤点が不足しています")
        return np.empty((0,3))
    pL = left_v[np.argmin(left_v[:,2])]
    pR = right_v[np.argmin(right_v[:,2])]

    h = red[np.abs(red[:,2]-Z_HORIZ) < Z_TOL]
    left_h  = h[h[:,1] < y_med]
    right_h = h[h[:,1] >= y_med]
    if left_h.size == 0 or right_h.size == 0:
        logging.warning("❗ 水平線用の赤点が不足しています")
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
    if pts.shape[0] < 3:
        logging.warning("❗ 点数不足でレイキャストスキップ")
        return np.empty((0, 3))
    try:
        yz = pts[:,1:3]
        origin = yz[ConvexHull(yz).vertices].mean(axis=0)
    except QhullError as e:
        logging.warning(f"❗ ConvexHull失敗: {e}")
        return np.empty((0, 3))

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
    if len(hits) == 0:
        return np.empty((0,3))
    minb = pts.min(axis=0)
    ijk = np.floor((hits-minb)/VOXEL_SIZE).astype(int)
    uidx = np.unique(ijk,axis=0)
    centers = minb + (uidx+0.5)*VOXEL_SIZE
    z_lim = WATER_LEVEL + CLEARANCE
    return centers[centers[:,2] <= z_lim]

def process_file(input_path, output_path):
    las = laspy.read(input_path)
    pts = np.vstack([las.x, las.y, las.z]).T
    cols = np.vstack([las.red, las.green, las.blue]).T.astype(np.float64) / 65535.0

    interp = interpolate_red_l_shape(pts, cols)
    interp_color = np.tile([1.0, 1.0, 1.0], (len(interp), 1))
    logging.info(f"補間点数: {len(interp)}")

    pts_with_interp = np.vstack([pts, interp])
    hull_pts = run_raycast_hull(pts_with_interp)
    hull_color = np.tile([0.0, 1.0, 0.0], (len(hull_pts), 1))
    logging.info(f"緑点数: {len(hull_pts)}")

    all_pts  = np.vstack([pts, interp, hull_pts])
    all_cols = np.vstack([cols, interp_color, hull_color])
    red_all   = (all_cols[:,0] * 65535).astype(np.uint16)
    green_all = (all_cols[:,1] * 65535).astype(np.uint16)
    blue_all  = (all_cols[:,2] * 65535).astype(np.uint16)

    out = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    out.header = las.header
    out.x, out.y, out.z = all_pts.T
    out.red, out.green, out.blue = red_all, green_all, blue_all
    if hasattr(las.header,'crs') and las.header.crs:
        out.header.crs = las.header.crs
    out.write(output_path)
    logging.info(f"✅ 出力: {os.path.basename(output_path)}（点数: {len(all_pts)}）")

# === 一括処理 ===
logging.basicConfig(level=logging.INFO)
for filename in sorted(os.listdir(INPUT_DIR)):
    if filename.lower().endswith(".las"):
        in_path = os.path.join(INPUT_DIR, filename)
        name_wo_ext = os.path.splitext(filename)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{name_wo_ext}_navigable.las")
        process_file(in_path, out_path)
