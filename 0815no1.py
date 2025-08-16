# -*- coding: utf-8 -*-
"""
M5（3D占有ボクセル接続）＋白色点群除外＋統計的外れ値除去（SOR）付き
"""

import os
import math
import numpy as np
import laspy
import cv2
import open3d as o3d  # SORに使用

def l2(p, q):
    return math.hypot(q[0] - p[0], q[1] - p[1])

def downfill_on_closed(closed_uint8, z_min, grid_res, anchor_z, tol):
    closed_bool = (closed_uint8 > 0)
    gh, gw = closed_bool.shape
    i_anchor = int(round((anchor_z - z_min) / grid_res))
    pad = max(0, int(np.ceil(tol / grid_res)))
    i_lo = max(0, i_anchor - pad)
    i_hi = min(gh - 1, i_anchor + pad)
    if i_lo > gh - 1 or i_hi < 0:
        return (closed_bool.astype(np.uint8) * 255)
    out = closed_bool.copy()
    for j in range(gw):
        col = closed_bool[:, j]
        if not np.any(col): 
            continue
        if np.any(col[i_lo:i_hi+1]):
            imax = np.max(np.where(col)[0])
            out[:imax+1, j] = True
    return (out.astype(np.uint8) * 255)

def rectangles_and_free(points_vz, grid_res, morph_radius, use_anchor, anchor_z, anchor_tol):
    if len(points_vz) == 0:
        return None, None
    v_min, v_max = points_vz[:, 0].min(), points_vz[:, 0].max()
    z_min, z_max = points_vz[:, 1].min(), points_vz[:, 1].max()
    gw = max(1, int(np.ceil((v_max - v_min) / grid_res)))
    gh = max(1, int(np.ceil((z_max - z_min) / grid_res)))
    grid_raw = np.zeros((gh, gw), dtype=np.uint8)
    yi = ((points_vz[:, 0] - v_min) / grid_res).astype(int)
    zi = ((points_vz[:, 1] - z_min) / grid_res).astype(int)
    ok = (yi >= 0) & (yi < gw) & (zi >= 0) & (zi < gh)
    grid_raw[zi[ok], yi[ok]] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*morph_radius+1, 2*morph_radius+1))
    closed0 = cv2.morphologyEx(grid_raw, cv2.MORPH_CLOSE, kernel)
    closed = downfill_on_closed(closed0, z_min, grid_res, anchor_z, anchor_tol) if use_anchor else closed0
    free_bitmap = ~(closed > 0)
    bbox = (v_min, z_min, gw, gh)
    return free_bitmap, bbox

def method5_voxel_connect(slices_meta, header, output_las):
    voxels = set()
    for i, meta in enumerate(slices_meta):
        fb = meta["free_bitmap"]
        if fb is None:
            continue
        c = meta["c"]
        n_hat = meta["n_hat"]
        grid_res = meta["grid_res"]
        v_min = meta["v_min"]
        z_min = meta["z_min"]
        gw = meta["gw"]
        gh = meta["gh"]
        for row in range(gh):
            for col in range(gw):
                if fb[row, col]:
                    v = v_min + col * grid_res
                    z = z_min + row * grid_res
                    pt2d = c + v * n_hat
                    voxel_x = int(round(pt2d[0] / VOXEL_RES_S))
                    voxel_y = int(round(pt2d[1] / VOXEL_RES_S))
                    voxel_z = int(round(z / VOXEL_RES_V))
                    voxels.add((voxel_x, voxel_y, voxel_z))
    if not voxels:
        raise RuntimeError("緑点が生成されませんでした")
    if VOXEL_BORDER_ONLY:
        from scipy.ndimage import binary_dilation
        voxels_arr = np.array(list(voxels))
        min_corner = voxels_arr.min(axis=0)
        max_corner = voxels_arr.max(axis=0)
        shape = max_corner - min_corner + 3
        grid = np.zeros(shape, dtype=bool)
        offset = min_corner - 1
        for v in voxels:
            idx = tuple(np.array(v) - offset)
            grid[idx] = True
        dilated = binary_dilation(grid)
        border = dilated & (~grid)
        green_indices = np.argwhere(border)
        green_points = (green_indices + offset) * [VOXEL_RES_S, VOXEL_RES_S, VOXEL_RES_V]
    else:
        green_points = np.array([np.array(v) * [VOXEL_RES_S, VOXEL_RES_S, VOXEL_RES_V] for v in voxels])
    new_las = laspy.create(point_format=header.point_format, file_version=header.version)
    new_las.header = header
    new_las.x = green_points[:, 0]
    new_las.y = green_points[:, 1]
    new_las.z = green_points[:, 2]
    new_las.red = np.full(len(green_points), 0, dtype=np.uint16)
    new_las.green = np.full(len(green_points), 65535, dtype=np.uint16)
    new_las.blue = np.full(len(green_points), 0, dtype=np.uint16)
    new_las.write(output_las)

def main():
    # ===== 入出力 =====
    INPUT_LAS  = r"/output/0731_suidoubasi_ue.las"
    OUTPUT_LAS = r"/output/0815no1ver2_23_M5_voxel_SOR.las"
    os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)

    # ===== パラメータ =====
    global VOXEL_RES_V, VOXEL_RES_S, VOXEL_BORDER_ONLY
    UKC = -1.0
    BIN_X = 2.0
    MIN_PTS_PER_XBIN = 50
    GAP_DIST = 50.0
    SECTION_INTERVAL = 0.5
    LINE_LENGTH = 60.0
    SLICE_THICKNESS = 0.20
    MIN_PTS_PER_SLICE = 80
    Z_MAX_FOR_NAV = 1.9
    GRID_RES = 0.10
    MORPH_RADIUS = 23
    USE_ANCHOR_DOWNFILL = True
    ANCHOR_Z = 1.50
    ANCHOR_TOL = 0.5
    VOXEL_RES_V = GRID_RES
    VOXEL_RES_S = SECTION_INTERVAL
    VOXEL_BORDER_ONLY = True

    las = laspy.read(INPUT_LAS)
    if {"red", "green", "blue"} <= set(las.point_format.dimension_names):
        R = np.asarray(las.red)
        G = np.asarray(las.green)
        B = np.asarray(las.blue)
        is_not_white = ~((R==65535) & (G==65535) & (B==65535))
        X = np.asarray(las.x, float)[is_not_white]
        Y = np.asarray(las.y, float)[is_not_white]
        Z = np.asarray(las.z, float)[is_not_white]
    else:
        X = np.asarray(las.x, float)
        Y = np.asarray(las.y, float)
        Z = np.asarray(las.z, float)

    print("\U0001f9f9 SORノイズ除去中...")
    pts = np.column_stack([X, Y, Z])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    filtered_pts = np.asarray(pcd.points)
    X = filtered_pts[:, 0]
    Y = filtered_pts[:, 1]
    Z = filtered_pts[:, 2]
    xy = np.column_stack([X, Y])
    print(f"✅ SOR後の点数: {len(X):,}")

    x_min, x_max = xy[:,0].min(), xy[:,0].max()
    edges = np.arange(x_min, x_max + BIN_X, BIN_X)
    through = []
    for i in range(len(edges)-1):
        x0, x1 = edges[i], edges[i+1]
        m = (xy[:,0] >= x0) & (xy[:,0] < x1)
        if np.count_nonzero(m) < MIN_PTS_PER_XBIN:
            continue
        slab_xy = xy[m]; slab_z = Z[m]
        order = np.argsort(slab_xy[:,1])
        slab_xy = slab_xy[order]; slab_z = slab_z[order]
        under = slab_z <= UKC
        if not np.any(under):
            continue
        idx = np.where(under)[0]
        left  = slab_xy[idx[0]]
        right = slab_xy[idx[-1]]
        c = 0.5 * (left + right)
        through.append(c)
    if len(through) < 2:
        raise RuntimeError("中心線が作れません。UKCやBIN_Xを調整してください。")
    through = np.asarray(through, float)
    thinned = [through[0]]
    for p in through[1:]:
        if l2(thinned[-1], p) >= GAP_DIST:
            thinned.append(p)
    through = np.asarray(thinned, float)

    centers = []
    for i in range(len(through)-1):
        p, q = through[i], through[i+1]
        d = l2(p, q)
        if d < 1e-9:
            continue
        n_steps = int(d / SECTION_INTERVAL)
        for s_i in range(n_steps+1):
            s = min(s_i * SECTION_INTERVAL, d)
            t = s / d
            centers.append((1-t)*p + t*q)
    centers = np.asarray(centers, float)

    half_len = LINE_LENGTH * 0.5
    half_th  = SLICE_THICKNESS * 0.5
    slices_meta = []
    for i in range(len(centers)-1):
        c = centers[i]
        cn = centers[i+1]
        t_vec = cn - c
        norm = np.linalg.norm(t_vec)
        if norm < 1e-9:
            slices_meta.append({"c": c, "n_hat": np.array([1.0, 0.0]), "grid_res": GRID_RES, "v_min": 0, "z_min": 0, "gw": 0, "gh": 0, "free_bitmap": None})
            continue
        t_hat = t_vec / norm
        n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)
        dxy = xy - c
        u = dxy @ t_hat
        v = dxy @ n_hat
        m_band = (np.abs(u) <= half_th) & (np.abs(v) <= half_len)
        m_nav = m_band & (Z <= Z_MAX_FOR_NAV)
        if np.count_nonzero(m_nav) < MIN_PTS_PER_SLICE:
            slices_meta.append({"c": c, "n_hat": n_hat, "grid_res": GRID_RES, "v_min": 0, "z_min": 0, "gw": 0, "gh": 0, "free_bitmap": None})
            continue
        points_vz = np.column_stack([v[m_nav], Z[m_nav]])
        free_bitmap, bbox = rectangles_and_free(points_vz, GRID_RES, MORPH_RADIUS, USE_ANCHOR_DOWNFILL, ANCHOR_Z, ANCHOR_TOL)
        if free_bitmap is None:
            slices_meta.append({"c": c, "n_hat": n_hat, "grid_res": GRID_RES, "v_min": 0, "z_min": 0, "gw": 0, "gh": 0, "free_bitmap": None})
            continue
        v_min, z_min, gw, gh = bbox
        slices_meta.append({"c": c, "n_hat": n_hat, "grid_res": GRID_RES, "v_min": v_min, "z_min": z_min, "gw": gw, "gh": gh, "free_bitmap": free_bitmap})

    method5_voxel_connect(slices_meta, las.header, OUTPUT_LAS)

    print("\u2705 完了: M5（ボクセル接続＋SOR＋白色除去）")
    print(f"  gap=50適用後 中心線点数: {len(through)}")
    print(f"  断面数（内挿）        : {len(centers)}")

if __name__ == "__main__":
    main()
