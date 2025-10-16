# -*- coding: utf-8 -*-
"""
【機能（M0 + 3D連結性付き）】
- 入力LASを読み込み、中心線に沿ったスライスを生成
- 各スライスで v-z occupancy を生成し free_bitmap を保存
- 全スライスの free_bitmap を積層し、3D occupancy として連結成分ラベリング
- 最大コンポーネントに属するスライスに対してのみ:
    - M0方式で最大長方形を探索（最初の長方形に接していない長方形は無視）
    - 縁セルを緑点に変換して出力
"""

import os
import math
import numpy as np
import laspy
import cv2
from scipy import ndimage

# ===== 入出力 =====
INPUT_LAS  = "/data/0828_01_500_suidoubasi_ue.las"
OUTPUT_LAS = r"/output/0908no8_ue_M0_connected.las"

# ===== パラメータ =====
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
MIN_RECT_SIZE = 5

# ==== ユーティリティ関数 ====
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

def find_max_rectangle(bitmap_bool: np.ndarray):
    h, w = bitmap_bool.shape
    height = [0]*w; best = (0, 0, 0, 0); max_area = 0
    for i in range(h):
        for j in range(w): height[j] = height[j] + 1 if bitmap_bool[i, j] else 0
        stack = []; j = 0
        while j <= w:
            cur = height[j] if j < w else 0
            if not stack or cur >= height[stack[-1]]:
                stack.append(j); j += 1
            else:
                top_idx = stack.pop()
                width = j if not stack else j - stack[-1] - 1
                area = height[top_idx]*width
                if area > max_area:
                    max_area = area
                    top = i - height[top_idx] + 1
                    left = (stack[-1] + 1) if stack else 0
                    best = (top, left, height[top_idx], width)
    return best

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
        if not np.any(col): continue
        if np.any(col[i_lo:i_hi+1]):
            imax = np.max(np.where(col)[0])
            out[:imax+1, j] = True
    return (out.astype(np.uint8) * 255)

def rectangles_on_slice(points_vz, grid_res, morph_radius, use_anchor, anchor_z, anchor_tol, min_rect_size):
    rect_edge_pts_vz = []
    if len(points_vz) == 0:
        return rect_edge_pts_vz, None, None
    v_min, v_max = points_vz[:,0].min(), points_vz[:,0].max()
    z_min, z_max = points_vz[:,1].min(), points_vz[:,1].max()
    gw = max(1, int(np.ceil((v_max - v_min) / grid_res)))
    gh = max(1, int(np.ceil((z_max - z_min) / grid_res)))
    grid_raw = np.zeros((gh, gw), dtype=np.uint8)
    yi = ((points_vz[:,0] - v_min) / grid_res).astype(int)
    zi = ((points_vz[:,1] - z_min) / grid_res).astype(int)
    ok = (yi >= 0) & (yi < gw) & (zi >= 0) & (zi < gh)
    grid_raw[zi[ok], yi[ok]] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*morph_radius+1, 2*morph_radius+1))
    closed0 = cv2.morphologyEx(grid_raw, cv2.MORPH_CLOSE, kernel)
    closed = downfill_on_closed(closed0, z_min, grid_res, anchor_z, anchor_tol) if use_anchor else closed0
    closed_bool = (closed > 0)
    free_bitmap = ~closed_bool
    def has_points_above_after_interp(top, left, h, w):
        gh_, gw_ = closed_bool.shape
        z_above_start = top + h
        if z_above_start >= gh_: return False
        sub = closed_bool[z_above_start:gh_, left:left+w]
        return np.any(sub)
    free_work = free_bitmap.copy()
    first_rect = None
    while np.any(free_work):
        top, left, h, w = find_max_rectangle(free_work)
        if h < min_rect_size or w < min_rect_size:
            break
        if not has_points_above_after_interp(top, left, h, w):
            if first_rect is None:
                first_rect = (top, left, h, w)
            else:
                ft, fl, fh, fw = first_rect
                if not (top+h >= ft-1 and top <= ft+fh+1 and left+w >= fl-1 and left <= fl+fw+1):
                    free_work[top:top+h, left:left+w] = False
                    continue
            for zi_ in range(top, top+h):
                for yi_ in range(left, left+w):
                    if zi_ in (top, top+h-1) or yi_ in (left, left+w-1):
                        v = v_min + (yi_ + 0.5) * grid_res
                        z = z_min + (zi_ + 0.5) * grid_res
                        rect_edge_pts_vz.append([v, z])
        free_work[top:top+h, left:left+w] = False
    bbox = (v_min, z_min, gw, gh)
    return rect_edge_pts_vz, bbox, free_bitmap

def vz_to_world_on_slice(vz, c, n_hat):
    v, z = vz
    p_xy = c + v * n_hat
    return [p_xy[0], p_xy[1], z]

def write_green_las(path, header_src, pts_xyz):
    header = copy_header_with_metadata(header_src)
    las_out = laspy.LasData(header)
    N = len(pts_xyz)
    las_out.points = laspy.ScaleAwarePointRecord.zeros(N, header=header)
    pts_xyz = np.asarray(pts_xyz, float)
    las_out.x = pts_xyz[:,0]; las_out.y = pts_xyz[:,1]; las_out.z = pts_xyz[:,2]
    if {"red","green","blue"} <= set(las_out.point_format.dimension_names):
        las_out.red = np.zeros(N, dtype=np.uint16)
        las_out.green = np.full(N, 65535, dtype=np.uint16)
        las_out.blue = np.zeros(N, dtype=np.uint16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    las_out.write(path)
    print(f"✅ 出力: {path} 点数: {N}")

def main():
    las = laspy.read(INPUT_LAS)
    X, Y, Z = np.asarray(las.x,float), np.asarray(las.y,float), np.asarray(las.z,float)
    xy = np.column_stack([X, Y])

    # 中心線 → スライス中心生成
    x_min, x_max = xy[:,0].min(), xy[:,0].max()
    edges = np.arange(x_min, x_max + BIN_X, BIN_X)
    through = []
    for i in range(len(edges)-1):
        x0,x1 = edges[i], edges[i+1]
        m = (xy[:,0]>=x0)&(xy[:,0]<x1)
        if np.count_nonzero(m)<MIN_PTS_PER_XBIN: continue
        slab_xy, slab_z = xy[m], Z[m]
        order = np.argsort(slab_xy[:,1])
        slab_xy, slab_z = slab_xy[order], slab_z[order]
        under = slab_z <= UKC
        if not np.any(under): continue
        idx = np.where(under)[0]
        left, right = slab_xy[idx[0]], slab_xy[idx[-1]]
        through.append(0.5*(left+right))
    through = np.asarray(through,float)
    centers=[]
    for i in range(len(through)-1):
        p,q=through[i],through[i+1]; d=l2(p,q)
        if d<1e-9: continue
        n_steps=int(d/SECTION_INTERVAL)
        for s_i in range(n_steps+1):
            s=min(s_i*SECTION_INTERVAL,d)
            t=s/d
            centers.append((1-t)*p+t*q)
    centers=np.asarray(centers,float)

    # スライス内処理 + free_bitmap 蓄積
    slices_meta = []
    half_len = LINE_LENGTH * 0.5
    half_th  = SLICE_THICKNESS * 0.5
    for i in range(len(centers)-1):
        c, cn = centers[i], centers[i+1]
        t_vec = cn - c
        norm = np.linalg.norm(t_vec)
        if norm < 1e-9: continue
        t_hat = t_vec / norm
        n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)
        dxy = xy - c
        u = dxy @ t_hat
        v = dxy @ n_hat
        m_band = (np.abs(u) <= half_th) & (np.abs(v) <= half_len)
        m_nav = m_band & (Z <= Z_MAX_FOR_NAV)
        if np.count_nonzero(m_nav) < MIN_PTS_PER_SLICE:
            slices_meta.append({"free_bitmap": None})
            continue
        points_vz = np.column_stack([v[m_nav], Z[m_nav]])
        _, bbox, free_bitmap = rectangles_on_slice(points_vz, GRID_RES, MORPH_RADIUS, USE_ANCHOR_DOWNFILL, ANCHOR_Z, ANCHOR_TOL, MIN_RECT_SIZE)
        slices_meta.append({"c": c, "n_hat": n_hat, "free_bitmap": free_bitmap, "bbox": bbox, "points_vz": points_vz})

    # 3D occupancy連結性チェック
    valid = [s for s in slices_meta if s["free_bitmap"] is not None]
    if not valid: raise RuntimeError("有効なスライスなし")
    v_all_min = min(s["bbox"][0] for s in valid)
    z_all_min = min(s["bbox"][1] for s in valid)
    v_all_max = max(s["bbox"][0] + s["bbox"][2]*GRID_RES for s in valid)
    z_all_max = max(s["bbox"][1] + s["bbox"][3]*GRID_RES for s in valid)
    gw = int(np.ceil((v_all_max - v_all_min)/GRID_RES))
    gh = int(np.ceil((z_all_max - z_all_min)/GRID_RES))
    gu = len(slices_meta)
    mask = np.zeros((gu, gh, gw), dtype=bool)
    for u, s in enumerate(slices_meta):
        fb = s["free_bitmap"]
        if fb is None: continue
        v_min, z_min, w, h = s["bbox"]
        off_v = int(round((v_min - v_all_min)/GRID_RES))
        off_z = int(round((z_min - z_all_min)/GRID_RES))
        mask[u, off_z:off_z+h, off_v:off_v+w] = fb
    labeled, nf = ndimage.label(mask)
    keep_mask = np.ones_like(mask, dtype=bool)
    if nf > 1:
        counts = np.bincount(labeled.ravel()); counts[0] = 0
        keep = counts.argmax()
        keep_mask = (labeled == keep)

    # 長方形抽出（最大連結成分のみ）
    GREEN = []
    for u, s in enumerate(slices_meta):
        if s["free_bitmap"] is None: continue
        if not np.any(keep_mask[u]): continue
        c, n_hat = s["c"], s["n_hat"]
        rect_edges_vz, _, _ = rectangles_on_slice(s["points_vz"], GRID_RES, MORPH_RADIUS, USE_ANCHOR_DOWNFILL, ANCHOR_Z, ANCHOR_TOL, MIN_RECT_SIZE)
        for vv, zz in rect_edges_vz:
            GREEN.append(vz_to_world_on_slice([vv, zz], c, n_hat))

    if not GREEN:
        raise RuntimeError("連結スライスで長方形が見つかりませんでした。")
    write_green_las(OUTPUT_LAS, las.header, GREEN)
    print("✅ M0+連結性スライス 完了 点数:", len(GREEN))

if __name__ == "__main__":
    main()
