# -*- coding: utf-8 -*-
"""
【機能】M0-on-M5統合版（3番目判定＋群接続対応・BFS版）
--------------------------------------------------------
- 中心線に沿ってスライスを生成（M5方式）
- Occupancy構築 → Morphology閉処理（固定半径25）＋Downfill
- Z上限を指定（Z_MAX_FOR_NAV）
- 3番目の矩形が1番目に接しているかでモード切替
  ├─ 結合モード: 1番に接する矩形群のみ結合（seed={0}）
  └─ 独立モード: 1〜3番＋それぞれに接する矩形群を全統合（seed={0,1,2}）
- 面積・幅条件 (7㎡, 3m) 適用
- 外周セルを点群化してスライス毎に LAS 出力（緑点）
--------------------------------------------------------
"""

import os
import math
import numpy as np
import laspy
import cv2
from collections import deque

# ===== 入出力 =====
INPUT_LAS  = "/data/0828_01_500_suidoubasi_ue.las"
OUTPUT_DIR = "/output/1010_final_merge_v3_fixed"

# ===== パラメータ =====
UKC = 0.0
BIN_X = 2.0
MIN_PTS_PER_XBIN = 50
GAP_DIST = 50.0
SECTION_INTERVAL = 0.5
LINE_LENGTH = 100.0
SLICE_THICKNESS = 0.30
MIN_PTS_PER_SLICE = 80

Z_MAX_FOR_NAV = 1.9
GRID_RES = 0.10
MORPH_RADIUS = 25  # 固定
USE_ANCHOR_DOWNFILL = True
ANCHOR_Z = 1.50
ANCHOR_TOL = 0.5

MIN_RECT_AREA = 7.0   # m²
MIN_RECT_WIDTH = 3.0  # m

# ==== ユーティリティ ====
def copy_header_with_metadata(src_header):
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales = src_header.scales
    header.offsets = src_header.offsets
    if getattr(src_header, "srs", None): header.srs = src_header.srs
    if getattr(src_header, "vlrs", None): header.vlrs.extend(src_header.vlrs)
    if getattr(src_header, "evlrs", None): header.evlrs.extend(src_header.evlrs)
    return header

def l2(p, q): return math.hypot(q[0]-p[0], q[1]-p[1])

def find_max_rectangle(bitmap_bool: np.ndarray):
    """ヒストグラム法で最大長方形を探索"""
    h, w = bitmap_bool.shape
    height = [0]*w
    best = (0,0,0,0); max_area = 0
    for i in range(h):
        for j in range(w):
            height[j] = height[j]+1 if bitmap_bool[i,j] else 0
        stack=[]; j=0
        while j <= w:
            cur = height[j] if j < w else 0
            if not stack or cur >= height[stack[-1]]:
                stack.append(j); j+=1
            else:
                top_idx = stack.pop()
                width = j if not stack else j-stack[-1]-1
                area = height[top_idx]*width
                if area > max_area:
                    max_area = area
                    top = i - height[top_idx] + 1
                    left = (stack[-1]+1) if stack else 0
                    best = (top, left, height[top_idx], width)
    return best

def downfill_on_closed(closed_uint8, z_min, grid_res, anchor_z, tol):
    """アンカーダウンフィル処理"""
    closed_bool = (closed_uint8 > 0)
    gh, gw = closed_bool.shape
    i_anchor = int(round((anchor_z - z_min) / grid_res))
    pad = max(0, int(np.ceil(tol / grid_res)))
    i_lo = max(0, i_anchor - pad)
    i_hi = min(gh-1, i_anchor + pad)
    if i_lo > gh-1 or i_hi < 0:
        return (closed_bool.astype(np.uint8)*255)
    out = closed_bool.copy()
    for j in range(gw):
        col = closed_bool[:, j]
        if not np.any(col): continue
        if np.any(col[i_lo:i_hi+1]):
            imax = np.max(np.where(col)[0])
            out[:imax+1, j] = True
    return (out.astype(np.uint8)*255)

def is_touching(rectA, rectB):
    """2つの矩形が辺または角で接しているか（1セルの隙間も許容）"""
    tA,lA,hA,wA = rectA["top"], rectA["left"], rectA["h"], rectA["w"]
    tB,lB,hB,wB = rectB["top"], rectB["left"], rectB["h"], rectB["w"]
    if (tA+hA >= tB-1 and tA <= tB+hB+1) and (lA+wA >= lB-1 and lA <= lB+wB+1):
        return True
    return False

def rectangles_on_slice(points_vz, grid_res, morph_radius, use_anchor, anchor_z, anchor_tol):
    """occupancy構築＋（面積/幅フィルタ込み）長方形抽出"""
    rects = []
    if len(points_vz) == 0: return []

    # occupancy grid
    v_min, v_max = points_vz[:,0].min(), points_vz[:,0].max()
    z_min, z_max = points_vz[:,1].min(), points_vz[:,1].max()
    gw = max(1, int(np.ceil((v_max-v_min)/grid_res)))
    gh = max(1, int(np.ceil((z_max-z_min)/grid_res)))
    grid_raw = np.zeros((gh,gw), dtype=np.uint8)

    yi = ((points_vz[:,0]-v_min)/grid_res).astype(int)
    zi = ((points_vz[:,1]-z_min)/grid_res).astype(int)
    ok = (yi>=0)&(yi<gw)&(zi>=0)&(zi<gh)
    grid_raw[zi[ok], yi[ok]] = 255

    # Morphology閉処理＋Downfill
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*morph_radius+1,2*morph_radius+1))
    closed0 = cv2.morphologyEx(grid_raw, cv2.MORPH_CLOSE, kernel)
    closed = downfill_on_closed(closed0, z_min, grid_res, anchor_z, anchor_tol) if use_anchor else closed0
    closed_bool = (closed > 0)
    free_bitmap = ~closed_bool
    free_work = free_bitmap.copy()

    while np.any(free_work):
        top, left, h, w = find_max_rectangle(free_work)
        if h == 0 or w == 0: break
        width = w * grid_res
        height = h * grid_res
        area = width * height
        # 面積・幅フィルタ
        if area >= MIN_RECT_AREA and width >= MIN_RECT_WIDTH:
            rects.append({
                "top": top, "left": left, "h": h, "w": w,
                "v_min": v_min, "z_min": z_min, "grid_res": grid_res,
                "area": area
            })
        free_work[top:top+h, left:left+w] = False
    return rects

def bfs_connected_indices(rects, seed_indices):
    """接触関係グラフ上で seed から到達可能な矩形のインデックス集合（BFS）"""
    n = len(rects)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if is_touching(rects[i], rects[j]):
                adj[i].append(j)
                adj[j].append(i)
    visited = set(seed_indices)
    q = deque(seed_indices)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                q.append(v)
    return sorted(visited)

def merge_rectangles(rects, idx_list):
    """指定された矩形群を統合（外周矩形化）"""
    tops    = [rects[i]["top"]           for i in idx_list]
    lefts   = [rects[i]["left"]          for i in idx_list]
    bottoms = [rects[i]["top"]+rects[i]["h"]  for i in idx_list]
    rights  = [rects[i]["left"]+rects[i]["w"] for i in idx_list]
    v_min   = rects[0]["v_min"]; z_min = rects[0]["z_min"]; grid_res = rects[0]["grid_res"]
    return [{
        "top": min(tops), "left": min(lefts),
        "h":   max(bottoms)-min(tops),
        "w":   max(rights)-min(lefts),
        "v_min": v_min, "z_min": z_min, "grid_res": grid_res
    }]

def rect_to_points(rect):
    """長方形の外周を点群化"""
    top, left, h, w = rect["top"], rect["left"], rect["h"], rect["w"]
    v_min, z_min, grid_res = rect["v_min"], rect["z_min"], rect["grid_res"]
    pts = []
    for zi in range(top, top+h):
        for yi in range(left, left+w):
            if zi in (top, top+h-1) or yi in (left, left+w-1):
                v = v_min + (yi+0.5)*grid_res
                z = z_min + (zi+0.5)*grid_res
                pts.append([v, z])
    return pts

def vz_to_world_on_slice(vz, c, n_hat):
    v,z = vz
    p_xy = c + v*n_hat
    return [p_xy[0], p_xy[1], z]

def write_green_las(path, header_src, pts_xyz):
    header = copy_header_with_metadata(header_src)
    las_out = laspy.LasData(header)
    N = len(pts_xyz)
    if N == 0: return
    try:
        las_out.points = laspy.ScaleAwarePointRecord.zeros(N, header=header)
    except AttributeError:
        las_out.points = laspy.PointRecord.zeros(N, header=header)
    pts_xyz = np.asarray(pts_xyz, float)
    las_out.x = pts_xyz[:,0]; las_out.y = pts_xyz[:,1]; las_out.z = pts_xyz[:,2]
    if {"red","green","blue"} <= set(las_out.point_format.dimension_names):
        las_out.red = np.zeros(N, dtype=np.uint16)
        las_out.green = np.full(N, 65535, dtype=np.uint16)
        las_out.blue = np.zeros(N, dtype=np.uint16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    las_out.write(path)
    print(f"✅ 出力: {path} 点数: {N}")

# ========= メイン処理 =========
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    las = laspy.read(INPUT_LAS)
    X,Y,Z = np.asarray(las.x,float), np.asarray(las.y,float), np.asarray(las.z,float)
    xy = np.column_stack([X,Y])

    # --- 中心線抽出 ---
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

    # --- 点間間引き ---
    thinned=[through[0]]
    for p in through[1:]:
        if l2(thinned[-1],p)>=GAP_DIST: thinned.append(p)
    through=np.asarray(thinned,float)

    # --- 補間中心生成 ---
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

    # --- スライス処理 ---
    half_len=LINE_LENGTH*0.5; half_th=SLICE_THICKNESS*0.5
    for i in range(len(centers)-1):
        c=centers[i]; cn=centers[i+1]
        t_vec=cn-c; norm=np.linalg.norm(t_vec)
        if norm<1e-9: continue
        t_hat=t_vec/norm; n_hat=np.array([-t_hat[1],t_hat[0]],float)
        dxy=xy-c
        u=dxy@t_hat; v=dxy@n_hat
        m_band=(np.abs(u)<=half_th)&(np.abs(v)<=half_len)
        m_nav=m_band&(Z<=Z_MAX_FOR_NAV)  # Z下限制限なし
        if np.count_nonzero(m_nav)<MIN_PTS_PER_SLICE: continue

        points_vz=np.column_stack([v[m_nav],Z[m_nav]])
        rects = rectangles_on_slice(points_vz, GRID_RES, MORPH_RADIUS,
                                    USE_ANCHOR_DOWNFILL, ANCHOR_Z, ANCHOR_TOL)
        if len(rects) < 1: continue
        rects_sorted = sorted(rects, key=lambda r: r["area"], reverse=True)
        if len(rects_sorted) < 3:  # 3番目判定のため
            # 3つ未満でも seed={0} で結合モード的に処理する選択もあるが、ここではskip
            continue

        # --- モード判定：3番目が1番目に接しているか ---
        mode = "merge" if is_touching(rects_sorted[0], rects_sorted[2]) else "independent"

        if mode == "merge":
            # seed={0}（1番に接続する群のみ）
            idxs = bfs_connected_indices(rects_sorted, seed_indices=[0])
        else:
            # seed={0,1,2}（1〜3番それぞれに接続する群を全て）
            idxs = bfs_connected_indices(rects_sorted, seed_indices=[0,1,2])

        merged = merge_rectangles(rects_sorted, idxs)

        GREEN=[]
        for rect in merged:
            for vv,zz in rect_to_points(rect):
                GREEN.append(vz_to_world_on_slice([vv,zz], c, n_hat))

        out_path = os.path.join(OUTPUT_DIR, f"slice_{i:04d}_rect.las")
        write_green_las(out_path, las.header, GREEN)

if __name__=="__main__":
    main()
