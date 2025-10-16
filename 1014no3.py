# -*- coding: utf-8 -*-
"""
【機能】緑線(最緩線, 極値PCA版)＋緑壁(縦積み)＋XYグリッド塗りつぶしで外側削除
-------------------------------------------------------------------
- PCAで幅方向を推定し、左右端の上下端(Zmin/Zmax)＝4隅を取得
- 初期接続線＋最緩線（緑）を生成（元ロジック維持）
- 緑線をZ方向に縦積み→壁→flood fillで外側削除
- 出力LAS＝灰(内側航行空間)＋緑(最緩線)
-------------------------------------------------------------------
出力: /workspace/output/1014_navspace_centercut_innertrim_gridfill_extreme.las
"""

import os, re
import numpy as np
import laspy
from glob import glob
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
import cv2

# ===== 入出力 =====
INPUT_DIR = "/workspace/output/917slices_m0style_rect/"
OUTPUT_LAS_FINAL = "/workspace/output/1014_navspace_centercut_innertrim_gridfill_extreme.las"

# ===== パラメータ =====
ANGLE_THRESH_DEG = 35.0
LOOKAHEAD_SLICES = 30
LINE_STEP = 0.01
UNION_EPS = 1e-6
Z_MIN_FOR_NAV = -3.0
Z_MAX_FOR_NAV = 1.9
Z_STEP = 0.05
GRID_RES = Z_STEP
DILATE_ITER = 2

# ===== 色 =====
COLOR_INNER = (52000, 52000, 52000)
COLOR_GREEN = (0, 65535, 0)

# ===== 関数 =====
def copy_header_with_metadata(src_header):
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales, header.offsets = src_header.scales, src_header.offsets
    if getattr(src_header, "srs", None): header.srs = src_header.srs
    if getattr(src_header, "vlrs", None): header.vlrs.extend(src_header.vlrs)
    if getattr(src_header, "evlrs", None): header.evlrs.extend(src_header.evlrs)
    return header

def ensure_points_alloc(las_out, n):
    try:
        las_out.points = laspy.ScaleAwarePointRecord.zeros(n, header=las_out.header)
    except AttributeError:
        las_out.points = laspy.PointRecord.zeros(n, header=las_out.header)

def angle_turn_deg(p_prev, p_curr, p_next):
    a = np.asarray(p_prev[:2]) - np.asarray(p_curr[:2])
    b = np.asarray(p_next[:2]) - np.asarray(p_curr[:2])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9: return 0.0
    cosv = np.clip(np.dot(a,b)/(na*nb), -1.0, 1.0)
    inner = np.degrees(np.arccos(cosv))
    return abs(inner - 180.0)

def get_extreme_points_pca(pts_xyz):
    """PCAで幅方向を推定し、左右端の上下端(Zmin/Zmax)を抽出"""
    if len(pts_xyz) < 4:
        return None
    xy = pts_xyz[:, :2]
    mu = xy.mean(axis=0)
    A = xy - mu
    C = A.T @ A / max(1, len(A)-1)
    w, V = np.linalg.eigh(C)
    axis = V[:, np.argmax(w)]
    vcoord = A @ axis
    vmin, vmax = vcoord.min(), vcoord.max()
    if vmax - vmin < 1e-6:
        return None

    band = max(0.02*(vmax-vmin), 0.05)
    left_pts = pts_xyz[vcoord <= vmin + band]
    right_pts = pts_xyz[vcoord >= vmax - band]
    if len(left_pts) == 0 or len(right_pts) == 0:
        return None

    left_low = left_pts[np.argmin(left_pts[:,2])]
    left_high = left_pts[np.argmax(left_pts[:,2])]
    right_low = right_pts[np.argmin(right_pts[:,2])]
    right_high = right_pts[np.argmax(right_pts[:,2])]
    return np.array([left_low, left_high, right_low, right_high])

def rect_polygon_from_corners(c4):
    LL, LU, RL, RU = c4
    return Polygon([(LL[0],LL[1]), (RL[0],RL[1]), (RU[0],RU[1]), (LU[0],LU[1])])

def clip_and_sample_inside(p1, p2, poly_union, step):
    line = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
    inter = line.intersection(poly_union)
    if inter.is_empty: return np.empty((0,3), float)
    segs = [inter] if isinstance(inter, LineString) else list(inter.geoms)
    out = []
    v2 = np.asarray(p2[:2]) - np.asarray(p1[:2])
    vv = max(np.dot(v2,v2), 1e-12)
    for seg in segs:
        coords = np.asarray(seg.coords, float)
        for s in range(len(coords)-1):
            a2, b2 = coords[s], coords[s+1]
            d2 = np.linalg.norm(b2-a2)
            if d2 < 1e-9: continue
            n = max(1,int(np.ceil(d2/step)))
            t = np.linspace(0,1,n+1)
            xy = a2[None,:] + (b2-a2)[None,:]*t[:,None]
            proj = np.dot(xy - np.asarray(p1[:2])[None,:], v2)/vv
            proj = np.clip(proj, 0.0, 1.0)
            z = p1[2] + (p2[2]-p1[2])*proj
            out.append(np.column_stack([xy,z]))
    return np.vstack(out) if out else np.empty((0,3), float)

def flood_fill_union_inside(wall_bool, seeds_xy_idx):
    H, W = wall_bool.shape
    inside = np.zeros((H, W), np.uint8)
    base_mask = np.zeros((H+2, W+2), np.uint8)
    base_mask[1:H+1, 1:W+1][wall_bool] = 1
    for (xi, yi) in seeds_xy_idx:
        if not (0 <= xi < W and 0 <= yi < H):
            continue
        if wall_bool[yi, xi]:
            found=False
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    xj,yj=xi+dx,yi+dy
                    if 0<=xj<W and 0<=yj<H and not wall_bool[yj,xj]:
                        xi,yi=xj,yj;found=True;break
                if found:break
            if not found: continue
        mask=base_mask.copy()
        img_dummy=np.zeros((H,W),np.uint8)
        flags=cv2.FLOODFILL_MASK_ONLY|4|(255<<8)
        cv2.floodFill(img_dummy,mask,(xi,yi),0,flags=flags)
        filled=(mask[1:H+1,1:W+1]==255)
        inside|=filled.astype(np.uint8)
    return (inside>0)

# ===== メイン =====
def main():
    all_files = sorted(glob(os.path.join(INPUT_DIR, "*.las")))
    slice_files = []
    for f in all_files:
        m = re.search(r"slice_(\d+)_rect\.las", os.path.basename(f))
        if m:
            slice_files.append((int(m.group(1)), f))
    slice_files = [f for _, f in sorted(slice_files)]

    if not slice_files:
        raise RuntimeError("スライスが見つかりません (pattern mismatch)")

    raw_seq, corners_seq = [], []
    for f in slice_files:
        las = laspy.read(f)
        P = np.column_stack([las.x, las.y, las.z])
        raw_seq.append(P)
        c4 = get_extreme_points_pca(P)
        if c4 is not None:
            corners_seq.append(c4)

    N = len(corners_seq)
    print(f"✅ 有効スライス数: {N}")
    if N < 2: raise RuntimeError("スライスが少なすぎます")

    connect_to = {c: np.array([i+1 for i in range(N)], int) for c in range(4)}
    series = {c: np.array([corners_seq[i][c] for i in range(N)]) for c in range(4)}
    rect_polys = [rect_polygon_from_corners(corners_seq[k]) for k in range(N)]

    # 初期接続
    bridge_initial=[]
    for i in range(N-1):
        corridor=unary_union([rect_polys[i],rect_polys[i+1]]).buffer(UNION_EPS)
        for c in range(4):
            seg=clip_and_sample_inside(series[c][i],series[c][i+1],corridor,LINE_STEP)
            if seg.size>0:bridge_initial.append(seg)
    bridge_initial=np.vstack(bridge_initial) if bridge_initial else np.empty((0,3),float)

    # 最緩線
    disabled={c:np.zeros(N,bool) for c in range(4)}
    for i in range(1,N-1):
        needs_reconnect=any(angle_turn_deg(series[c][i-1],series[c][i],series[c][i+1])>=ANGLE_THRESH_DEG for c in range(4) if not disabled[c][i])
        if not needs_reconnect:continue
        last=min(N-1,i+LOOKAHEAD_SLICES)
        best_j,best_score=i+1,(1e18,1e18,1e18)
        for j in range(i+2,last+1):
            angs,dsum=[],0.0
            for c in range(4):
                p_prev,p_curr,p_j=series[c][i-1],series[c][i],series[c][j]
                angs.append(angle_turn_deg(p_prev,p_curr,p_j))
                dsum+=np.linalg.norm(series[c][j,:2]-series[c][i,:2])
            cand=(np.mean(angs),dsum,j-i)
            if cand<best_score:best_score,best_j=cand,j
        if best_j!=i+1:
            for c in range(4):
                connect_to[c][i]=best_j;disabled[c][i+1:best_j]=True

    bridge_pts_list=[]
    for i in range(N-1):
        j=int(connect_to[1][i])
        if j<=i or j>=N:continue
        corridor=unary_union([rect_polys[k] for k in range(i,j+1)]).buffer(UNION_EPS)
        for c in range(4):
            seg=clip_and_sample_inside(series[c][i],series[c][j],corridor,LINE_STEP)
            if seg.size>0:bridge_pts_list.append(seg)
    bridge_pts=np.vstack(bridge_pts_list) if bridge_pts_list else np.empty((0,3),float)

    map_pts=np.vstack(raw_seq)

    # --- グリッド塗りつぶし ---
    x_min,x_max=map_pts[:,0].min(),map_pts[:,0].max()
    y_min,y_max=map_pts[:,1].min(),map_pts[:,1].max()
    nx=int(np.ceil((x_max-x_min)/GRID_RES))+1
    ny=int(np.ceil((y_max-y_min)/GRID_RES))+1

    def to_idx_xy(arr_xy):
        xi=((arr_xy[:,0]-x_min)/GRID_RES).astype(int)
        yi=((arr_xy[:,1]-y_min)/GRID_RES).astype(int)
        xi=np.clip(xi,0,nx-1)
        yi=np.clip(yi,0,ny-1)
        return xi,yi

    wall_bool=np.zeros((ny,nx),np.uint8)
    xi,yi=to_idx_xy(bridge_pts[:,:2])
    wall_bool[yi,xi]=1
    if DILATE_ITER>0:
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        wall_bool=cv2.dilate(wall_bool,kernel,iterations=DILATE_ITER)
    wall_bool=(wall_bool>0)

    seeds=[]
    for poly in rect_polys:
        c=poly.centroid
        xi_c=int(round((c.x-x_min)/GRID_RES))
        yi_c=int(round((c.y-y_min)/GRID_RES))
        if 0<=xi_c<nx and 0<=yi_c<ny:seeds.append((xi_c,yi_c))

    inside_bool=flood_fill_union_inside(wall_bool,seeds)
    xi_pts,yi_pts=to_idx_xy(map_pts[:,:2])
    keep_mask=inside_bool[yi_pts,xi_pts]
    map_pts_trim=map_pts[keep_mask]

    # 出力
    out_xyz=np.vstack([map_pts_trim,bridge_pts])
    color_all=np.vstack([
        np.tile(COLOR_INNER,(len(map_pts_trim),1)),
        np.tile(COLOR_GREEN,(len(bridge_pts),1))
    ])
    header=copy_header_with_metadata(laspy.read(slice_files[0]).header)
    las_out=laspy.LasData(header)
    ensure_points_alloc(las_out,len(out_xyz))
    las_out.x,las_out.y,las_out.z=out_xyz[:,0],out_xyz[:,1],out_xyz[:,2]
    las_out.red,las_out.green,las_out.blue=color_all[:,0],color_all[:,1],color_all[:,2]
    os.makedirs(os.path.dirname(OUTPUT_LAS_FINAL),exist_ok=True)
    las_out.write(OUTPUT_LAS_FINAL)

    print(f"✅ 出力完了: {OUTPUT_LAS_FINAL}")
    print(f"  航行空間(灰): {len(map_pts_trim):,} / 全体: {len(map_pts):,}")
    print(f"  緑線: {len(bridge_pts):,} 点, グリッド: {nx}×{ny}, 解像度={GRID_RES}")

if __name__ == "__main__":
    main()
