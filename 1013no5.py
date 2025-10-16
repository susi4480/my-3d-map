# -*- coding: utf-8 -*-
"""
【機能】Unionマスク方式：緑線群＋スライス矩形Unionの内側のみ航行空間（灰）として残す
--------------------------------------------------------------------
- PCA＋フォールバックで各スライス矩形(LL,LU,RL,RU)を抽出
- 初期接続(水色)と最緩再接続(緑線群)を構築（角度最小）
- 各スライス矩形＋緑線群をUnionして「通行可能回廊」を形成
- Union内 → 灰（航行空間）
- Union外 → 赤（除外）
- 出力: 灰=航行空間, 赤=外側, 水色=初期線, 緑=最緩線群
--------------------------------------------------------------------
出力: /workspace/output/1013_navspace_unionmask_all.las
"""

import os, re
import numpy as np
import laspy
from glob import glob
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union

# ===== 入出力 =====
INPUT_DIR = "/workspace/output/917slices_m0style_rect/"
OUTPUT_LAS_FINAL = "/workspace/output/1013_navspace_unionmask_all.las"

# ===== パラメータ =====
ANGLE_THRESH_DEG = 35.0
LOOKAHEAD_SLICES = 30
LINE_STEP = 0.03
UNION_EPS = 1e-6
KEEP_ALL_MAP_PTS = True

# ===== 色設定 =====
COLOR_INNER = (52000, 52000, 52000)  # 灰（Union内：航行空間）
COLOR_OUTER = (65535, 0, 0)          # 赤（Union外：除外）
COLOR_GREEN = (0, 65535, 0)          # 緑（最緩線群）
COLOR_CYAN  = (0, 52000, 65535)      # 水色（初期線）

# ===== 関数群 =====
def copy_header_with_metadata(src_header):
    h = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    h.scales, h.offsets = src_header.scales, src_header.offsets
    if getattr(src_header, "srs", None): h.srs = src_header.srs
    if getattr(src_header, "vlrs", None): h.vlrs.extend(src_header.vlrs)
    if getattr(src_header, "evlrs", None): h.evlrs.extend(src_header.evlrs)
    return h

def ensure_points_alloc(las_out, n):
    try: las_out.points = laspy.ScaleAwarePointRecord.zeros(n, header=las_out.header)
    except: las_out.points = laspy.PointRecord.zeros(n, header=las_out.header)

def angle_turn_deg(p_prev, p_curr, p_next):
    a = np.asarray(p_prev[:2]) - np.asarray(p_curr[:2])
    b = np.asarray(p_next[:2]) - np.asarray(p_curr[:2])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9: return 0.0
    cosv = np.clip(np.dot(a,b)/(na*nb), -1.0, 1.0)
    inner = np.degrees(np.arccos(cosv))
    return abs(inner - 180.0)

def pca_rect_corners_safe(pts):
    if pts.shape[0] < 4: return None
    try:
        xy = pts[:, :2]; c = xy.mean(axis=0)
        X = xy - c
        _, _, VT = np.linalg.svd(np.cov(X.T)); R = VT.T
        uv = X @ R
        umin, vmin = uv.min(axis=0); umax, vmax = uv.max(axis=0)
        corners_uv = np.array([[umin,vmin],[umin,vmax],[umax,vmin],[umax,vmax]], float)
        corners_xy = corners_uv @ R.T + c
        z_med = np.median(pts[:,2])
        return np.column_stack([corners_xy, np.full(4, z_med)])
    except Exception:
        xs = pts[:,0]
        left_mask  = xs <= xs.mean()
        right_mask = ~left_mask
        if np.count_nonzero(left_mask)==0 or np.count_nonzero(right_mask)==0:
            xmin, ymin = np.min(pts[:,:2], axis=0)
            xmax, ymax = np.max(pts[:,:2], axis=0)
            z_med = np.median(pts[:,2])
            return np.array([[xmin,ymin,z_med],[xmin,ymax,z_med],
                             [xmax,ymin,z_med],[xmax,ymax,z_med]], float)
        L = pts[left_mask]; Rg = pts[right_mask]
        LL = L[np.argmin(L[:,2])]; LU = L[np.argmax(L[:,2])]
        RL = Rg[np.argmin(Rg[:,2])]; RU = Rg[np.argmax(Rg[:,2])]
        return np.array([LL, LU, RL, RU], float)

def rect_polygon_from_corners(c4):
    LL, LU, RL, RU = c4
    ring = [tuple(LL[:2]), tuple(RL[:2]), tuple(RU[:2]), tuple(LU[:2])]
    return Polygon(ring)

def clip_and_sample_inside(p1, p2, poly_union, step):
    line = LineString([(p1[0],p1[1]), (p2[0],p2[1])])
    inter = line.intersection(poly_union)
    if inter.is_empty: return np.empty((0,3), float)
    segs = [inter] if isinstance(inter, LineString) else list(inter.geoms)
    out=[]
    v2=np.asarray(p2[:2])-np.asarray(p1[:2])
    vv=max(np.dot(v2,v2),1e-12)
    for seg in segs:
        coords=np.asarray(seg.coords,float)
        for s in range(len(coords)-1):
            a2,b2=coords[s],coords[s+1]
            d2=np.linalg.norm(b2-a2)
            if d2<1e-9: continue
            n=max(1,int(np.ceil(d2/step)))
            t=np.linspace(0.0,1.0,n+1)
            xy=a2[None,:]+(b2-a2)[None,:]*t[:,None]
            proj=np.dot(xy-np.asarray(p1[:2])[None,:],v2)/vv
            proj=np.clip(proj,0.0,1.0)
            z=p1[2]+(p2[2]-p1[2])*proj
            out.append(np.column_stack([xy,z]))
    return np.vstack(out) if out else np.empty((0,3),float)

# ===== メイン =====
def main():
    slice_files = sorted(
        glob(os.path.join(INPUT_DIR,"slice_*_rect.las")),
        key=lambda f:int(re.search(r"slice_(\d+)_rect\.las",os.path.basename(f)).group(1))
    )
    if not slice_files: raise RuntimeError("スライスが見つかりません")

    raw_seq, corners_seq = [], []
    for f in slice_files:
        las = laspy.read(f)
        P = np.column_stack([las.x, las.y, las.z])
        raw_seq.append(P)
        c4 = pca_rect_corners_safe(P)
        if c4 is not None:
            corners_seq.append(c4)
    N = len(corners_seq)
    print(f"✅ 有効スライス数: {N}")

    # --- 初期接続（水色） ---
    rect_polys = [rect_polygon_from_corners(c) for c in corners_seq]
    bridge_initial=[]
    for i in range(N-1):
        corridor = unary_union([rect_polys[i], rect_polys[i+1]]).buffer(UNION_EPS)
        for c in range(4):
            seg = clip_and_sample_inside(corners_seq[i][c], corners_seq[i+1][c], corridor, LINE_STEP)
            if seg.size>0: bridge_initial.append(seg)
    bridge_initial = np.vstack(bridge_initial) if bridge_initial else np.empty((0,3),float)

    # --- 最緩再結合（緑線群） ---
    series = {c: np.array([corners_seq[i][c] for i in range(N)]) for c in range(4)}
    connect_to = {c: np.array([i+1 for i in range(N)], int) for c in range(4)}

    for i in range(1, N-1):
        best_j, best_score = i+1, (1e18,1e18,1e18)
        for j in range(i+2, min(N, i+LOOKAHEAD_SLICES)):
            angs, dsum = [], 0.0
            for c in range(4):
                angs.append(angle_turn_deg(series[c][i-1], series[c][i], series[c][j]))
                dsum += np.linalg.norm(series[c][j,:2]-series[c][i,:2])
            cand = (np.mean(angs), dsum, j-i)
            if cand < best_score:
                best_score, best_j = cand, j
        if best_score[0] < ANGLE_THRESH_DEG:
            for c in range(4): connect_to[c][i] = best_j

    bridge_green=[]
    for i in range(N-1):
        j = connect_to[0][i]
        if j<=i or j>=N: continue
        corridor = unary_union([rect_polys[k] for k in range(i,j+1)]).buffer(UNION_EPS)
        for c in range(4):
            seg = clip_and_sample_inside(series[c][i], series[c][j], corridor, LINE_STEP)
            if seg.size>0: bridge_green.append(seg)
    bridge_green = np.vstack(bridge_green) if bridge_green else np.empty((0,3),float)

    # --- ✅ Unionマスク生成（矩形＋緑線群） ---
    all_geoms = rect_polys.copy()
    if len(bridge_green) > 1:
        all_geoms.append(LineString(bridge_green[:,:2]))
    navigable_union = unary_union(all_geoms).buffer(UNION_EPS)

    # --- map点群のUnion内外で赤/灰に分類 ---
    map_pts = np.vstack(raw_seq) if KEEP_ALL_MAP_PTS else np.empty((0,3),float)
    colors_map = np.zeros((len(map_pts),3),np.uint16)
    if len(map_pts) > 0:
        print("🧭 Union内外判定中...")
        from shapely.strtree import STRtree
        # 高速化: 各点をPoint化
        points = [Point(xy) for xy in map_pts[:,:2]]
        inside_mask = np.array([navigable_union.contains(p) for p in points])
        colors_map[inside_mask]  = COLOR_INNER
        colors_map[~inside_mask] = COLOR_OUTER
        print(f"  内側: {np.sum(inside_mask):,}, 外側: {np.sum(~inside_mask):,}")

    # --- 出力 ---
    out_xyz = np.vstack([map_pts, bridge_initial, bridge_green])
    out_col = np.vstack([
        colors_map,
        np.tile(COLOR_CYAN, (len(bridge_initial),1)),
        np.tile(COLOR_GREEN, (len(bridge_green),1))
    ])

    header = copy_header_with_metadata(laspy.read(slice_files[0]).header)
    las_out = laspy.LasData(header)
    ensure_points_alloc(las_out, len(out_xyz))
    las_out.x, las_out.y, las_out.z = out_xyz[:,0], out_xyz[:,1], out_xyz[:,2]
    las_out.red, las_out.green, las_out.blue = out_col[:,0], out_col[:,1], out_col[:,2]
    os.makedirs(os.path.dirname(OUTPUT_LAS_FINAL),exist_ok=True)
    las_out.write(OUTPUT_LAS_FINAL)

    print(f"✅ 出力完了: {OUTPUT_LAS_FINAL}")
    print(f"  灰(内側): {np.sum(colors_map[:,0]==52000):,}, 赤(外側): {np.sum(colors_map[:,0]==65535):,}")
    print(f"  水線: {len(bridge_initial):,}, 緑線: {len(bridge_green):,}")

if __name__=="__main__":
    main()
