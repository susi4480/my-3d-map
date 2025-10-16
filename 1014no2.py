# -*- coding: utf-8 -*-
"""
【機能】PCA＋フォールバック＋スライス中心基準（緑線強調＋“緑より外側”トリミング）
-------------------------------------------------------------------
- 各スライスの矩形（slice_XXXX_rect.las）を読み込み、PCAで4隅推定（失敗時AABB）
- 初期接続線（内部用）＋最緩線（緑）を生成（※生成ロジックは元コードと同一）
- 各スライスで、スライス中心 c_i から見て「緑点と同じ側（外側）」の航行可能点を削除
  * 左右判定は、(LL,LU)の中点→(RL,RU)の中点ベクトルを n_hat とし、(p - c_i)·n_hat の符号で行う
- 出力LAS = 灰（トリミング後の航行空間）＋ 緑（最緩線） ※水色・赤は非出力
-------------------------------------------------------------------
出力: /workspace/output/1014_navspace_centercut_innertrim_slicenormal.las
"""

import os, re
import numpy as np
import laspy
from glob import glob
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union

# ===== 入出力 =====
INPUT_DIR = "/workspace/output/917slices_m0style_rect/"
OUTPUT_LAS_FINAL = "/workspace/output/1014_navspace_centercut_innertrim_slicenormal.las"

# ===== パラメータ（※緑線生成は元コードと同一パラメータ） =====
ANGLE_THRESH_DEG = 35.0
LOOKAHEAD_SLICES = 30
LINE_STEP = 0.01
UNION_EPS = 1e-6

# ===== 着色 =====
COLOR_INNER = (52000, 52000, 52000)  # 灰（トリミング後）
COLOR_GREEN = (0, 65535, 0)          # 緑（最緩線）

# ===== ユーティリティ =====
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

def pca_rect_corners_safe(pts):
    """PCAベース＋外接矩形（失敗時はAABB）"""
    if pts.shape[0] < 4: return None
    try:
        xy = pts[:, :2]; c = xy.mean(axis=0)
        X = xy - c; C = np.cov(X.T)
        _, _, VT = np.linalg.svd(C); R = VT.T
        uv = X @ R
        umin, vmin = uv.min(axis=0); umax, vmax = uv.max(axis=0)
        corners_uv = np.array([[umin,vmin],[umin,vmax],[umax,vmin],[umax,vmax]])
        corners_xy = corners_uv @ R.T + c
        z_med = np.median(pts[:,2])
        return np.column_stack([corners_xy, np.full(4,z_med)])
    except:
        xy = pts[:, :2]
        xmin, ymin = np.min(xy, axis=0); xmax, ymax = np.max(xy, axis=0)
        z_med = np.median(pts[:,2])
        return np.array([[xmin,ymin,z_med],[xmin,ymax,z_med],[xmax,ymin,z_med],[xmax,ymax,z_med]])

def rect_polygon_from_corners(c4):
    LL, LU, RL, RU = c4
    ring = [tuple(LL[:2]), tuple(RL[:2]), tuple(RU[:2]), tuple(LU[:2])]
    return Polygon(ring)

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
            if d2<1e-9: continue
            n = max(1,int(np.ceil(d2/step)))
            t = np.linspace(0,1,n+1)
            xy = a2[None,:] + (b2-a2)[None,:]*t[:,None]
            proj = np.dot(xy - np.asarray(p1[:2])[None,:], v2)/vv
            proj = np.clip(proj, 0.0, 1.0)
            z = p1[2] + (p2[2]-p1[2])*proj
            out.append(np.column_stack([xy,z]))
    return np.vstack(out) if out else np.empty((0,3), float)

# ===== メイン処理 =====
def main():
    # スライス矩形（緑エッジ群）読み込み
    slice_files = sorted(
        glob(os.path.join(INPUT_DIR, "slice_*_rect.las")),
        key=lambda f: int(re.search(r"slice_(\d+)_rect\.las", os.path.basename(f)).group(1))
    )
    if not slice_files:
        raise RuntimeError("スライスが見つかりません")

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
    if N < 2:
        raise RuntimeError("スライスが少なすぎます")

    # 隅系列・矩形ポリゴン
    connect_to = {c: np.array([i+1 for i in range(N)], int) for c in range(4)}
    series = {c: np.array([corners_seq[i][c] for i in range(N)]) for c in range(4)}
    rect_polys = [rect_polygon_from_corners(corners_seq[k]) for k in range(N)]

    # --- 初期接続（内部用：水色は出力しない） ---
    bridge_initial = []
    for i in range(N-1):
        corridor = unary_union([rect_polys[i], rect_polys[i+1]]).buffer(UNION_EPS)
        for c in range(4):
            seg = clip_and_sample_inside(series[c][i], series[c][i+1], corridor, LINE_STEP)
            if seg.size > 0:
                bridge_initial.append(seg)
    bridge_initial = np.vstack(bridge_initial) if bridge_initial else np.empty((0,3), float)

    # --- 角度急変の先読み再接続（元コード同様） ---
    disabled = {c: np.zeros(N,bool) for c in range(4)}
    for i in range(1, N-1):
        needs_reconnect = any(
            angle_turn_deg(series[c][i-1], series[c][i], series[c][i+1]) >= ANGLE_THRESH_DEG
            for c in range(4) if not disabled[c][i]
        )
        if not needs_reconnect: continue
        last = min(N-1, i+LOOKAHEAD_SLICES)
        best_j, best_score = i+1, (1e18,1e18,1e18)
        for j in range(i+2, last+1):
            angs, dsum = [], 0.0
            for c in range(4):
                p_prev, p_curr, p_j = series[c][i-1], series[c][i], series[c][j]
                angs.append(angle_turn_deg(p_prev, p_curr, p_j))
                dsum += np.linalg.norm(series[c][j,:2]-series[c][i,:2])
            cand = (np.mean(angs), dsum, j-i)
            if cand < best_score:
                best_score, best_j = cand, j
        if best_j != i+1:
            for c in range(4):
                connect_to[c][i] = best_j
                disabled[c][i+1:best_j] = True

    # --- 最緩線（緑） ---
    bridge_pts_list = []
    for i in range(N-1):
        j = int(connect_to[1][i])
        if j <= i or j >= N: continue
        corridor = unary_union([rect_polys[k] for k in range(i, j+1)]).buffer(UNION_EPS)
        for c in range(4):
            seg = clip_and_sample_inside(series[c][i], series[c][j], corridor, LINE_STEP)
            if seg.size > 0:
                bridge_pts_list.append(seg)
    bridge_pts = np.vstack(bridge_pts_list) if bridge_pts_list else np.empty((0,3), float)

    # --- 航行空間（灰＝元スライス点群の集約）
    map_pts = np.vstack(raw_seq)

    # --- スライスごとに「緑より外側」を削除（座標軸に依存しないローカル判定）
    #     ・スライス中心 c_i = polygon.centroid
    #     ・左右方向 n_hat = mid(右側) - mid(左側)
    #     ・緑の代表 g_i = そのスライス内の緑点の平均
    keep_mask_global = np.ones(len(map_pts), dtype=bool)

    for i in range(N):
        poly = rect_polys[i]
        if poly.is_empty: continue

        # スライス中心
        c_i = np.array([poly.centroid.x, poly.centroid.y], dtype=float)

        # 左右中点（LL,LUの中点 / RL,RUの中点）から横断方向ベクトルを作る
        LL, LU, RL, RU = corners_seq[i]
        left_mid  = 0.5*(LL[:2] + LU[:2])
        right_mid = 0.5*(RL[:2] + RU[:2])
        n = right_mid - left_mid
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            # フォールバック：ポリゴンの長辺方向から横断方向を推定
            n = np.array([1.0, 0.0], float)
        else:
            n = n / n_norm  # n_hat

        # このスライスに属する灰点（多いのでまずはAABBで粗選別→polygon.containsで厳密化）
        minx, miny, maxx, maxy = poly.bounds
        rough_idx = np.where(
            (map_pts[:,0] >= minx) & (map_pts[:,0] <= maxx) &
            (map_pts[:,1] >= miny) & (map_pts[:,1] <= maxy)
        )[0]
        if len(rough_idx) == 0: continue

        # 厳密にポリゴン内だけに絞る
        inside_idx = []
        for k in rough_idx:
            if poly.contains(Point(map_pts[k,0], map_pts[k,1])):
                inside_idx.append(k)
        if len(inside_idx) == 0: continue
        inside_idx = np.array(inside_idx, dtype=int)
        pts_xy = map_pts[inside_idx, :2]

        # スライス内の緑代表点 g_i（平均）
        greens_in_slice = bridge_pts[
            (bridge_pts[:,0] >= minx) & (bridge_pts[:,0] <= maxx) &
            (bridge_pts[:,1] >= miny) & (bridge_pts[:,1] <= maxy)
        ]
        if len(greens_in_slice) == 0:
            # 緑が無い場合はスキップ（削らない）
            continue
        g_i = greens_in_slice[:, :2].mean(axis=0)

        # 「緑より外側」判定：
        # ・スライス中心から見て、(p - c_i)·n と (g_i - c_i)·n の符号が同じなら“外側”として削除
        sign_g = np.sign(np.dot(g_i - c_i, n))
        if sign_g == 0.0:
            # ほぼ中心上：外側定義が曖昧なので削らない
            continue

        proj_pts = (pts_xy - c_i[None,:]) @ n
        same_side = (np.sign(proj_pts) == sign_g)
        # 同じ側（=緑と同側）は削除
        keep_mask_global[inside_idx[same_side]] = False

    # トリミング後の灰点群
    map_pts_trim = map_pts[keep_mask_global]

    # --- 出力（灰：トリミング後のみ、緑：最緩線）
    out_xyz = np.vstack([map_pts_trim, bridge_pts])
    color_all = np.vstack([
        np.tile(COLOR_INNER, (len(map_pts_trim), 1)),
        np.tile(COLOR_GREEN, (len(bridge_pts), 1))
    ])

    header = copy_header_with_metadata(laspy.read(slice_files[0]).header)
    las_out = laspy.LasData(header)
    ensure_points_alloc(las_out, len(out_xyz))
    las_out.x, las_out.y, las_out.z = out_xyz[:,0], out_xyz[:,1], out_xyz[:,2]
    las_out.red, las_out.green, las_out.blue = color_all[:,0], color_all[:,1], color_all[:,2]

    os.makedirs(os.path.dirname(OUTPUT_LAS_FINAL), exist_ok=True)
    las_out.write(OUTPUT_LAS_FINAL)

    print(f"✅ 出力完了: {OUTPUT_LAS_FINAL}")
    print(f"  航行空間(灰, トリミング後): {len(map_pts_trim):,} 点")
    print(f"  緑線: {len(bridge_pts):,} 点")

if __name__ == "__main__":
    main()
