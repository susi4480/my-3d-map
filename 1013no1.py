# -*- coding: utf-8 -*-
"""
【機能】PCAなし版：四隅抽出＋初期接続＋最緩再結合＋航行空間抽出（統合出力）
--------------------------------------------------------------------------------
前提：
- 入力の /workspace/output/917slices_m0style_rect/slice_*_rect.las は
  「中心線ベース（M0 on M5）」で生成され、スライス座標軸が一貫している。

やること：
- 各スライスから PCA を使わずに 4隅（LL,LU,RL,RU）を抽出
  （左右= Xの平均で2分割、各側でZ min/max）
- 隅ごとに i→i+1 の初期接続（水色）
- 急角なら i→(i+2..i+30) の中から最緩（平均角度→距離→近さ）で再結合（緑）
- すべての線は Union(スライス矩形群) 内で完全クリップして 0.03 m 間隔で点群化
- 緑線中心 ±NAV_WIDTH m のバッファ内のみ航行空間（灰）として残す
- 出力は 1 本の LAS（灰＝航行空間、 水色＝初期線、 緑＝最緩線）
"""

import os, re
import numpy as np
import laspy
from glob import glob
from shapely.geometry import Polygon, LineString, MultiLineString, MultiPoint
from shapely.ops import unary_union

# ===== 入出力 =====
INPUT_DIR = "/workspace/output/917slices_m0style_rect/"
OUTPUT_LAS_FINAL = "/workspace/output/1013_noPCA_navspace_filtered_all.las"

# ===== パラメータ =====
ANGLE_THRESH_DEG = 35.0
LOOKAHEAD_SLICES = 30
LINE_STEP = 0.03
UNION_EPS = 1e-6
NAV_WIDTH = 2.5
KEEP_ALL_MAP_PTS = True

# ===== 着色 =====
COLOR_MAP   = (52000, 52000, 52000)  # 灰（航行空間）
COLOR_CYAN  = (0, 52000, 65535)      # 水色（初期接続）
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
    """XY平面での進行角の折れを0～180°で返す（直進=0°）。"""
    a = np.asarray(p_prev[:2]) - np.asarray(p_curr[:2])
    b = np.asarray(p_next[:2]) - np.asarray(p_curr[:2])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cosv = np.clip(np.dot(a,b)/(na*nb), -1.0, 1.0)
    inner = np.degrees(np.arccos(cosv))
    return abs(inner - 180.0)

def interpolate_line_dense(p1, p2, step):
    """3D直線を等間隔サンプル（端点含む）→ (N,3)"""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = np.linalg.norm(p2 - p1)
    if d < 1e-9:
        return p1[None,:]
    n = max(1, int(np.ceil(d/step)))
    t = np.linspace(0.0, 1.0, n+1)
    return p1[None,:] + (p2 - p1)[None,:]*t[:,None]

def get_extreme_corners_simple(pts):
    """
    PCAなしの4隅抽出。
    - 左右：Xの平均で2分割
    - 各側で Z min / Z max を取って [LL, LU, RL, RU] を返す
    """
    if len(pts) == 0:
        return None
    xs = pts[:,0]; zs = pts[:,2]
    left_mask  = xs <= xs.mean()
    right_mask = ~left_mask
    left_pts, right_pts = pts[left_mask], pts[right_mask]
    if len(left_pts)==0 or len(right_pts)==0:
        return None
    left_low   = left_pts[np.argmin(left_pts[:,2])]
    left_high  = left_pts[np.argmax(left_pts[:,2])]
    right_low  = right_pts[np.argmin(right_pts[:,2])]
    right_high = right_pts[np.argmax(right_pts[:,2])]
    return np.array([left_low, left_high, right_low, right_high], float)  # LL, LU, RL, RU

def rect_polygon_from_corners(c4):
    """四隅 [LL, LU, RL, RU] → 外周順（LL→RL→RU→LU）のPolygon"""
    LL, LU, RL, RU = c4
    ring = [tuple(LL[:2]), tuple(RL[:2]), tuple(RU[:2]), tuple(LU[:2])]
    return Polygon(ring)

def clip_and_sample_inside(p1, p2, poly_union, step):
    """
    p1→p2 線分を 回廊(poly_union) でクリップし、内部を等間隔サンプル（Zは線形補間）
    """
    line = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
    inter = line.intersection(poly_union)
    if inter.is_empty:
        return np.empty((0,3), float)

    segs = [inter] if isinstance(inter, LineString) else list(inter.geoms)
    out = []
    v2 = np.asarray(p2[:2]) - np.asarray(p1[:2])
    vv = max(np.dot(v2, v2), 1e-12)
    for seg in segs:
        coords = np.asarray(seg.coords, float)
        for s in range(len(coords)-1):
            a2, b2 = coords[s], coords[s+1]
            d2 = np.linalg.norm(b2 - a2)
            if d2 < 1e-9:
                continue
            n = max(1, int(np.ceil(d2/step)))
            t = np.linspace(0.0, 1.0, n+1)
            xy = a2[None,:] + (b2 - a2)[None,:]*t[:,None]
            proj = np.dot(xy - np.asarray(p1[:2])[None,:], v2)/vv
            proj = np.clip(proj, 0.0, 1.0)
            z = p1[2] + (p2[2]-p1[2]) * proj
            out.append(np.column_stack([xy, z]))
    return np.vstack(out) if out else np.empty((0,3), float)

# ===== メイン =====
def main():
    slice_files = sorted(
        glob(os.path.join(INPUT_DIR, "slice_*_rect.las")),
        key=lambda f: int(re.search(r"slice_(\d+)_rect\.las", os.path.basename(f)).group(1))
    )
    if not slice_files:
        raise RuntimeError("入力スライスLASが見つかりません")

    # 各スライスの点群・四隅
    raw_seq, corners_seq = [], []
    for f in slice_files:
        las = laspy.read(f)
        P = np.column_stack([las.x, las.y, las.z])
        raw_seq.append(P)
        c4 = get_extreme_corners_simple(P)  # ★ PCAなし
        if c4 is not None:
            corners_seq.append(c4)

    if len(corners_seq) < 3:
        raise RuntimeError("有効なスライス（四隅抽出可）が少なすぎます。")

    N = len(corners_seq)
    # 隅ごとに i→i+1 で初期接続（のちに最緩へ更新）
    connect_to = {c: np.array([i+1 for i in range(N)], dtype=int) for c in range(4)}
    series     = {c: np.array([corners_seq[i][c] for i in range(N)]) for c in range(4)}

    # 矩形回廊（Union用）
    rect_polys = [rect_polygon_from_corners(corners_seq[k]) for k in range(N)]

    # --- 初期接続（水色） ---
    bridge_initial = []
    for i in range(N-1):
        corridor = unary_union([rect_polys[i], rect_polys[i+1]]).buffer(UNION_EPS)
        for c in range(4):
            p1, p2 = series[c][i], series[c][i+1]
            seg_pts = clip_and_sample_inside(p1, p2, corridor, LINE_STEP)
            if seg_pts.size > 0:
                bridge_initial.append(seg_pts)
    bridge_initial = np.vstack(bridge_initial) if bridge_initial else np.empty((0,3), float)

    # --- 最緩再結合（緑） ---
    disabled = {c: np.zeros(N, dtype=bool) for c in range(4)}
    for i in range(1, N-1):
        # どれかの隅が急折れなら再結合を検討
        needs_reconnect = False
        for c in range(4):
            if disabled[c][i]:
                continue
            ang = angle_turn_deg(series[c][i-1], series[c][i], series[c][i+1])
            if ang >= ANGLE_THRESH_DEG:
                needs_reconnect = True
        if not needs_reconnect:
            continue

        last = min(N-1, i + LOOKAHEAD_SLICES)
        best_j = i+1
        best_score = (1e18, 1e18, 1e18)  # (平均角, 距離合計, j-i)
        for j in range(i+2, last+1):
            angs, dsum = [], 0.0
            for c in range(4):
                p_prev, p_curr, p_j = series[c][i-1], series[c][i], series[c][j]
                angs.append(angle_turn_deg(p_prev, p_curr, p_j))
                dsum += np.linalg.norm(series[c][j,:2] - series[c][i,:2])
            cand = (float(np.mean(angs)), dsum, j - i)
            if cand < best_score:
                best_score, best_j = cand, j

        if best_j != i+1:
            for c in range(4):
                connect_to[c][i] = best_j
                disabled[c][i+1:best_j] = True  # 中間スライスの同隅は無効化

    # --- 最緩結合線（緑） ---
    bridge_pts_list = []
    for i in range(N-1):
        j = int(connect_to[1][i])  # LU系列で代表的に j を参照（4隅まとめて同じ j）
        if j <= i or j >= N:
            continue
        corridor = unary_union([rect_polys[k] for k in range(i, j+1)]).buffer(UNION_EPS)
        for c in range(4):
            p1, p2 = series[c][i], series[c][j]
            seg_pts = clip_and_sample_inside(p1, p2, corridor, LINE_STEP)
            if seg_pts.size > 0:
                bridge_pts_list.append(seg_pts)
    bridge_pts = np.vstack(bridge_pts_list) if bridge_pts_list else np.empty((0,3), float)

    # --- 航行空間フィルタ（緑線 ± NAV_WIDTH m ） ---
    map_pts = np.vstack(raw_seq) if KEEP_ALL_MAP_PTS and raw_seq else np.empty((0,3), float)
    if len(bridge_pts) > 1 and len(map_pts) > 0:
        nav_line = LineString(bridge_pts[:, :2])
        nav_area = nav_line.buffer(NAV_WIDTH, cap_style=2, join_style=2)
        navigable = unary_union(rect_polys).intersection(nav_area)
        mp = MultiPoint(map_pts[:, :2])
        inside_mask = np.array([navigable.contains(p) for p in mp.geoms])  # MultiPointは.geomsで反復
        filtered_pts = map_pts[inside_mask]
    else:
        filtered_pts = map_pts

    # --- 出力（統合：灰＝航行空間、 水色＝初期線、 緑＝最緩線） ---
    out_xyz = np.vstack([filtered_pts, bridge_initial, bridge_pts])
    colors = np.zeros((len(out_xyz), 3), np.uint16)
    idx0 = len(filtered_pts)
    idx1 = idx0 + len(bridge_initial)
    if idx0 > 0: colors[:idx0] = COLOR_MAP
    if len(bridge_initial) > 0: colors[idx0:idx1] = COLOR_CYAN
    if len(bridge_pts) > 0: colors[idx1:] = COLOR_GREEN

    header = copy_header_with_metadata(laspy.read(slice_files[0]).header)
    las_out = laspy.LasData(header)
    ensure_points_alloc(las_out, len(out_xyz))
    las_out.x, las_out.y, las_out.z = out_xyz[:,0], out_xyz[:,1], out_xyz[:,2]
    las_out.red, las_out.green, las_out.blue = colors[:,0], colors[:,1], colors[:,2]

    os.makedirs(os.path.dirname(OUTPUT_LAS_FINAL) or ".", exist_ok=True)
    las_out.write(OUTPUT_LAS_FINAL)

    print(f"✅ 統合LAS出力: {OUTPUT_LAS_FINAL}")
    print(f"  航行空間(灰): {len(filtered_pts):,}点")
    print(f"  初期線(水色): {len(bridge_initial):,}点")
    print(f"  最緩線(緑)  : {len(bridge_pts):,}点")
    print(f"  NAV_WIDTH=±{NAV_WIDTH}m, 閾値={ANGLE_THRESH_DEG}°, 先読み={LOOKAHEAD_SLICES}, STEP={LINE_STEP}m")

if __name__ == "__main__":
    main()
