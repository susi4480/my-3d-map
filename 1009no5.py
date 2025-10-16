# -*- coding: utf-8 -*-
"""
【機能】単純隅定義＋初期接続＋最緩探索＋緑線連結（可視化つき）
---------------------------------------------------------------
- 各スライスで 左下・左上・右下・右上 を Z極値で抽出
- 初期接続は i→i+1（単純版と同じ）
- 角度が急なら i→i+2..i+30 の中から最緩ペアを探索（平均角度最小・距離最短）
- 採用された i→best_j の隅同士を 緑点＋緑線で可視化
- 線は Union(rectangle[i..j]) 内にクリップ
"""

import os, re
import numpy as np
import laspy
from glob import glob
from shapely.geometry import Polygon, LineString, MultiLineString
from shapely.ops import unary_union

# ===== 入出力 =====
INPUT_DIR  = "/workspace/output/917slices_m0style_rect/"
OUTPUT_LAS = "/workspace/output/1010_hybrid_simplecorners_greenlines.las"

# ===== パラメータ =====
ANGLE_THRESH_DEG   = 35.0
LOOKAHEAD_SLICES   = 30
LINE_STEP          = 0.03
UNION_EPS          = 1e-6
KEEP_ALL_MAP_PTS   = True

COLOR_MAP   = (52000, 52000, 52000)
COLOR_GREEN = (0, 65535, 0)

# ===== ユーティリティ =====
def copy_header_with_metadata(src_header):
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales, header.offsets = src_header.scales, src_header.offsets
    if getattr(src_header, "srs", None): header.srs = src_header.srs
    if getattr(src_header, "vlrs", None): header.vlrs.extend(src_header.vlrs)
    if getattr(src_header, "evlrs", None): header.evlrs.extend(src_header.evlrs)
    return header

def angle_turn_deg(p_prev, p_curr, p_next):
    """角度変化の絶対値（直進=0°）"""
    a = np.asarray(p_prev[:2]) - np.asarray(p_curr[:2])
    b = np.asarray(p_next[:2]) - np.asarray(p_curr[:2])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cosv = np.clip(np.dot(a,b)/(na*nb), -1.0, 1.0)
    inner = np.degrees(np.arccos(cosv))
    return abs(inner - 180.0)

def interpolate_line_dense(p1, p2, step):
    """2点間を一定間隔でサンプル"""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = np.linalg.norm(p2 - p1)
    if d < 1e-9:
        return p1[None,:]
    n = max(1, int(np.ceil(d/step)))
    t = np.linspace(0.0, 1.0, n+1)
    return p1[None,:] + (p2 - p1)[None,:]*t[:,None]

def get_extreme_points(pts):
    """矩形点群から 左下・左上・右下・右上 を返す"""
    if len(pts) == 0:
        return None
    xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]
    left_mask  = xs <= xs.mean()
    right_mask = ~left_mask
    left_pts, right_pts = pts[left_mask], pts[right_mask]
    if len(left_pts)==0 or len(right_pts)==0:
        return None
    left_low  = left_pts[np.argmin(left_pts[:,2])]
    left_high = left_pts[np.argmax(left_pts[:,2])]
    right_low  = right_pts[np.argmin(right_pts[:,2])]
    right_high = right_pts[np.argmax(right_pts[:,2])]
    return [left_low, left_high, right_low, right_high]

def rect_polygon_from_corners(c4):
    """四隅→矩形ポリゴン"""
    LL, LU, RL, RU = c4
    ring = [tuple(LL[:2]), tuple(RL[:2]), tuple(RU[:2]), tuple(LU[:2])]
    return Polygon(ring)

def clip_and_sample_inside(p1, p2, poly_union, step):
    """線分をUnion矩形内にクリップしてサンプリング"""
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

# ===== メイン処理 =====
def main():
    # --- 入力読み込み ---
    slice_files = sorted(
        glob(os.path.join(INPUT_DIR, "slice_*_rect.las")),
        key=lambda f: int(re.search(r"slice_(\d+)_rect\.las", os.path.basename(f)).group(1))
    )
    if not slice_files:
        raise RuntimeError("入力スライスが見つかりません")

    raw_seq = []
    corners_seq = []

    for f in slice_files:
        las = laspy.read(f)
        P = np.column_stack([las.x, las.y, las.z])
        raw_seq.append(P)
        c4 = get_extreme_points(P)
        if c4 is not None:
            corners_seq.append(np.array(c4))

    N = len(corners_seq)
    if N < 3:
        raise RuntimeError("スライスが少なすぎます")

    # --- 初期接続：単純にi→i+1 ---
    connect_to = {c: np.array([i+1 for i in range(N)], dtype=int) for c in range(4)}
    series     = {c: np.array([corners_seq[i][c] for i in range(N)]) for c in range(4)}

    # --- 最緩探索：採用ペアを記録 ---
    bridge_segments = []  # [(p1, p2), ...]

    for i in range(1, N-1):
        needs_reconnect = False
        for c in range(4):
            ang = angle_turn_deg(series[c][i-1], series[c][i], series[c][i+1])
            if ang >= ANGLE_THRESH_DEG:
                needs_reconnect = True
        if not needs_reconnect:
            continue

        last = min(N-1, i + LOOKAHEAD_SLICES)
        best_j = i+1
        best_score = (1e18, 1e18, 1e18)
        for j in range(i+2, last+1):
            angs, dsum = [], 0.0
            for c in range(4):
                p_prev, p_curr, p_j = series[c][i-1], series[c][i], series[c][j]
                angs.append(angle_turn_deg(p_prev, p_curr, p_j))
                dsum += np.linalg.norm(series[c][j,:2] - series[c][i,:2])
            mean_ang = float(np.mean(angs))
            cand = (mean_ang, dsum, j - i)
            if cand < best_score:
                best_score, best_j = cand, j

        # 採用された最緩ペアを登録（隅ごと）
        if best_j != i+1:
            for c in range(4):
                p1 = series[c][i]
                p2 = series[c][best_j]
                bridge_segments.append((p1, p2))

    # --- 回廊構築 ---
    rect_polys = [rect_polygon_from_corners(corners_seq[k]) for k in range(N)]

    # --- 線生成 ---
    bridge_pts_list = []
    for (p1, p2) in bridge_segments:
        # 該当区間の回廊をUnion
        idx1 = next((k for k in range(N) if np.allclose(series[0][k][:2], p1[:2], atol=1e-4)), 0)
        idx2 = next((k for k in range(N) if np.allclose(series[0][k][:2], p2[:2], atol=1e-4)), N-1)
        i, j = sorted([idx1, idx2])
        corridor = unary_union([rect_polys[k] for k in range(i, j+1)]).buffer(UNION_EPS)
        seg_pts = clip_and_sample_inside(p1, p2, corridor, LINE_STEP)
        if seg_pts.size > 0:
            bridge_pts_list.append(seg_pts)

    bridge_pts = np.vstack(bridge_pts_list) if bridge_pts_list else np.empty((0,3), float)

    # --- 出力LAS ---
    map_pts = np.vstack(raw_seq) if KEEP_ALL_MAP_PTS else np.empty((0,3), float)
    out_xyz = np.vstack([map_pts, bridge_pts]) if bridge_pts.size else map_pts
    colors  = np.zeros((len(out_xyz), 3), np.uint16)
    if map_pts.size:
        colors[:len(map_pts)] = COLOR_MAP
    if bridge_pts.size:
        colors[len(map_pts):] = COLOR_GREEN

    las0 = laspy.read(slice_files[0])
    header = copy_header_with_metadata(las0.header)
    las_out = laspy.LasData(header)
    las_out.x, las_out.y, las_out.z = out_xyz[:,0], out_xyz[:,1], out_xyz[:,2]
    if {"red","green","blue"} <= set(las_out.point_format.dimension_names):
        las_out.red, las_out.green, las_out.blue = colors[:,0], colors[:,1], colors[:,2]

    os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)
    las_out.write(OUTPUT_LAS)

    print(f"✅ 出力: {OUTPUT_LAS}")
    print(f"  地図点: {len(map_pts):,} / 緑線点: {len(bridge_pts):,}")
    print(f"  最緩ペア数: {len(bridge_segments)}")
    print(f"  閾値: {ANGLE_THRESH_DEG}°, 先読み: {LOOKAHEAD_SLICES} スライス")
    print("  仕様: 単純隅＋初期接続＋最緩ペアを緑線で可視化")

if __name__ == "__main__":
    main()
