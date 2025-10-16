# -*- coding: utf-8 -*-
"""
【GPU対応】改良IBGAL（局所版, スキップ有り）:
  灰: 全スライス点 / 青: 隣接(i,i+1)回廊内線 / 緑: 回廊内“最緩”i→i+PAIR_OFFSET（成功時のみ）
---------------------------------------------------------------------------------------
- CuPy 自動切替（GPU優先）
- ファイル名に含まれる数字でソート（桁数混在OK）
- 各スライスを外接矩形化（LL,LU,RU,RL）
- 青線：隣接(i,i+1)をその2枚の union(rect) を回廊として clip & サンプリング
- 緑線：i↔i+PAIR_OFFSET で seed 4本 → union→buffer(NAV_WIDTH) の IBGAL回廊を作成
    * 直結が回廊内なら採用
    * はみ出す場合：相手スライス側の“同一辺”上をスライドして回廊内かつ ΔZ/XY 最小を採用
    * それでもダメなら少しだけ両側スライドを試す
    * 1本でも失敗したら「この i の緑」は破棄（青は残す）、i を+1 して再トライ（スキップは張らない）
- 成功ペア(i, j)が確定したら中間 i+1..j-1 を disabled=True でスキップ（多重緑防止）
- 出力：/workspace/output/1016_corridor_relaxed_skip_gpu.las
依存:
    pip install laspy shapely numpy
    # GPUを使うなら（環境に合わせて）
    pip install cupy-cuda12x   または   cupy-cuda11x
"""

import os, re
import laspy
import numpy as _np
from glob import glob

# ==== 数値バックエンド（CuPy→NumPy 自動切替） ====
try:
    import cupy as xp
    GPU_ENABLED = True
    def to_np(a): return xp.asnumpy(a)
    def to_xp(a): return xp.asarray(a)
except Exception:
    import numpy as xp
    GPU_ENABLED = False
    def to_np(a): return a
    def to_xp(a): return a

from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

# ========= 入出力 =========
INPUT_DIR   = "/workspace/output/917slices_m0style_rect"
OUTPUT_LAS  = "/workspace/output/1016_corridor_relaxed_skip_gpu.las"

# ========= パラメータ =========
LINE_STEP        = 0.10   # サンプリング間隔 [m]
UNION_EPS        = 1e-6   # union時の微小buffer
PAIR_OFFSET      = 30     # i → i+30 を“最緩”候補に
NAV_WIDTH        = 2.5    # 回廊半幅 [m]
SLIDE_STEPS      = 41     # 端点スライド分解能（奇数推薦）
CENTER_SLIDE_MIN = 0.4    # 両側スライド時に中心寄りだけ試す（負荷軽減）
CENTER_SLIDE_MAX = 0.6
KEEP_BLUE_ALWAYS = True   # 緑が失敗でも青線は残す

# ========= 色 =========
COLOR_GRAY  = (52000, 52000, 52000)  # 灰：全スライス点
COLOR_BLUE  = (0, 52000, 65535)      # 青：隣接（初期）
COLOR_GREEN = (0, 65535, 0)          # 緑：最緩

# ========= LASユーティリティ =========
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

def load_points_from_las(path):
    las = laspy.read(path)
    return _np.column_stack([las.x, las.y, las.z])

# ========= 幾何ユーティリティ =========
def rect_from_points_cpu(pts_np):
    """スライス点群 → 外接矩形 corners([LL,LU,RU,RL]) と Polygon（CPU）"""
    xmin, xmax = float(_np.min(pts_np[:,0])), float(_np.max(pts_np[:,0]))
    ymin, ymax = float(_np.min(pts_np[:,1])), float(_np.max(pts_np[:,1]))
    zmean      = float(_np.mean(pts_np[:,2]))
    corners_np = _np.array([
        [xmin, ymin, zmean],  # LL(0)
        [xmin, ymax, zmean],  # LU(1)
        [xmax, ymax, zmean],  # RU(2)
        [xmax, ymin, zmean],  # RL(3)
    ], dtype=_np.float64)
    poly = Polygon(corners_np[:, :2])
    return corners_np, poly

def interpolate_line_xp(p1_np, p2_np, step):
    """GPU/CPU両対応の直線補間（返りはCPU np.ndarray）"""
    p1 = to_xp(p1_np); p2 = to_xp(p2_np)
    d  = xp.linalg.norm(p2 - p1)
    if float(d) < 1e-9:
        return _np.asarray([p1_np], dtype=_np.float64)
    n  = int(max(1, xp.ceil(d / step)))
    t  = xp.linspace(0.0, 1.0, n + 1, dtype=xp.float64)
    pts = p1[None,:] + (p2 - p1)[None,:] * t[:, None]
    return to_np(pts)

def clip_line_inside_corridor_cpu(p1_np, p2_np, corridor_poly, step):
    """Shapelyでclip→サンプリング（返りはCPU np.ndarray）"""
    line = LineString([tuple(p1_np[:2]), tuple(p2_np[:2])])
    inter = line.intersection(corridor_poly)
    if inter.is_empty:
        return _np.empty((0,3), dtype=_np.float64)
    segs = [inter] if isinstance(inter, LineString) else list(inter.geoms)
    out = []
    for seg in segs:
        coords = _np.asarray(seg.coords, dtype=_np.float64)
        for k in range(len(coords)-1):
            a2 = _np.array([coords[k][0],   coords[k][1],   p1_np[2]], dtype=_np.float64)
            b2 = _np.array([coords[k+1][0], coords[k+1][1], p2_np[2]], dtype=_np.float64)
            samp = interpolate_line_xp(a2, b2, step)
            out.append(samp)
    return _np.vstack(out) if out else _np.empty((0,3), dtype=_np.float64)

def segment_in_corridor_cpu(p1_np, p2_np, corridor_poly):
    """直線が回廊内に完全に含まれるか（境界含む）"""
    seg = LineString([tuple(p1_np[:2]), tuple(p2_np[:2])])
    return corridor_poly.covers(seg)

def slope_cost_xp(p1_np, p2_np):
    """ΔZ / XY距離（小さいほど水平: “緩い”）"""
    p1 = to_xp(p1_np); p2 = to_xp(p2_np)
    dxy = xp.hypot(p2[0]-p1[0], p2[1]-p1[1])
    if float(dxy) < 1e-9:
        return float('inf')
    dz  = xp.abs(p2[2]-p1[2])
    return float(dz / dxy)

def side_endpoints(rect_np, corner_idx):
    """corner_idx と同一側の辺の端点（CPU np）"""
    if corner_idx in (0,1):   # 左辺：LL(0)↔LU(1)
        a, b = rect_np[0], rect_np[1]
    else:                     # 右辺：RL(3)↔RU(2)（下→上）
        a, b = rect_np[3], rect_np[2]
    return a.astype(_np.float64), b.astype(_np.float64)

def slide_on_side_np(a_np, b_np, t):
    return (1.0 - t) * a_np + t * b_np

# ========= メイン =========
def main():
    # 1) ファイル列（任意桁の数字でソート）
    slice_files = sorted(
        glob(os.path.join(INPUT_DIR, "slice_*_rect.las")),
        key=lambda f: int(re.search(r"slice_(\d+)_rect\.las", os.path.basename(f)).group(1))
    )
    if not slice_files:
        slice_files = sorted(
            glob(os.path.join(INPUT_DIR, "*.las")),
            key=lambda f: int(re.search(r"(\d+)", os.path.basename(f)).group(1))
        )
    if not slice_files:
        raise RuntimeError("矩形LASが見つかりません。")

    # 2) 読み込み（CPU）→ 必要に応じてGPU側で演算
    slice_pts_np = [load_points_from_las(f) for f in slice_files]
    all_gray_np  = _np.vstack(slice_pts_np)
    print(f"GPU: {'ON' if GPU_ENABLED else 'OFF'} | Slices: {len(slice_files)} | Points: {len(all_gray_np):,}")

    # 3) 各スライスを外接矩形化
    rects_np, polys = [], []
    for P in slice_pts_np:
        corners_np, poly = rect_from_points_cpu(P)
        rects_np.append(corners_np)
        polys.append(poly)
    N = len(rects_np)

    # 4) 青線（隣接 i→i+1）：局所回廊でclip
    blue_segs = []
    for i in range(N-1):
        corridor_local = unary_union([polys[i], polys[i+1]]).buffer(UNION_EPS)
        c1, c2 = rects_np[i], rects_np[i+1]
        for j in range(4):
            seg_pts = clip_line_inside_corridor_cpu(c1[j], c2[j], corridor_local, LINE_STEP)
            if seg_pts.size:
                blue_segs.append(seg_pts)
    blue_pts_np = _np.vstack(blue_segs) if blue_segs else _np.empty((0,3), _np.float64)
    print(f"🔵 Blue points: {len(blue_pts_np):,}")

    # 5) 緑線（i→i+PAIR_OFFSET）：IBGAL式回廊 + 回廊内“最緩”＋スキップ
    green_segs = []
    disabled = _np.zeros(N, dtype=bool)
    t_vals = _np.linspace(0.0, 1.0, SLIDE_STEPS, dtype=_np.float64)
    center_t = _np.linspace(CENTER_SLIDE_MIN, CENTER_SLIDE_MAX, 5, dtype=_np.float64)

    i = 0
    while i <= N - 1 - PAIR_OFFSET:
        if disabled[i]:
            i += 1
            continue

        j = i + PAIR_OFFSET
        c1, c2 = rects_np[i], rects_np[j]

        # IBGAL回廊：seed 4本 → union → buffer
        seed_lines = [LineString([tuple(c1[k][:2]), tuple(c2[k][:2])]) for k in range(4)]
        corridor = unary_union(seed_lines).buffer(NAV_WIDTH, cap_style=2, join_style=2)

        pair_ok = True
        tmp_segments = []   # このペアで4本全てが確定したら反映

        for k in range(4):
            p1 = c1[k].astype(_np.float64)
            p2 = c2[k].astype(_np.float64)

            # 直結が回廊内なら採用
            if segment_in_corridor_cpu(p1, p2, corridor):
                tmp_segments.append(interpolate_line_xp(p1, p2, LINE_STEP))
                continue

            # 相手側辺上をスライド
            a2, b2 = side_endpoints(c2, k)
            best_q = None
            best_cost = float('inf')

            for t in t_vals:
                q2 = slide_on_side_np(a2, b2, float(t))
                if segment_in_corridor_cpu(p1, q2, corridor):
                    cost = slope_cost_xp(p1, q2)
                    if cost < best_cost:
                        best_cost, best_q = cost, q2

            if best_q is not None:
                tmp_segments.append(interpolate_line_xp(p1, best_q, LINE_STEP))
                continue

            # 両側を少しスライド（中心寄りのみ、負荷軽減）
            a1, b1 = side_endpoints(c1, k)
            rescued = False
            for t1 in center_t:
                q1 = slide_on_side_np(a1, b1, float(t1))
                for t2 in t_vals:
                    q2 = slide_on_side_np(a2, b2, float(t2))
                    if segment_in_corridor_cpu(q1, q2, corridor):
                        cost = slope_cost_xp(q1, q2)
                        if cost < best_cost:
                            best_cost, best_q = cost, (q1, q2)
                            rescued = True
                if rescued:
                    break

            if rescued and isinstance(best_q, tuple):
                tmp_segments.append(interpolate_line_xp(best_q[0], best_q[1], LINE_STEP))
            else:
                # この角は救えなかった → ペア失敗（緑は出さない/青はそのまま）
                pair_ok = False
                break

        if pair_ok:
            # 4本そろって成功 → 追加してスキップ張る
            green_segs.extend(tmp_segments)
            if j - (i + 1) > 0:
                disabled[i+1:j] = True
            i = j  # 成功した相手へジャンプ
        else:
            # 失敗 → iを+1（青は既に保持）
            i += 1

    green_pts_np = _np.vstack(green_segs) if green_segs else _np.empty((0,3), _np.float64)
    print(f"✅ Green points: {len(green_pts_np):,}")

    # 6) 出力（LAS: 灰 + 青 + 緑）
    out_xyz_np = _np.vstack([all_gray_np, blue_pts_np, green_pts_np])
    colors = _np.zeros((len(out_xyz_np), 3), _np.uint16)
    n_gray = len(all_gray_np)
    n_blue = len(blue_pts_np)
    colors[:n_gray] = COLOR_GRAY
    if n_blue:
        colors[n_gray:n_gray+n_blue] = COLOR_BLUE
    if len(green_pts_np):
        colors[n_gray+n_blue:] = COLOR_GREEN

    header = copy_header_with_metadata(laspy.read(slice_files[0]).header)
    las_out = laspy.LasData(header)
    ensure_points_alloc(las_out, len(out_xyz_np))
    las_out.x, las_out.y, las_out.z = out_xyz_np[:,0], out_xyz_np[:,1], out_xyz_np[:,2]
    if {"red","green","blue"} <= set(las_out.point_format.dimension_names):
        las_out.red, las_out.green, las_out.blue = colors[:,0], colors[:,1], colors[:,2]

    os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)
    las_out.write(OUTPUT_LAS)

    print("🏁 完了")
    print(f"  灰(all slices): {n_gray:,}")
    print(f"  青(adjacent):   {n_blue:,}")
    print(f"  緑(relaxed):     {len(green_pts_np):,}")
    print(f"  出力: {OUTPUT_LAS}")

if __name__ == "__main__":
    main()
