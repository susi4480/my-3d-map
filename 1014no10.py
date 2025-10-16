# -*- coding: utf-8 -*-
"""
【機能】GPU対応・初期接続線のみ出力（緑線＝矩形間の単純接続）
-----------------------------------------------------------------
- 各スライスの矩形（slice_XXXX_rect.las）を読み込み
- PCAで四隅推定（失敗時はAABB）
- 隣接スライス間を単純接続して「初期接続線」を生成（GPU処理）
- 出力: 灰=スライス外周, 緑=初期接続線
-----------------------------------------------------------------
依存:
    pip install laspy cupy-cuda12x opencv-python opencv-contrib-python
-----------------------------------------------------------------
出力: /workspace/output/1014_navspace_initial_connect_gpu.las
"""

import os, re
import laspy
import numpy as np
import cupy as cp
from glob import glob
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

# ===== 入出力 =====
INPUT_DIR = "/workspace/output/917slices_m0style_rect/"
OUTPUT_LAS_FINAL = "/workspace/output/1014_navspace_initial_connect_gpu.las"

# ===== パラメータ =====
LINE_STEP = 0.01
UNION_EPS = 1e-6
COLOR_INNER = (52000, 52000, 52000)  # 灰
COLOR_GREEN = (0, 65535, 0)          # 緑

# ===== 関数群 =====
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

def pca_rect_corners_safe(pts):
    """PCAベース＋外接矩形（失敗時はAABB）"""
    if pts.shape[0] < 4:
        return None
    try:
        xy = pts[:, :2]
        c = xy.mean(axis=0)
        X = xy - c
        C = np.cov(X.T)
        _, _, VT = np.linalg.svd(C)
        R = VT.T
        uv = X @ R
        umin, vmin = uv.min(axis=0)
        umax, vmax = uv.max(axis=0)
        corners_uv = np.array([[umin,vmin],[umin,vmax],[umax,vmin],[umax,vmax]])
        corners_xy = corners_uv @ R.T + c
        z_med = np.median(pts[:,2])
        return np.column_stack([corners_xy, np.full(4,z_med)])
    except:
        xy = pts[:, :2]
        xmin, ymin = np.min(xy, axis=0)
        xmax, ymax = np.max(xy, axis=0)
        z_med = np.median(pts[:,2])
        return np.array([[xmin,ymin,z_med],[xmin,ymax,z_med],[xmax,ymin,z_med],[xmax,ymax,z_med]])

def rect_polygon_from_corners(c4):
    LL, LU, RL, RU = c4
    ring = [tuple(LL[:2]), tuple(RL[:2]), tuple(RU[:2]), tuple(LU[:2])]
    return Polygon(ring)

def clip_and_sample_inside_gpu(p1, p2, poly_union, step):
    """GPU対応・線分内サンプリング（Shapely部分はCPU）"""
    line = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
    inter = line.intersection(poly_union)
    if inter.is_empty:
        return np.empty((0,3), float)
    segs = [inter] if isinstance(inter, LineString) else list(inter.geoms)
    out_list = []

    # GPUに転送して高速補間
    v2 = cp.asarray(p2[:2]) - cp.asarray(p1[:2])
    vv = max(float(cp.dot(v2, v2)), 1e-12)

    for seg in segs:
        coords = np.asarray(seg.coords, float)
        if len(coords) < 2:
            continue
        coords_cp = cp.asarray(coords)
        diff = coords_cp[1:] - coords_cp[:-1]
        dists = cp.sqrt(cp.sum(diff**2, axis=1))
        for k in range(len(dists)):
            d2 = float(dists[k])
            if d2 < 1e-9:
                continue
            n = max(1, int(np.ceil(d2 / step)))
            t = cp.linspace(0, 1, n + 1)
            a2 = coords_cp[k]
            b2 = coords_cp[k + 1]
            xy = a2[None, :] + (b2 - a2)[None, :] * t[:, None]
            proj = cp.dot(xy - cp.asarray(p1[:2])[None, :], v2) / vv
            proj = cp.clip(proj, 0.0, 1.0)
            z = p1[2] + (p2[2] - p1[2]) * proj
            out_list.append(cp.column_stack([xy, z]).get())  # CPU側に戻す

    return np.vstack(out_list) if out_list else np.empty((0,3), float)

# ===== メイン =====
def main():
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

    # 矩形ポリゴン
    rect_polys = [rect_polygon_from_corners(corners_seq[k]) for k in range(N)]
    series = {c: np.array([corners_seq[i][c] for i in range(N)]) for c in range(4)}

    # ===== 初期接続線（GPU処理） =====
    bridge_initial = []
    for i in range(N - 1):
        corridor = unary_union([rect_polys[i], rect_polys[i + 1]]).buffer(UNION_EPS)
        for c in range(4):
            seg = clip_and_sample_inside_gpu(series[c][i], series[c][i + 1], corridor, LINE_STEP)
            if seg.size > 0:
                bridge_initial.append(seg)
    bridge_initial = np.vstack(bridge_initial) if bridge_initial else np.empty((0, 3), float)

    # ===== 出力 =====
    map_pts = np.vstack(raw_seq)
    out_xyz = np.vstack([map_pts, bridge_initial])
    color_all = np.vstack([
        np.tile(COLOR_INNER, (len(map_pts), 1)),
        np.tile(COLOR_GREEN, (len(bridge_initial), 1))
    ])

    header = copy_header_with_metadata(laspy.read(slice_files[0]).header)
    las_out = laspy.LasData(header)
    ensure_points_alloc(las_out, len(out_xyz))
    las_out.x, las_out.y, las_out.z = out_xyz[:, 0], out_xyz[:, 1], out_xyz[:, 2]
    las_out.red, las_out.green, las_out.blue = color_all[:, 0], color_all[:, 1], color_all[:, 2]

    os.makedirs(os.path.dirname(OUTPUT_LAS_FINAL), exist_ok=True)
    las_out.write(OUTPUT_LAS_FINAL)

    print(f"✅ 出力完了: {OUTPUT_LAS_FINAL}")
    print(f"  初期接続線(緑): {len(bridge_initial):,} 点")
    print(f"  スライス外周(灰): {len(map_pts):,} 点")
    print("  GPU処理完了 ✅")

if __name__ == "__main__":
    main()
