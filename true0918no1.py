# -*- coding: utf-8 -*-
"""
【機能】スライス矩形LAS群を読み込み、横線で接続して統合出力
- 入力: /workspace/output/917slices_m0style_rect/slice_????_rect.las
- 各スライスの外周点群から PCA により v 方向（幅方向）を推定し、
  左下・左上・右下・右上（合計4点）を抽出
- 隣接スライス間で対応4点を直線で接続
  (A) 線を点群として 0.10 m 間隔でサンプリング → LAS 1本に統合
  (B) 線をエッジとして PLY に出力（ポリライン）
"""

import os
import numpy as np
import laspy
from glob import glob

# ===== 入出力 =====
INPUT_DIR   = "/workspace/output/917slices_m0style_rect"
OUTPUT_LAS  = "/workspace/output/all_slices_with_bridges.las"     # 点群(スライス点 + 線サンプリング点)
OUTPUT_PLY  = "/workspace/output/bridges_lines.ply"               # ポリライン(エッジ)

# ===== パラメータ =====
LINE_STEP = 0.10   # 横線を点群化する補間間隔 [m]（LAS 用）

# ==== ユーティリティ ====
def copy_header_with_metadata(src_header):
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales  = src_header.scales
    header.offsets = src_header.offsets
    if getattr(src_header, "srs", None): header.srs = src_header.srs
    if getattr(src_header, "vlrs", None): header.vlrs.extend(src_header.vlrs)
    if getattr(src_header, "evlrs", None): header.evlrs.extend(src_header.evlrs)
    return header

def write_green_las(path, header_src, pts_xyz):
    """RGB(0,65535,0) の緑点で LAS 書き出し"""
    if len(pts_xyz) == 0: 
        raise RuntimeError("出力点がありません")
    header = copy_header_with_metadata(header_src)
    las_out = laspy.LasData(header)
    N = len(pts_xyz)
    try:
        las_out.points = laspy.ScaleAwarePointRecord.zeros(N, header=header)
    except AttributeError:
        las_out.points = laspy.PointRecord.zeros(N, header=header)
    pts_xyz = np.asarray(pts_xyz, float)
    las_out.x, las_out.y, las_out.z = pts_xyz[:,0], pts_xyz[:,1], pts_xyz[:,2]
    if {"red","green","blue"} <= set(las_out.point_format.dimension_names):
        las_out.red   = np.zeros(N, dtype=np.uint16)
        las_out.green = np.full(N, 65535, dtype=np.uint16)
        las_out.blue  = np.zeros(N, dtype=np.uint16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    las_out.write(path)
    print(f"✅ LAS出力: {path} 点数: {N}")

def write_ply_lines(path, vertices, edges):
    """ポリライン（エッジ）を持つ PLY（ASCII）を書き出し"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for e in edges:
            f.write(f"{e[0]} {e[1]}\n")
    print(f"✅ PLY出力(ポリライン): {path} 辺数: {len(edges)}")

def interpolate_line(p1, p2, step=0.1):
    """2点間を直線補間して点群生成（ゼロ除算対策付き）"""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = np.linalg.norm(p2 - p1)
    if d < 1e-9:
        return [p1]  # 同一点
    n = max(1, int(d / step))  # 少なくとも1分割
    return [p1 + (p2 - p1) * (t / n) for t in range(n + 1)]

def get_extreme_points_pca(pts_xyz):
    """
    PCA で幅方向を推定し、左/右の集合から Zmin/Zmax を抽出して 4 点を返す。
    返り値: [left_low, left_high, right_low, right_high]（各 shape=(3,)）
    """
    if len(pts_xyz) < 4:
        return None
    xy = pts_xyz[:, :2]
    mu = xy.mean(axis=0)
    A = xy - mu
    C = A.T @ A / max(1, len(A)-1)
    w, V = np.linalg.eigh(C)      # 2x2 固有分解
    # スライス厚みが薄い想定：幅方向の分散が大きい方を選ぶ
    axis = V[:, np.argmax(w)]     # 幅方向（v軸に相当）
    vcoord = A @ axis             # 各点の v 座標（スライス座標系っぽく）
    vmin, vmax = vcoord.min(), vcoord.max()
    if vmax - vmin < 1e-6:
        return None
    # 左右を「端側の帯域」で抽出（端に1セル相当= 2% 幅のマージン）
    band = max(0.02*(vmax-vmin), 0.05)  # 5cm 以上 or 全幅の2%
    left_pts  = pts_xyz[vcoord <= vmin + band]
    right_pts = pts_xyz[vcoord >= vmax - band]
    if len(left_pts)==0 or len(right_pts)==0:
        # 端の帯域が取れなければ厳密に min/max で代用
        left_pts  = pts_xyz[vcoord == vmin]
        right_pts = pts_xyz[vcoord == vmax]
        if len(left_pts)==0 or len(right_pts)==0:
            return None
    left_low   = left_pts[np.argmin(left_pts[:,2])]
    left_high  = left_pts[np.argmax(left_pts[:,2])]
    right_low  = right_pts[np.argmin(right_pts[:,2])]
    right_high = right_pts[np.argmax(right_pts[:,2])]
    return [left_low, left_high, right_low, right_high]

# ========= メイン処理 =========
def main():
    slice_files = sorted(glob(os.path.join(INPUT_DIR, "slice_*_rect.las")))
    if not slice_files:
        raise RuntimeError(f"入力がありません: {INPUT_DIR}/slice_*_rect.las")

    # 1) 全スライス点群の集約（LAS用）
    ALL_POINTS = []
    # 2) 各スライスの代表4点（左下・左上・右下・右上）
    extremes_pts_per_slice = []   # list of [4 x (3,)]
    # 3) PLY 用の頂点（各スライスの4点をそのまま頂点化）
    ply_vertices = []
    ply_extreme_indices = []      # 各スライスの 4 頂点のインデックス（PLY）

    las0 = laspy.read(slice_files[0])  # ヘッダ用

    for f in slice_files:
        las = laspy.read(f)
        pts = np.column_stack([las.x, las.y, las.z])
        if len(pts) == 0:
            extremes_pts_per_slice.append(None)
            continue
        ALL_POINTS.extend(pts)  # 元のスライスの緑点を統合

        # PCA から 4 極値点を取得
        extremes = get_extreme_points_pca(pts)
        extremes_pts_per_slice.append(extremes)

        if extremes is not None:
            base = len(ply_vertices)
            ply_vertices.extend(extremes)  # 4点追加
            ply_extreme_indices.append([base+0, base+1, base+2, base+3])
        else:
            ply_extreme_indices.append(None)

    # 隣接スライス間の 4 本の横線を作成
    # (A) LAS 用：点群としてサンプリング
    BRIDGE_POINTS = []
    # (B) PLY 用：エッジとして接続（頂点 index 使用）
    ply_edges = []

    for i in range(len(extremes_pts_per_slice)-1):
        e1 = extremes_pts_per_slice[i]
        e2 = extremes_pts_per_slice[i+1]
        id1 = ply_extreme_indices[i]
        id2 = ply_extreme_indices[i+1]
        if e1 is None or e2 is None or id1 is None or id2 is None:
            continue
        # 4 本（左下・左上・右下・右上）のペアを接続
        for j in range(4):
            # (A) LAS 点群として 0.10m 間隔で補間
            line_pts = interpolate_line(e1[j], e2[j], step=LINE_STEP)
            BRIDGE_POINTS.extend(line_pts)
            # (B) PLY エッジ
            ply_edges.append((id1[j], id2[j]))

    # (A) LAS: 元スライス点 + 横線サンプリング点 を統合出力
    ALL_POINTS.extend(BRIDGE_POINTS)
    write_green_las(OUTPUT_LAS, las0.header, np.asarray(ALL_POINTS))

    # (B) PLY: 4 本の横線をポリライン（エッジ）として出力
    if len(ply_vertices) > 0 and len(ply_edges) > 0:
        write_ply_lines(OUTPUT_PLY, np.asarray(ply_vertices), ply_edges)
    else:
        print("⚠️ PLY のエッジ出力はスキップ（代表点が十分に抽出できませんでした）")

if __name__ == "__main__":
    main()
