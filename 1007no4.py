# -*- coding: utf-8 -*-
"""
【機能】スライス矩形LAS群を読み込み、横線で接続して1つのLASに統合
------------------------------------------------------------
- 入力: /workspace/output/xxx_filtered_slices/*.las
- 各スライスの矩形点群を読み込み、左下・左上・右下・右上の極値点を取得
- 隣接スライス間を直線補間した点群で接続（横線を点群化）
- すべてのスライス点群＋横線点群を統合して1つのLASを出力
------------------------------------------------------------
出力:
  /output/xxx_all_slices_with_bridges.las
（緑点）
"""

import os
import re
import numpy as np
import laspy
from glob import glob

# ===== 入出力 =====
INPUT_DIR  = "/output/1003no2_7_3_filtered_slices/"
OUTPUT_LAS = "/output/1003no2_7_3_all_slices_sita_with_bridges.las"

# ===== パラメータ =====
LINE_STEP = 0.05   # 横線の補間間隔[m]

# ==== ユーティリティ ====
def copy_header_with_metadata(src_header):
    """LASヘッダをメタデータ込みでコピー"""
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales  = src_header.scales
    header.offsets = src_header.offsets
    if getattr(src_header, "srs", None):
        header.srs = src_header.srs
    if getattr(src_header, "vlrs", None):
        header.vlrs.extend(src_header.vlrs)
    if getattr(src_header, "evlrs", None):
        header.evlrs.extend(src_header.evlrs)
    return header


def interpolate_line(p1, p2, step=0.1):
    """2点間を直線補間して点群を生成（ゼロ距離対策付き）"""
    p1, p2 = np.array(p1), np.array(p2)
    d = np.linalg.norm(p2 - p1)
    if d < 1e-6:
        return [p1]
    n = max(1, int(d / step))
    return [p1 + (p2 - p1) * (t / n) for t in range(n + 1)]


def get_extreme_points(pts):
    """
    矩形点群から 左下・左上・右下・右上 の4点を正確に抽出
    - 左右を「X最小・最大」で特定
    - それぞれで上下(Zmin/Zmax)を求める
    """
    if len(pts) == 0:
        return None

    xs = pts[:, 0]
    x_min, x_max = xs.min(), xs.max()
    tol = (x_max - x_min) * 0.02  # 全幅の2%を端の帯域とする

    left_mask  = xs <= (x_min + tol)
    right_mask = xs >= (x_max - tol)

    left_pts  = pts[left_mask]
    right_pts = pts[right_mask]

    if len(left_pts) == 0 or len(right_pts) == 0:
        return None

    left_low  = left_pts[np.argmin(left_pts[:, 2])]
    left_high = left_pts[np.argmax(left_pts[:, 2])]
    right_low  = right_pts[np.argmin(right_pts[:, 2])]
    right_high = right_pts[np.argmax(right_pts[:, 2])]

    return [left_low, left_high, right_low, right_high]


def extract_index(filename):
    """ファイル名からスライス番号を抽出（例: slice_0123_rect.las → 123）"""
    m = re.search(r"slice_(\d+)_rect\.las", os.path.basename(filename))
    return int(m.group(1)) if m else -1


def write_green_las(path, header_src, pts_xyz):
    """緑色点群としてLAS出力"""
    header = copy_header_with_metadata(header_src)
    las_out = laspy.LasData(header)
    N = len(pts_xyz)
    if N == 0:
        print("⚠ 出力点が0です。ファイルは作成されません。")
        return
    try:
        las_out.points = laspy.ScaleAwarePointRecord.zeros(N, header=header)
    except AttributeError:
        las_out.points = laspy.PointRecord.zeros(N, header=header)
    pts_xyz = np.asarray(pts_xyz, float)
    las_out.x, las_out.y, las_out.z = pts_xyz[:, 0], pts_xyz[:, 1], pts_xyz[:, 2]
    if {"red", "green", "blue"} <= set(las_out.point_format.dimension_names):
        las_out.red   = np.zeros(N, dtype=np.uint16)
        las_out.green = np.full(N, 65535, dtype=np.uint16)
        las_out.blue  = np.zeros(N, dtype=np.uint16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    las_out.write(path)
    print(f"✅ 出力: {path} 点数: {N:,}")


# ========= メイン処理 =========
def main():
    slice_files = sorted(
        glob(os.path.join(INPUT_DIR, "slice_*_rect.las")),
        key=extract_index
    )
    if not slice_files:
        raise RuntimeError("入力スライスLASが見つかりません")

    print(f"📥 スライス数: {len(slice_files)}")
    ALL_POINTS = []
    extreme_points_list = []

    # --- 各スライスを読み込み ---
    for f in slice_files:
        las = laspy.read(f)
        pts = np.column_stack([las.x, las.y, las.z])
        ALL_POINTS.extend(pts)
        extreme = get_extreme_points(pts)
        if extreme is not None:
            extreme_points_list.append(extreme)

    # --- 横線を生成（直線補間） ---
    print("🔗 隣接スライス間の接続線生成中 ...")
    BRIDGE_POINTS = []
    for i in range(len(extreme_points_list) - 1):
        p1 = extreme_points_list[i]
        p2 = extreme_points_list[i + 1]
        for j in range(4):  # 左下・左上・右下・右上
            line_points = interpolate_line(p1[j], p2[j], step=LINE_STEP)
            BRIDGE_POINTS.extend(line_points)

    ALL_POINTS.extend(BRIDGE_POINTS)

    # --- 統合LAS出力 ---
    las0 = laspy.read(slice_files[0])
    write_green_las(OUTPUT_LAS, las0.header, np.array(ALL_POINTS))
    print("🎉 完了: 全スライス＋橋線を統合しました。")


if __name__ == "__main__":
    main()
