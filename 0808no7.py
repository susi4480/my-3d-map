# -*- coding: utf-8 -*-
"""
【機能（航行可能空間は出さず“スライスのみの3D地図”を作る）】
1) 固定Y軸スライス（幅 BIN_Y）で各スライスの X中央値（Z≤Z_LIMIT）から中心線を推定
2) 中心線を弧長ステップ slice_interval（例: 0.5m）で等間隔サンプリング
3) 各サンプル点の接線に“垂直”な帯で切り出し（厚み slice_thickness = 0.2 → ±0.1m）
4) 元点群（Z≤）のうち帯に入る点だけを集めて LAS 出力（色は変更しない）
- CRS/VLR継承、RGB有無に自動対応
"""

import os
import numpy as np
import laspy

# === 入出力 ===
INPUT_LAS  = r"C:\Users\user\Documents\lab\outcome\0731_suidoubasi_ue.las"
OUTPUT_LAS = r"C:\Users\user\Documents\lab\output_las\0808_slices_only_map.las"

# === パラメータ ===
Z_LIMIT         = 0.8     # [m] これ以下のみ使用（橋など上部ノイズを除去）
BIN_Y           = 2.0     # [m] 固定Y軸スライス幅（中心線推定用）
MIN_PTS_PER_BIN = 50      # スライス内の最低点数
SMOOTH_WINDOW_M = 10.0    # [m] 中心線Xの移動平均（0で無効）

slice_thickness = 0.2     # [m] スライス帯の厚み（接線方向に ±0.1m）
slice_interval  = 0.5     # [m] 中心線に沿ったスライス間隔（等間隔サンプリング）
VOXEL_SIZE      = 0.0     # [m] 出力点数を抑えたい場合のみ >0（色は保持しない簡易版）

VERBOSE = True


# === ユーティリティ群 ===
def moving_average_1d(arr, win_m, bin_m):
    if win_m <= 0 or len(arr) < 2:
        return arr
    win = max(1, int(round(win_m / bin_m)))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    arr_pad = np.pad(arr, (pad, pad), mode='edge')
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(arr_pad, kernel, mode='valid')

def make_centerline_ybins(points_xyz, z_limit, bin_y, min_pts, smooth_window_m):
    """固定Y軸スライスで中心線（X中央値, スライス中央Y）を作る"""
    pts = points_xyz[points_xyz[:,2] <= z_limit]
    if len(pts) == 0:
        raise RuntimeError("Z制限後に点がありません。")

    y_min, y_max = pts[:,1].min(), pts[:,1].max()
    edges = np.arange(y_min, y_max + bin_y, bin_y)
    y_centers = 0.5 * (edges[:-1] + edges[1:])

    Xc, Yc = [], []
    for i in range(len(edges)-1):
        y0, y1 = edges[i], edges[i+1]
        m = (pts[:,1] >= y0) & (pts[:,1] < y1)
        if np.count_nonzero(m) < min_pts:
            continue
        slab = pts[m]
        Xc.append(np.median(slab[:,0]))
        Yc.append(y_centers[i])

    if len(Xc) < 2:
        raise RuntimeError("有効なYスライスが不足（中心線を作成できません）。")

    Xc = np.array(Xc, float)
    Yc = np.array(Yc, float)
    order = np.argsort(Yc)
    Xc, Yc = Xc[order], Yc[order]
    Xc = moving_average_1d(Xc, smooth_window_m, bin_y)
    return np.column_stack([Xc, Yc])

def resample_polyline_by_arclength(xy, step):
    """折れ線xyを弧長stepごとに等間隔サンプリング"""
    seg = np.diff(xy, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    L = np.concatenate([[0.0], np.cumsum(seglen)])
    total = L[-1]
    if total <= 0:
        return xy.copy()
    targets = np.arange(0.0, total + 1e-9, step)
    out = []
    j = 0
    for s in targets:
        # L[j] <= s <= L[j+1] となる j を探す
        while j+1 < len(L) and L[j+1] < s:
            j += 1
        if j+1 >= len(L):
            out.append(xy[-1])
            break
        t = (s - L[j]) / max(L[j+1] - L[j], 1e-12)
        p = xy[j] * (1 - t) + xy[j+1] * t
        out.append(p)
    return np.array(out)

def tangents_normals_from_polyline(xy):
    """前後差分で接線t・法線n（単位）"""
    n = xy.shape[0]
    t = np.zeros((n,2), float)
    if n >= 2:
        t[1:-1] = xy[2:] - xy[:-2]
        t[0]    = xy[1] - xy[0]
        t[-1]   = xy[-1] - xy[-2]
    norm = np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    t /= norm
    nvec = np.stack([-t[:,1], t[:,0]], axis=1)  # 左法線
    return t, nvec

def copy_header_with_metadata(src_header):
    """CRS/VLR含めメタデータ継承"""
    try:
        header = src_header.copy()
    except AttributeError:
        header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
        header.scales  = src_header.scales.copy()
        header.offsets = src_header.offsets.copy()
        header.vlrs  = list(getattr(src_header, "vlrs", []))
        header.evlrs = list(getattr(src_header, "evlrs", []))
    return header

def voxel_downsample(points_xyz, voxel_size):
    if voxel_size <= 0:
        return points_xyz
    keys = np.floor(points_xyz / voxel_size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points_xyz[np.sort(idx)]


# === メイン ===
def main():
    os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)

    # 入力
    las = laspy.read(INPUT_LAS)
    pts_all = np.vstack([las.x, las.y, las.z]).T

    # 入力RGBの有無
    in_dims = set(las.point_format.dimension_names)
    has_rgb_in = {"red","green","blue"} <= in_dims
    if has_rgb_in:
        rgb_all = np.vstack([las.red, las.green, las.blue]).T
    else:
        rgb_all = None

    # 1) 固定Y軸スライスで中心線
    centerline = make_centerline_ybins(
        pts_all, Z_LIMIT, BIN_Y, MIN_PTS_PER_BIN, SMOOTH_WINDOW_M
    )

    # 2) 中心線を弧長 slice_interval で等間隔サンプリング
    cl_samp = resample_polyline_by_arclength(centerline, slice_interval)

    # 3) サンプル中心線の接線・法線
    t, n = tangents_normals_from_polyline(cl_samp)

    # 4) Z≤ の元点群だけ対象
    zmask = pts_all[:,2] <= Z_LIMIT
    pts = pts_all[zmask]
    XY  = pts[:, :2]
    Z   = pts[:, 2]
    if has_rgb_in:
        rgb = rgb_all[zmask]

    # 5) 各スライス帯（接線に垂直、± slice_thickness/2）で抽出
    half_thick = slice_thickness * 0.5
    keep_xyz = []
    keep_rgb = []

    for i in range(len(cl_samp)):
        c  = cl_samp[i]
        ti = t[i]      # 接線 unit

        dxy = XY - c
        s = dxy @ ti   # 接線方向スカラー
        m = np.abs(s) <= half_thick
        if not np.any(m):
            continue

        keep_xyz.append(pts[m])
        if has_rgb_in:
            keep_rgb.append(rgb[m])

    if len(keep_xyz) == 0:
        raise RuntimeError("スライスに入る点がありません。slice_thickness / slice_interval を見直してください。")

    out_xyz = np.vstack(keep_xyz)
    if VOXEL_SIZE > 0:
        out_xyz = voxel_downsample(out_xyz, VOXEL_SIZE)
        # ※ 色は保持しない簡易版。必要なら最近傍で色再付与ロジックを追加可能。

    # 6) LAS出力（色は変更しない）
    header = copy_header_with_metadata(las.header)
    las_out = laspy.LasData(header)

    N = out_xyz.shape[0]
    try:
        las_out.points = laspy.ScaleAwarePointRecord.zeros(N, header=header)
    except AttributeError:
        las_out.points = laspy.PointRecord.zeros(N, header=header)

    las_out.x = out_xyz[:,0]
    las_out.y = out_xyz[:,1]
    las_out.z = out_xyz[:,2]

    out_dims = set(las_out.point_format.dimension_names)
    if VOXEL_SIZE == 0 and has_rgb_in and {"red","green","blue"} <= out_dims:
        out_rgb = np.vstack(keep_rgb)
        las_out.red   = out_rgb[:,0]
        las_out.green = out_rgb[:,1]
        las_out.blue  = out_rgb[:,2]
    # RGBなし/出力PF非対応/ダウンサンプル時は色書き込みをスキップ

    las_out.write(OUTPUT_LAS)

    if VERBOSE:
        print(f"✅ 出力: {OUTPUT_LAS}")
        print(f"  Z_LIMIT={Z_LIMIT}m, BIN_Y={BIN_Y}m, MIN_PTS_PER_BIN={MIN_PTS_PER_BIN}, SMOOTH={SMOOTH_WINDOW_M}m")
        print(f"  slice_thickness=±{half_thick:.2f}m（合計 {slice_thickness}m）, slice_interval={slice_interval}m")
        print(f"  抽出点数: {N}（入力Z≤: {np.count_nonzero(zmask)}）")
        print(f"  RGB入出力: in={has_rgb_in}, out_rgb_dims={'yes' if {'red','green','blue'} <= out_dims else 'no'}")

if __name__ == "__main__":
    main()
