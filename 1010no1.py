# -*- coding: utf-8 -*-
"""
【機能（GPU対応 M0 on M5 スタイル, 1スライス毎の出力＋中心線Excel）】
------------------------------------------------------------
- GPU優先のモルフォロジー閉処理（優先順: cv2.cuda → CuPy(cupyx) → CPU(cv2)）
- 入力LASを中心線に沿ってスライス化
- 各スライスで v–z occupancy 作成＋M0方式長方形検出
- 各スライスごとに航行可能空間LAS出力（緑点）
- 航行可能空間が存在するスライスのみ中心線Excel出力
------------------------------------------------------------
出力例:
  /workspace/output/0916no1_slices_M0onM5_maxrect_only_gpu/slice_0001_rect.las
  /workspace/output/0916no1_slices_M0onM5_maxrect_only_gpu/slices_with_navspace.xlsx
"""

import os
import math
import numpy as np
import cupy as cp
import cv2
import laspy
import pandas as pd

# cupyx の ndimage（GPUでのbinary_closing用）
try:
    from cupyx.scipy import ndimage as cndimage
    _HAS_CUPYX = True
except Exception:
    _HAS_CUPYX = False

# ===== 入出力 =====
INPUT_LAS  = "/workspace/data/0828_01_500_suidoubasi_ue.las"
OUTPUT_DIR = r"/workspace/output/0916no1_slices_M0onM5_maxrect_only_gpu"
WRITE_MERGED = False
MERGED_LAS   = r"/workspace/output/0916no1_M0onM5_maxrect_only_merged_gpu.las"
EXCEL_PATH   = os.path.join(OUTPUT_DIR, "slices_with_navspace.xlsx")

# ===== パラメータ =====
UKC = -1.0
BIN_X = 2.0
MIN_PTS_PER_XBIN = 50
GAP_DIST = 50.0
SECTION_INTERVAL = 0.5
LINE_LENGTH = 80.0
SLICE_THICKNESS = 0.20
MIN_PTS_PER_SLICE = 80
Z_MAX_FOR_NAV = 1.9
GRID_RES = 0.10
MORPH_RADIUS = 23               # 2*R+1 の楕円カーネル（cv2）/ 正方構造要素（cupyx）
USE_ANCHOR_DOWNFILL = True
ANCHOR_Z = 1.50
ANCHOR_TOL = 0.5
MIN_RECT_SIZE = 5               # [セル数] 高さ/幅の最小サイズ

# ==== GPU / CPU モルフォロジー閉処理 ====
def gpu_morph_close(binary_uint8: np.ndarray, radius: int) -> np.ndarray:
    """
    1) cv2.cuda が使えればそれで閉処理
    2) だめなら CuPy + cupyx.scipy.ndimage.binary_closing
    3) それも無理なら CPU(cv2.morphologyEx)
    すべて uint8(0/255) で入出力
    """
    ksize = max(1, 2*int(radius) + 1)

    # まず OpenCV CUDA を試す
    try:
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(binary_uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            closed_gpu = cv2.cuda.morphologyEx(gpu_mat, cv2.MORPH_CLOSE, kernel)
            return closed_gpu.download()
    except Exception as e:
        print(f"⚠ cv2.cuda 実行不可（{e}）→ 次の手段へ")

    # 次に CuPy + cupyx を試す
    if _HAS_CUPYX:
        try:
            bin_bool = cp.asarray(binary_uint8 > 0)
            # cupyx は楕円が手軽でないので、正方の構造要素で代替
            structure = cp.ones((ksize, ksize), dtype=cp.bool_)
            closed = cndimage.binary_closing(bin_bool, structure=structure)
            return cp.asnumpy(closed.astype(cp.uint8) * 255)
        except Exception as e:
            print(f"⚠ cupyx.scipy.ndimage 実行不可（{e}）→ CPUへフォールバック")

    # 最後に CPU（cv2）で実行
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(binary_uint8, cv2.MORPH_CLOSE, kernel)

# ==== 通常ユーティリティ ====
def copy_header_with_metadata(src_header):
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales = src_header.scales
    header.offsets = src_header.offsets
    if getattr(src_header, "srs", None):
        header.srs = src_header.srs
    if getattr(src_header, "vlrs", None):
        header.vlrs.extend(src_header.vlrs)
    if getattr(src_header, "evlrs", None):
        header.evlrs.extend(src_header.evlrs)
    return header

def l2(p, q): 
    return math.hypot(q[0]-p[0], q[1]-p[1])

def find_max_rectangle(bitmap_bool: np.ndarray):
    """最大長方形探索（ヒストグラム法, CPU）"""
    h, w = bitmap_bool.shape
    height = [0]*w
    best = (0,0,0,0); max_area = 0
    for i in range(h):
        for j in range(w):
            height[j] = height[j] + 1 if bitmap_bool[i, j] else 0
        stack = []; j = 0
        while j <= w:
            cur = height[j] if j < w else 0
            if not stack or cur >= height[stack[-1]]:
                stack.append(j); j += 1
            else:
                top_idx = stack.pop()
                width = j if not stack else j - stack[-1] - 1
                area = height[top_idx]*width
                if area > max_area:
                    max_area = area
                    top = i - height[top_idx] + 1
                    left = (stack[-1]+1) if stack else 0
                    best = (top, left, height[top_idx], width)
    return best

def downfill_on_closed_gpu(closed_uint8, z_min, grid_res, anchor_z, tol):
    """アンカーダウンフィル（列方向）をGPU（CuPy）で実施。CuPy不可ならCPUへフォールバック。"""
    try:
        closed_bool = cp.asarray(closed_uint8 > 0)
        gh, gw = closed_bool.shape
        i_anchor = int(round((anchor_z - z_min) / grid_res))
        pad = max(0, int(np.ceil(tol / grid_res)))
        i_lo = max(0, i_anchor - pad)
        i_hi = min(gh-1, i_anchor + pad)
        if i_lo > gh-1 or i_hi < 0:
            return cp.asnumpy(closed_bool.astype(cp.uint8)*255)
        out = closed_bool.copy()
        # 列ごと処理（forでもGPU上なので高速）
        for j in range(gw):
            col = closed_bool[:, j]
            if not cp.any(col):
                continue
            if cp.any(col[i_lo:i_hi+1]):
                imax = cp.max(cp.where(col)[0])
                out[:int(imax)+1, j] = True
        return cp.asnumpy(out.astype(cp.uint8)*255)
    except Exception as e:
        # CPUフォールバック
        closed_bool = (closed_uint8 > 0)
        gh, gw = closed_bool.shape
        i_anchor = int(round((anchor_z - z_min) / grid_res))
        pad = max(0, int(np.ceil(tol / grid_res)))
        i_lo = max(0, i_anchor - pad)
        i_hi = min(gh-1, i_anchor + pad)
        if i_lo > gh-1 or i_hi < 0:
            return (closed_bool.astype(np.uint8)*255)
        out = closed_bool.copy()
        for j in range(gw):
            col = closed_bool[:, j]
            if not np.any(col):
                continue
            if np.any(col[i_lo:i_hi+1]):
                imax = np.max(np.where(col)[0])
                out[:imax+1, j] = True
        return (out.astype(np.uint8)*255)

def rectangles_on_slice_M0_gpu(points_vz, grid_res, morph_radius, use_anchor, anchor_z, anchor_tol, min_rect_size):
    """最大長方形＋接する長方形の結合 → 外周セルを点群化"""
    rect_edge_pts_vz = []
    if len(points_vz) == 0:
        return rect_edge_pts_vz, None

    # occupancy grid 構築
    v_min, v_max = points_vz[:,0].min(), points_vz[:,0].max()
    z_min, z_max = points_vz[:,1].min(), points_vz[:,1].max()
    gw = max(1, int(np.ceil((v_max - v_min)/grid_res)))
    gh = max(1, int(np.ceil((z_max - z_min)/grid_res)))

    grid_raw = np.zeros((gh, gw), dtype=np.uint8)
    yi = ((points_vz[:,0] - v_min) / grid_res).astype(int)
    zi = ((points_vz[:,1] - z_min) / grid_res).astype(int)
    ok = (yi>=0)&(yi<gw)&(zi>=0)&(zi<gh)
    grid_raw[zi[ok], yi[ok]] = 255

    # モルフォロジー閉処理（GPU優先）
    closed0 = gpu_morph_close(grid_raw, morph_radius)
    # アンカーダウンフィル（GPU優先）
    closed = downfill_on_closed_gpu(closed0, z_min, grid_res, anchor_z, anchor_tol) if use_anchor else closed0

    closed_bool = (closed > 0)
    free_bitmap = ~closed_bool
    free_work = free_bitmap.copy()

    merged_mask = np.zeros_like(free_work, dtype=bool)
    merged_bounds = None
    first_bounds = None

    while np.any(free_work):
        top, left, h, w = find_max_rectangle(free_work)
        if h < min_rect_size or w < min_rect_size:
            break

        if merged_bounds is None:
            merged_bounds = [top, left, top+h, left+w]
            first_bounds  = merged_bounds.copy()
            merged_mask[top:top+h, left:left+w] = True
            free_work[top:top+h, left:left+w] = False
            continue

        # 「最初の最大長方形」との接触判定（±1セル許容）
        ft, fl, fb, fr = first_bounds
        if not (top+h >= ft-1 and top <= fb+1 and left+w >= fl-1 and left <= fr+1):
            free_work[top:top+h, left:left+w] = False
            continue

        mt, ml, mb, mr = merged_bounds
        merged_bounds = [min(mt, top), min(ml, left), max(mb, top+h), max(mr, left+w)]
        merged_mask[top:top+h, left:left+w] = True
        free_work[top:top+h, left:left+w] = False

    # 外周セルを点群化
    if merged_bounds is not None:
        mt, ml, mb, mr = merged_bounds
        for zi in range(mt, mb):
            for yi in range(ml, mr):
                if not merged_mask[zi, yi]:
                    continue
                # 外周 or 3x3近傍が全埋まりでない
                z0, z1 = max(mt, zi-1), min(mb, zi+2)
                y0, y1 = max(ml, yi-1), min(mr, yi+2)
                if zi in (mt, mb-1) or yi in (ml, mr-1) or not merged_mask[z0:z1, y0:y1].all():
                    v = v_min + (yi+0.5)*grid_res
                    z = z_min + (zi+0.5)*grid_res
                    rect_edge_pts_vz.append([v, z])

    return rect_edge_pts_vz, (v_min, z_min, gw, gh)

def vz_to_world_on_slice(vz, c, n_hat):
    v, z = vz
    p_xy = c + v*n_hat
    return [p_xy[0], p_xy[1], z]

def write_green_las(path, header_src, pts_xyz):
    header = copy_header_with_metadata(header_src)
    las_out = laspy.LasData(header)
    N = len(pts_xyz)
    try:
        las_out.points = laspy.ScaleAwarePointRecord.zeros(N, header=header)
    except AttributeError:
        las_out.points = laspy.PointRecord.zeros(N, header=header)
    if N > 0:
        pts_xyz = np.asarray(pts_xyz, float)
        las_out.x = pts_xyz[:,0]
        las_out.y = pts_xyz[:,1]
        las_out.z = pts_xyz[:,2]
    # 緑色
    if {"red","green","blue"} <= set(las_out.point_format.dimension_names):
        las_out.red   = np.zeros(N, dtype=np.uint16)
        las_out.green = np.full(N, 65535, dtype=np.uint16)
        las_out.blue  = np.zeros(N, dtype=np.uint16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    las_out.write(path)

# ========= メイン処理 =========
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    las = laspy.read(INPUT_LAS)
    X, Y, Z = np.asarray(las.x, float), np.asarray(las.y, float), np.asarray(las.z, float)
    xy = np.column_stack([X, Y])

    # --- 中心線抽出 (M5スタイル) ---
    x_min, x_max = xy[:,0].min(), xy[:,0].max()
    edges = np.arange(x_min, x_max+BIN_X, BIN_X)
    through = []
    for i in range(len(edges)-1):
        x0, x1 = edges[i], edges[i+1]
        m = (xy[:,0] >= x0) & (xy[:,0] < x1)
        if np.count_nonzero(m) < MIN_PTS_PER_XBIN:
            continue
        slab_xy, slab_z = xy[m], Z[m]
        order = np.argsort(slab_xy[:,1])
        slab_xy, slab_z = slab_xy[order], slab_z[order]
        under = slab_z <= UKC
        if not np.any(under):
            continue
        idx = np.where(under)[0]
        left, right = slab_xy[idx[0]], slab_xy[idx[-1]]
        through.append(0.5*(left + right))

    through = np.asarray(through, float)
    if len(through) < 2:
        raise RuntimeError("中心線が作れません")

    # 間引き
    thinned = [through[0]]
    for p in through[1:]:
        if l2(thinned[-1], p) >= GAP_DIST:
            thinned.append(p)
    through = np.asarray(thinned, float)

    # センター列生成
    centers = []
    for i in range(len(through)-1):
        p, q = through[i], through[i+1]
        d = l2(p, q)
        if d < 1e-9:
            continue
        n_steps = int(d/SECTION_INTERVAL)
        for s_i in range(n_steps+1):
            s = min(s_i*SECTION_INTERVAL, d)
            t = s/d
            centers.append((1-t)*p + t*q)
    centers = np.asarray(centers, float)

    # --- スライス処理 ---
    half_len = LINE_LENGTH*0.5
    half_th  = SLICE_THICKNESS*0.5

    merged_xyz = []      # 全スライス結合（任意）
    excel_rows = []      # 航行可能スライスのみ記録

    for i in range(len(centers)-1):
        c  = centers[i]
        cn = centers[i+1]

        t_vec = cn - c
        norm = np.linalg.norm(t_vec)
        if norm < 1e-9:
            # 空でも出力
            out_path = os.path.join(OUTPUT_DIR, f"slice_{i:04d}_rect.las")
            write_green_las(out_path, las.header, [])
            continue

        t_hat = t_vec / norm
        n_hat = np.array([-t_hat[1], t_hat[0]], float)

        dxy = xy - c
        u = dxy @ t_hat
        v = dxy @ n_hat

        m_band = (np.abs(u) <= half_th) & (np.abs(v) <= half_len)
        m_nav  = m_band & (Z <= Z_MAX_FOR_NAV)

        # スキップなし方針：点が少なくても処理続行
        points_vz = np.column_stack([v[m_nav], Z[m_nav]]) if np.any(m_nav) else np.empty((0,2), float)

        rect_edges_vz, _ = rectangles_on_slice_M0_gpu(
            points_vz, GRID_RES, MORPH_RADIUS,
            USE_ANCHOR_DOWNFILL, ANCHOR_Z, ANCHOR_TOL,
            MIN_RECT_SIZE
        )

        # VZ→XYZ
        slice_xyz = [vz_to_world_on_slice(vz, c, n_hat) for vz in rect_edges_vz]
        out_path = os.path.join(OUTPUT_DIR, f"slice_{i:04d}_rect.las")
        write_green_las(out_path, las.header, slice_xyz)

        if len(slice_xyz) > 0:
            excel_rows.append({"slice": i, "cx": float(c[0]), "cy": float(c[1]), "points": len(slice_xyz)})
            if WRITE_MERGED:
                merged_xyz.extend(slice_xyz)

    if WRITE_MERGED:
        write_green_las(MERGED_LAS, las.header, merged_xyz)

    if excel_rows:
        df = pd.DataFrame(excel_rows)
        # openpyxl が必要（未インストールなら pip install openpyxl）
        df.to_excel(EXCEL_PATH, index=False)
        print(f"✅ 航行可能スライス中心線Excel出力: {EXCEL_PATH}")

    print("✅ M0 on M5（GPU優先・自動フォールバック） 完了")

if __name__ == "__main__":
    main()
