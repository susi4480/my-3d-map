# -*- coding: utf-8 -*-
"""
【オーバーラップ×直方体版：航行可能“直方体”抽出（緑枠のみ出力）】
- LASを Z ≤ Z_LIMIT で制限
- X方向に幅 X_WIDTH、オーバーラップ X_OVERLAP のスラブに分割
- スラブ内の点だけで 3D occupancy を作成（軸：Z×Y×X、解像度：GRID_RES, X_RES）
- 各 X 層の自由空間(YZ)で最大内接長方形を取り、隣接 X 層へ可能な限り拡張 ⇒ 直方体
- 直方体ごとに「上方（+Z）の生占有ありなら不採用」
- 採用直方体の“12辺”をセル中心で緑点化
- 全スラブの緑点を結合し、重複を丸めユニーク化
- 出力は航行可能空間の緑枠のみ（元点群は含めない）
"""

import os
import numpy as np
import laspy
import cv2

# ===== 入出力 =====
INPUT_LAS  = r"/data/0731_suidoubasi_ue.las"
OUTPUT_LAS = r"/output/0811_overlap_cuboids_navspace_edges_only.las"
os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)

# ===== パラメータ =====
Z_LIMIT      = 3.5     # [m] Z上限（occupancy作成に使う高さ範囲）
GRID_RES     = 0.10    # [m] Y/Z解像度
X_WIDTH      = 10.0    # [m] スラブ幅
X_OVERLAP    = 2.0     # [m] スラブオーバーラップ
X_RES        = 0.50    # [m] X方向のボクセル幅（層厚）

# ▼ down-fill（水面高さ近傍の列だけ不可化）※各X層ごとに実施
USE_ANCHOR_DOWNFILL = True
ANCHOR_Z            = 1.9   # [m]
ANCHOR_TOL          = 0.15  # [m]

# ▼ モルフォロジー（各X層のYZに対して）
MORPH_RADIUS   = 21         # 構造要素半径[セル]
USE_MORPH_DIFF = False      # 直方体抽出では通常不要

# ▼ 直方体条件
MIN_RECT_SZ   = 5           # [セル] YZの長方形 最小高さ/最小幅（両方>=）
MIN_X_LAYERS  = 1           # [層]   X方向の最小層数（直方体の奥行き条件）
DEDUP_GREEN   = True
DEDUP_ROUND   = 4           # 1e-4 m で丸めて一意化

# ===== ユーティリティ =====
def to_nd(a):
    return np.asarray(a)

def copy_header_with_metadata(src_header):
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales  = src_header.scales
    header.offsets = src_header.offsets
    if getattr(src_header, "srs", None):   header.srs  = src_header.srs
    if getattr(src_header, "vlrs", None):  header.vlrs.extend(src_header.vlrs)
    if getattr(src_header, "evlrs", None): header.evlrs.extend(src_header.evlrs)
    return header

def find_max_rectangle(bitmap_bool: np.ndarray):
    """True=自由 の2D配列（行=Z, 列=Y）から最大内接長方形を返す。"""
    h, w = bitmap_bool.shape
    height = [0]*w
    best = (0, 0, 0, 0)
    max_area = 0
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
                area  = height[top_idx]*width
                if area > max_area:
                    max_area = area
                    top = i - height[top_idx] + 1
                    left = (stack[-1] + 1) if stack else 0
                    best = (top, left, height[top_idx], width)
    return best

def downfill_only_near_anchor(grid_uint8, z_min, grid_res, anchor_z, tol):
    """アンカー近傍に占有がある列だけ最上段まで down-fill（不可化）。"""
    occ = (grid_uint8 > 0)
    gh, gw = occ.shape
    i_anchor = int(round((anchor_z - z_min) / grid_res))
    pad = max(0, int(np.ceil(tol / grid_res)))
    i_lo = max(0, i_anchor - pad)
    i_hi = min(gh - 1, i_anchor + pad)
    if i_lo > gh - 1 or i_hi < 0:
        return (occ.astype(np.uint8) * 255)
    for j in range(gw):
        col = occ[:, j]
        idx = np.where(col)[0]
        if idx.size == 0:
            continue
        if np.any((idx >= i_lo) & (idx <= i_hi)):
            imax = idx.max()
            col[:imax + 1] = True
            occ[:, j] = col
    return (occ.astype(np.uint8) * 255)

def rectangle_has_raw_points_above(grid_raw_bool_3d, x_l, x_r, top, left, h, w):
    """
    3D生占有 grid_raw_bool_3d[Z,Y,X] に対し、
    直方体 (x∈[x_l,x_r], y∈[left,left+w), z∈[top,top+h)) の“上”に占有があるか？
    → z >= top + h, 同じ y/x 範囲で True が1つでもあれば True
    """
    gh, gw, gx = grid_raw_bool_3d.shape
    z_above = top + h
    if z_above >= gh:
        return False
    sub = grid_raw_bool_3d[z_above:gh, left:left+w, x_l:x_r+1]
    return np.any(sub)

# ===== メイン =====
def main():
    las = laspy.read(INPUT_LAS)
    dims = set(las.point_format.dimension_names)

    X = to_nd(las.x).astype(float)
    Y = to_nd(las.y).astype(float)
    Z = to_nd(las.z).astype(float)

    # Z制限
    mZ = (Z <= Z_LIMIT)
    X = X[mZ]; Y = Y[mZ]; Z = Z[mZ]
    if len(X) == 0:
        raise RuntimeError("Z制限内の点がありません。")

    # 全体範囲とグリッド数
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    z_min, z_max = Z.min(), Z_LIMIT
    gw = int(np.ceil((y_max - y_min) / GRID_RES))  # Yセル数
    gh = int(np.ceil((z_max - z_min) / GRID_RES))  # Zセル数

    green_pts = []

    # Xスラブ開始位置（オーバーラップ）
    step_x = max(1e-6, X_WIDTH - X_OVERLAP)
    slab_starts = np.arange(x_min, x_max, step_x)

    for x0 in slab_starts:
        x1 = x0 + X_WIDTH
        m_blk = (X >= x0) & (X < x1)
        if not np.any(m_blk):
            continue

        bx, by, bz = X[m_blk], Y[m_blk], Z[m_blk]

        # ---- このスラブをさらに X_RES で層分割（X層ごとにYZマップ）----
        x_edges = np.arange(x0, x1 + 1e-9, X_RES)
        if len(x_edges) < 2:
            x_edges = np.array([x0, x1])
        gx = len(x_edges) - 1  # X層数

        # 3D占有（Z×Y×X）
        grid = np.zeros((gh, gw, gx), dtype=np.uint8)

        # ボクセル割当
        yi = ((by - y_min) / GRID_RES).astype(int)
        zi = ((bz - z_min) / GRID_RES).astype(int)
        xi = np.clip(np.digitize(bx, x_edges) - 1, 0, gx-1)
        ok = (yi >= 0) & (yi < gw) & (zi >= 0) & (zi < gh) & (xi >= 0) & (xi < gx)
        yi = yi[ok]; zi = zi[ok]; xi = xi[ok]
        grid[zi, yi, xi] = 255

        # 生占有コピー
        grid_raw = grid.copy()

        # 各 X 層の処理：down-fill & クロージング → 自由空間
        free_layers = []
        grid_raw_layers = []  # 上方チェック用（bool）
        for xi_ in range(gx):
            layer = grid[:, :, xi_].copy()
            if USE_ANCHOR_DOWNFILL:
                layer = downfill_only_near_anchor(
                    layer, z_min=z_min, grid_res=GRID_RES,
                    anchor_z=ANCHOR_Z, tol=ANCHOR_TOL
                )
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*MORPH_RADIUS+1, 2*MORPH_RADIUS+1))
            closed = cv2.morphologyEx(layer, cv2.MORPH_CLOSE, kernel)

            free = (closed == 0)  # True=自由
            free_layers.append(free)
            grid_raw_layers.append((grid_raw[:, :, xi_] > 0))

        free_layers = np.stack(free_layers, axis=2)          # (Z,Y,X)
        grid_raw_bool_3d = np.stack(grid_raw_layers, axis=2) # (Z,Y,X)

        # ---- 直方体貪欲詰め：seedを取り、X方向にできるだけ延長 ----
        free_work = free_layers.copy()

        while True:
            # 各X層の最大長方形を試算し、一番面積が大きいものをseedにする
            best = None  # (area, xi, top, left, h, w)
            for xi_ in range(gx):
                if not np.any(free_work[:, :, xi_]):
                    continue
                top, left, h, w = find_max_rectangle(free_work[:, :, xi_])
                if h >= MIN_RECT_SZ and w >= MIN_RECT_SZ:
                    area = h * w
                    if (best is None) or (area > best[0]):
                        best = (area, xi_, top, left, h, w)
            if best is None:
                break  # もう直方体を作れない

            _, xi_seed, top, left, h, w = best

            # X方向へ拡張（右へ）
            xr = xi_seed
            while xr + 1 < gx:
                patch = free_work[top:top+h, left:left+w, xr+1]
                if patch.size == 0 or not np.all(patch):
                    break
                xr += 1
            # X方向へ拡張（左へ）
            xl = xi_seed
            while xl - 1 >= 0:
                patch = free_work[top:top+h, left:left+w, xl-1]
                if patch.size == 0 or not np.all(patch):
                    break
                xl -= 1

            # 最終直方体：x∈[xl,xr], y∈[left,left+w), z∈[top,top+h)
            if (xr - xl + 1) < MIN_X_LAYERS:
                # 小さすぎるので、seed層だけ使用済みにして続行
                free_work[top:top+h, left:left+w, xi_seed] = False
                continue

            # 上方チェック（生占有3D）
            if rectangle_has_raw_points_above(grid_raw_bool_3d, xl, xr, top, left, h, w):
                # 不採用：この領域を使用不可にして次へ
                free_work[top:top+h, left:left+w, xl: xr+1] = False
                continue

            # 採用：12辺を緑点化
            def y_from(j): return y_min + (j + 0.5) * GRID_RES
            def z_from(i): return z_min + (i + 0.5) * GRID_RES
            def x_from(k): return (x_edges[k] + x_edges[k+1]) * 0.5

            y_lo = left
            y_hi = left + w - 1
            z_lo = top
            z_hi = top + h - 1

            # X面（前後）に矩形の4辺
            def add_rect_edges_at_x(k):
                x_c = x_from(k)
                for jj in range(y_lo, y_hi+1):
                    green_pts.append([x_c, y_from(jj), z_from(z_lo)])
                    green_pts.append([x_c, y_from(jj), z_from(z_hi)])
                for ii in range(z_lo, z_hi+1):
                    green_pts.append([x_c, y_from(y_lo), z_from(ii)])
                    green_pts.append([x_c, y_from(y_hi), z_from(ii)])

            add_rect_edges_at_x(xl)
            add_rect_edges_at_x(xr)

            # X方向の4本の稜線（y=y_lo/y_hi, z=z_lo/z_hi）を xl..xr の各層で
            for k in range(xl, xr+1):
                x_c = x_from(k)
                green_pts.append([x_c, y_from(y_lo), z_from(z_lo)])
                green_pts.append([x_c, y_from(y_lo), z_from(z_hi)])
                green_pts.append([x_c, y_from(y_hi), z_from(z_lo)])
                green_pts.append([x_c, y_from(y_hi), z_from(z_hi)])

            # 使用済み領域を不可に
            free_work[top:top+h, left:left+w, xl: xr+1] = False

    # ===== 出力（緑枠のみ） =====
    green_pts = np.asarray(green_pts, dtype=float) if len(green_pts) else np.empty((0,3))
    if DEDUP_GREEN and len(green_pts):
        green_pts = np.unique(np.round(green_pts, DEDUP_ROUND), axis=0)

    if len(green_pts) == 0:
        raise RuntimeError("航行可能直方体の“緑枠”が生成されませんでした。パラメータを調整してください。")

    header = copy_header_with_metadata(las.header)
    las_out = laspy.LasData(header)
    N = green_pts.shape[0]
    try:
        las_out.points = laspy.ScaleAwarePointRecord.zeros(N, header=header)
    except AttributeError:
        las_out.points = laspy.PointRecord.zeros(N, header=header)

    las_out.x = green_pts[:,0]
    las_out.y = green_pts[:,1]
    las_out.z = green_pts[:,2]

    # フォーマットがRGB対応なら緑で塗る
    if {"red","green","blue"} <= set(las_out.point_format.dimension_names):
        las_out.red   = np.zeros(N, dtype=np.uint16)
        las_out.green = np.full (N, 65535, dtype=np.uint16)
        las_out.blue  = np.zeros(N, dtype=np.uint16)

    las_out.write(OUTPUT_LAS)
    print("✅ 出力:", OUTPUT_LAS)
    print(f"  緑枠点数: {N:,d}")

if __name__ == "__main__":
    main()
