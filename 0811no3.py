# -*- coding: utf-8 -*-
"""
PCAで川軸をX'に整列 → X'方向に幅W・オーバーラップ付きの直線スライス（スラブ）
→ 各スラブ内点を Y'–Z ビットマップ化（X'は無視）→ クロージング → （補間後を基準に）アンカーdown-fill
→ 自由空間から最大長方形のみ抽出 → スラブ中心 X'=x'c に縁セル中心を配置して世界座標へ
→ 緑点のみ LAS 出力
"""

import os
import numpy as np
import laspy
import cv2

# ===== 入出力 =====
INPUT_LAS  = r"/data/0731_suidoubasi_ue.las"
OUTPUT_LAS = r"/output/0811no3overlap_straight_slices_rect_edges_only.las"
os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)

# ===== スライス（スラブ）設定 =====
Z_LIMIT        = 1.9     # [m] 入力の高さ制限（まず絞る）
SLAB_WIDTH     = 10.0    # [m] X' 方向のスラブ幅
SLAB_OVERLAP   = 2.0     # [m] オーバーラップ
GRID_RES       = 0.10    # [m/セル] Y',Z 解像度
MORPH_RADIUS   = 21      # [セル] クロージング構造要素半径
USE_ANCHOR_DOWNFILL = True
ANCHOR_Z       = 1.9     # [m] 補間後の占有でこの高さ±tolに当たる列のみ down-fill
ANCHOR_TOL     = 0.45    # [m]
Z_MAX_FOR_NAV  = 1.9     # [m] 航行判定に使う z 上限（スラブ内で z≤ を適用）
MIN_RECT_SIZE  = 5       # [セル] 長方形の最小 h/w（両方>=）
MIN_PTS_PER_SLAB = 500   # [点] スラブ内の最小点数（少なければスキップ）
DEDUP_ROUND    = 4       # 出力の重複丸め（1e-4 m）

# ===== ユーティリティ =====
def to_nd(a): return np.asarray(a, dtype=float)

def copy_header_with_metadata(src_header):
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales  = src_header.scales
    header.offsets = src_header.offsets
    if getattr(src_header, "srs", None):   header.srs  = src_header.srs
    if getattr(src_header, "vlrs", None):  header.vlrs.extend(src_header.vlrs)
    if getattr(src_header, "evlrs", None): header.evlrs.extend(src_header.evlrs)
    return header

def pca_rotation(xy):
    mu = xy.mean(axis=0)
    X0 = xy - mu
    C = np.cov(X0.T)
    _, vecs = np.linalg.eigh(C)
    v1 = vecs[:, -1]  # 主成分（流下軸）
    theta = np.arctan2(v1[1], v1[0])
    c, s = np.cos(-theta), np.sin(-theta)  # XY→X'Y' 回転
    R = np.array([[c, -s],[s, c]], dtype=float)
    return mu, R

def find_max_rectangle(bitmap_bool: np.ndarray):
    # True=自由 の2D配列（行=Z, 列=Y）から最大内接長方形
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
                    top  = i - height[top_idx] + 1
                    left = (stack[-1] + 1) if stack else 0
                    best = (top, left, height[top_idx], width)
    return best  # (top, left, h, w)

def downfill_on_closed(closed_uint8, z_min, grid_res, anchor_z, tol):
    # 補間後の占有でアンカー帯にヒットする列だけ最上段まで down-fill
    closed_bool = (closed_uint8 > 0)
    gh, gw = closed_bool.shape
    i_anchor = int(round((anchor_z - z_min) / grid_res))
    pad = max(0, int(np.ceil(tol / grid_res)))
    i_lo = max(0, i_anchor - pad)
    i_hi = min(gh - 1, i_anchor + pad)
    if i_lo > gh - 1 or i_hi < 0:
        return (closed_bool.astype(np.uint8) * 255)
    out = closed_bool.copy()
    for j in range(gw):
        col = closed_bool[:, j]
        if np.any(col[i_lo:i_hi+1]):
            imax = np.max(np.where(col)[0])
            out[:imax+1, j] = True
    return (out.astype(np.uint8) * 255)

def rectangles_on_slab(points_yz, grid_res, morph_radius,
                       use_anchor, anchor_z, anchor_tol,
                       min_rect_size):
    """
    points_yz: (N,2) with [y', z]（※この中は z≤Z_MAX_FOR_NAV だけ）
    手順：raw占有(Y'Z) → クロージング → （補間後基準）down-fill → 自由空間 → 最大長方形
    """
    if len(points_yz) == 0:
        return [], None, None

    y_min, y_max = points_yz[:,0].min(), points_yz[:,0].max()
    z_min, z_max = points_yz[:,1].min(), points_yz[:,1].max()
    gw = max(1, int(np.ceil((y_max - y_min) / grid_res)))
    gh = max(1, int(np.ceil((z_max - z_min) / grid_res)))
    grid_raw = np.zeros((gh, gw), dtype=np.uint8)

    yi = ((points_yz[:,0] - y_min) / grid_res).astype(int)
    zi = ((points_yz[:,1] - z_min) / grid_res).astype(int)
    ok = (yi >= 0) & (yi < gw) & (zi >= 0) & (zi < gh)
    grid_raw[zi[ok], yi[ok]] = 255

    # 補間（クロージング）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*morph_radius+1, 2*morph_radius+1))
    closed0 = cv2.morphologyEx(grid_raw, cv2.MORPH_CLOSE, kernel)

    # （補間後を基準に）アンカー down-fill
    if use_anchor:
        closed = downfill_on_closed(closed0, z_min, grid_res, anchor_z, anchor_tol)
    else:
        closed = closed0

    closed_bool = (closed > 0)
    free_bitmap = ~closed_bool  # True=自由

    def has_above_after_interp(top, left, h, w):
        z_above = top + h
        if z_above >= closed_bool.shape[0]:
            return False
        sub = closed_bool[z_above:, left:left+w]
        return np.any(sub)

    rect_edge_pts_yz = []
    free_work = free_bitmap.copy()
    while np.any(free_work):
        top, left, h, w = find_max_rectangle(free_work)
        if h < min_rect_size or w < min_rect_size:
            break
        if not has_above_after_interp(top, left, h, w):
            # 縁セル中心 → (y', z)
            for zi in range(top, top+h):
                for yi_ in range(left, left+w):
                    if zi in (top, top+h-1) or yi_ in (left, left+w-1):
                        yv = y_min + (yi_ + 0.5) * grid_res
                        zv = z_min + (zi + 0.5) * grid_res
                        rect_edge_pts_yz.append([yv, zv])
        free_work[top:top+h, left:left+w] = False

    return rect_edge_pts_yz, y_min, z_min

# ===== メイン =====
def main():
    las = laspy.read(INPUT_LAS)
    X = to_nd(las.x); Y = to_nd(las.y); Z = to_nd(las.z)

    # 高さ制限
    mZ = (Z <= Z_LIMIT)
    X = X[mZ]; Y = Y[mZ]; Z = Z[mZ]
    if len(X) == 0:
        raise RuntimeError("Z制限後の点がありません。")

    # PCA回転（XY→X'Y'）
    XY = np.column_stack([X, Y])
    mu, R = pca_rotation(XY)
    XYr = (R @ (XY - mu).T).T  # (x', y')
    Xp = XYr[:,0]  # x'
    Yp = XYr[:,1]  # y'

    # X'スラブ開始位置（オーバーラップ）
    x_min, x_max = Xp.min(), Xp.max()
    step = max(1e-6, SLAB_WIDTH - SLAB_OVERLAP)
    slab_starts = np.arange(x_min, x_max, step)

    green_pts = []

    for x0 in slab_starts:
        x1 = x0 + SLAB_WIDTH
        m = (Xp >= x0) & (Xp < x1)
        if np.count_nonzero(m) < MIN_PTS_PER_SLAB:
            continue

        # スラブ中心の直線位置（ここに“潰す”＝一直線の薄スライス）
        x_c = 0.5*(x0 + x1)

        # 航行高さの点だけを使う
        Zs = Z[m]; Ys = Yp[m]
        m_nav = (Zs <= Z_MAX_FOR_NAV)
        if np.count_nonzero(m_nav) < MIN_PTS_PER_SLAB:
            continue

        points_yz = np.column_stack([Ys[m_nav], Zs[m_nav]])

        rect_edge_yz, y_min, z_min = rectangles_on_slab(
            points_yz,
            grid_res=GRID_RES,
            morph_radius=MORPH_RADIUS,
            use_anchor=USE_ANCHOR_DOWNFILL,
            anchor_z=ANCHOR_Z,
            anchor_tol=ANCHOR_TOL,
            min_rect_size=MIN_RECT_SIZE
        )

        # 出力：スラブ中心X'=x_c に縁セルを配置（＝完全に直線のスライスになる）
        for yv, zv in rect_edge_yz:
            # X'Y'→XY（世界）へ戻す
            XYp = np.array([x_c, yv])
            XYw = (R.T @ XYp) + mu
            green_pts.append([XYw[0], XYw[1], zv])

    if not green_pts:
        raise RuntimeError("航行可能空間の緑枠が生成されませんでした。パラメータを調整してください。")

    green_pts = np.asarray(green_pts, float)
    green_pts = np.unique(np.round(green_pts, DEDUP_ROUND), axis=0)

    # 出力（緑のみ）
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

    # 緑色
    if {"red","green","blue"} <= set(las_out.point_format.dimension_names):
        las_out.red   = np.zeros(N, dtype=np.uint16)
        las_out.green = np.full (N, 65535, dtype=np.uint16)
        las_out.blue  = np.zeros(N, dtype=np.uint16)

    las_out.write(OUTPUT_LAS)
    print("✅ 出力:", OUTPUT_LAS)
    print(f"  直線スライス数: {len(slab_starts)}")
    print(f"  緑点: {N:,d} / GRID_RES={GRID_RES}, MORPH_RADIUS={MORPH_RADIUS}, MIN_RECT_SIZE={MIN_RECT_SIZE}")
    print(f"  anchor(after-closing): z={ANCHOR_Z}±{ANCHOR_TOL}m, Z_MAX_FOR_NAV={Z_MAX_FOR_NAV}")

if __name__ == "__main__":
    main()
