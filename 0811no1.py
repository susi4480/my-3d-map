# -*- coding: utf-8 -*-
import os
import numpy as np
import laspy
import cv2

# ========= 入出力 =========
INPUT_LAS  = r"/output/0810no9_rect_edges_only.las"  # 緑枠LAS（各スライスの長方形の縁点のみ）
OUTPUT_LAS = r"/output/0811no1_nav_rect_edges_connections.las"             # つなぎ線のみ出力
os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)

# ========= パラメータ =========
# スライス分割（Xで束ねる）
SLICE_BIN_X = 0.50        # [m] 同一スライスとみなす X の幅（元のSECTION_INTERVALやslice_thicknessに合わせて調整）
MIN_PTS_PER_SLICE = 30    # [点] スライスとして成立する最小点数

# 矩形抽出（スライスのYZで）
GRID_RES = 0.10           # [m/セル] Y/Z のグリッド解像度（元の抽出と合わせる）
DILATE_RADIUS = 1         # 2D膨張半径（セル）：縁点の隙間を繋ぐため
MIN_RECT_CELLS = (5, 5)   # (高さ, 幅) [セル] 最小矩形サイズ（元のMIN_RECT_SIZEに合わせる）

# 対応付け（隣接スライス）
MAX_CENTER_DIST = 1.5     # [m] 中心点の距離ゲート
MAX_SIZE_DIFF   = 2.0     # [m] 高さ・幅それぞれの差の上限
COST_D_W = 1.0            # コスト: 中心距離の重み
COST_DH_W = 0.5           # コスト: 高さ差の重み
COST_DW_W = 0.5           # コスト: 幅差の重み

# 補間
DX_INTERP = 0.25          # [m] 4隅を結ぶ線の X 方向の点間隔

# ========= ユーティリティ =========
def copy_header_with_metadata(src_header):
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales  = src_header.scales
    header.offsets = src_header.offsets
    if getattr(src_header, "srs", None):   header.srs  = src_header.srs
    if getattr(src_header, "vlrs", None):  header.vlrs.extend(src_header.vlrs)
    if getattr(src_header, "evlrs", None): header.evlrs.extend(src_header.evlrs)
    return header

def slice_groups_by_x(X, bin_w=0.5, min_pts=50):
    """Xで等幅ビニング → 各ビンを1スライスとする。戻り: list of (indices, x_center)"""
    x_min, x_max = float(np.min(X)), float(np.max(X))
    edges = np.arange(x_min, x_max + bin_w, bin_w)
    groups = []
    for i in range(len(edges)-1):
        x0, x1 = edges[i], edges[i+1]
        m = (X >= x0) & (X < x1)
        idx = np.where(m)[0]
        if idx.size >= min_pts:
            x_center = 0.5 * (x0 + x1)
            groups.append((idx, x_center))
    return groups

def extract_rects_from_edges(Y, Z, y_min, z_min, y_max, z_max,
                             grid_res=0.10, dilate_r=1, min_cells=(5,5)):
    """
    スライス内の縁点(Y,Z) → グリッド化→膨張→連結成分→矩形（top,left,h,w）を返す。
    """
    if Y.size == 0:
        return []

    gw = max(1, int(np.ceil((y_max - y_min) / grid_res)))
    gh = max(1, int(np.ceil((z_max - z_min) / grid_res)))
    yi = ((Y - y_min) / grid_res).astype(int)
    zi = ((Z - z_min) / grid_res).astype(int)
    ok = (yi >= 0) & (yi < gw) & (zi >= 0) & (zi < gh)
    yi, zi = yi[ok], zi[ok]

    grid = np.zeros((gh, gw), dtype=np.uint8)
    grid[zi, yi] = 255

    if dilate_r > 0:
        k = 2*dilate_r + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        grid = cv2.dilate(grid, kernel, iterations=1)

    # 8近傍で連結成分
    num, labels = cv2.connectedComponents((grid>0).astype(np.uint8), connectivity=8)

    rects = []
    for l in range(1, num):
        mask = (labels == l)
        if not np.any(mask):
            continue
        # ラベル領域のバウンディングBoxを矩形として採用
        zs, ys = np.where(mask)
        top, bottom = int(np.min(zs)), int(np.max(zs))
        left, right = int(np.min(ys)), int(np.max(ys))
        h = bottom - top + 1
        w = right - left + 1
        if h >= min_cells[0] and w >= min_cells[1]:
            rects.append((top, left, h, w))
    return rects

def rect_center_yz(rect, y_min, z_min, grid_res):
    top, left, h, w = rect
    cy = y_min + (left + 0.5 * w) * grid_res
    cz = z_min + (top  + 0.5 * h) * grid_res
    return np.array([cy, cz], dtype=float)

def rect_size_hw_m(rect, grid_res):
    top, left, h, w = rect
    return np.array([h*grid_res, w*grid_res], dtype=float)  # [高さ, 幅]（m）

def rect_corners_yz(rect, y_min, z_min, grid_res):
    top, left, h, w = rect
    corners_idx = [
        (left,         top        ),  # 左下
        (left,         top + h - 1),  # 左上
        (left + w - 1, top + h - 1),  # 右上
        (left + w - 1, top        ),  # 右下
    ]
    return np.array([
        [y_min + (yi + 0.5) * grid_res,
         z_min + (zi + 0.5) * grid_res]
        for (yi, zi) in corners_idx
    ], dtype=float)  # (4,2) = (y,z)

def match_pairs(prev_rects, curr_rects, y_min, z_min, grid_res,
                max_center_dist=1.5, max_size_diff=2.0,
                w_d=1.0, w_dh=0.5, w_dw=0.5):
    """中心＋サイズでゲートしつつ最小コスト貪欲マッチ（一対一）"""
    if not prev_rects or not curr_rects:
        return []

    pc = np.array([rect_center_yz(r,y_min,z_min,grid_res) for r in prev_rects])  # (N,2)
    cc = np.array([rect_center_yz(r,y_min,z_min,grid_res) for r in curr_rects])  # (M,2)
    phw = np.array([rect_size_hw_m(r,grid_res) for r in prev_rects])             # (N,2)
    chw = np.array([rect_size_hw_m(r,grid_res) for r in curr_rects])             # (M,2)

    used = np.zeros(len(curr_rects), dtype=bool)
    pairs = []
    for i in range(len(prev_rects)):
        d  = np.linalg.norm(cc - pc[i], axis=1)          # 中心距離
        dh = np.abs(chw[:,0] - phw[i,0])                 # 高さ差
        dw = np.abs(chw[:,1] - phw[i,1])                 # 幅差
        ok = (~used) & (d <= max_center_dist) & (dh <= max_size_diff) & (dw <= max_size_diff)
        if not np.any(ok):
            continue
        cost = w_d*d + w_dh*dh + w_dw*dw
        j = np.argmin(np.where(ok, cost, np.inf))
        if np.isfinite(cost[j]):
            pairs.append((i, j))
            used[j] = True
    return pairs

def connect_corners(prev_rect, curr_rect, x_prev, x_curr,
                    y_min, z_min, grid_res, dx=0.25):
    """対応した2矩形の4隅をx方向に線形補間して点列を返す"""
    yz0 = rect_corners_yz(prev_rect, y_min, z_min, grid_res)  # (4,2)
    yz1 = rect_corners_yz(curr_rect, y_min, z_min, grid_res)  # (4,2)
    if dx <= 0:
        xs = np.array([x_prev, x_curr])
    else:
        n = max(2, int(np.ceil(abs(x_curr - x_prev) / dx)) + 1)
        xs = np.linspace(x_prev, x_curr, n)

    out = []
    for k in range(4):
        y0,z0 = yz0[k]; y1,z1 = yz1[k]
        for t in np.linspace(0.0, 1.0, len(xs)):
            x = (1-t)*x_prev + t*x_curr
            y = (1-t)*y0     + t*y1
            z = (1-t)*z0     + t*z1
            out.append([x,y,z])
    return np.asarray(out, float)

# ========= メイン =========
def main():
    las = laspy.read(INPUT_LAS)

    X = np.asarray(las.x, float)
    Y = np.asarray(las.y, float)
    Z = np.asarray(las.z, float)

    # 全体のYZ範囲（固定グリッド基準）
    y_min, y_max = float(np.min(Y)), float(np.max(Y))
    z_min, z_max = float(np.min(Z)), float(np.max(Z))

    # Xでスライス分割
    groups = slice_groups_by_x(X, bin_w=SLICE_BIN_X, min_pts=MIN_PTS_PER_SLICE)
    if len(groups) < 2:
        raise RuntimeError("スライスが2枚未満です。SLICE_BIN_XやMIN_PTS_PER_SLICEを調整してください。")

    # スライスごとに矩形を抽出
    slices_rects = []   # list of dict: { 'x':x_center, 'rects':[ (top,left,h,w), ... ] }
    for idx, x_center in groups:
        rects = extract_rects_from_edges(
            Y[idx], Z[idx],
            y_min=y_min, z_min=z_min, y_max=y_max, z_max=z_max,
            grid_res=GRID_RES, dilate_r=DILATE_RADIUS, min_cells=MIN_RECT_CELLS
        )
        slices_rects.append({'x': x_center, 'rects': rects})

    # 隣接スライス間で対応付け＆4隅を接続
    green = []
    for t in range(len(slices_rects)-1):
        prev = slices_rects[t]
        curr = slices_rects[t+1]
        pairs = match_pairs(
            prev['rects'], curr['rects'],
            y_min=y_min, z_min=z_min, grid_res=GRID_RES,
            max_center_dist=MAX_CENTER_DIST, max_size_diff=MAX_SIZE_DIFF,
            w_d=COST_D_W, w_dh=COST_DH_W, w_dw=COST_DW_W
        )
        if not pairs:
            continue
        for (i, j) in pairs:
            pts = connect_corners(
                prev['rects'][i], curr['rects'][j],
                x_prev=prev['x'], x_curr=curr['x'],
                y_min=y_min, z_min=z_min, grid_res=GRID_RES,
                dx=DX_INTERP
            )
            green.append(pts)

    if not green:
        raise RuntimeError("つなぎ線が生成できませんでした。しきい値（中心距離/サイズ差/スライス幅）を調整してください。")

    green = np.vstack(green)  # (N,3)

    # 出力（つなぎ線のみ）
    header = copy_header_with_metadata(las.header)
    las_out = laspy.LasData(header)
    N = green.shape[0]
    try:
        las_out.points = laspy.ScaleAwarePointRecord.zeros(N, header=header)
    except AttributeError:
        las_out.points = laspy.PointRecord.zeros(N, header=header)
    las_out.x = green[:,0]
    las_out.y = green[:,1]
    las_out.z = green[:,2]

    # RGBが使えるフォーマットなら緑に
    if {"red","green","blue"} <= set(las_out.point_format.dimension_names):
        las_out.red   = np.zeros(N, dtype=np.uint16)
        las_out.green = np.full (N, 65535, dtype=np.uint16)
        las_out.blue  = np.zeros(N, dtype=np.uint16)

    las_out.write(OUTPUT_LAS)
    print("✅ 出力:", OUTPUT_LAS)
    print(f"  スライス数: {len(slices_rects)}")
    print(f"  接続点数: {N:,d}")

if __name__ == "__main__":
    main()
