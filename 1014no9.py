# -*- coding: utf-8 -*-
"""
【GPU対応版】緑線(最緩線)＋緑壁(縦積み)＋XYグリッド塗りつぶしで“外側を完全削除”
--------------------------------------------------------------------------------
GPU（あれば使用）:
  - OpenCV CUDA: 緑壁ラスタの膨張(Dilate)をGPUで実行
  - CuPy: 大規模ブーリアン配列(wall/insideなど)の生成・論理演算をGPUで実行
CPUにフォールバック:
  - floodFill（OpenCV CUDAに未実装）→ CPUで実行

入出力:
  INPUT_DIR  : スライス矩形LAS群
  OUTPUT_LAS : 内側のみ（灰）＋緑線（最緩線）の最終LAS
"""

import os, re
import numpy as np
import laspy
from glob import glob
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
import cv2

# ====== GPU可否 ======
_HAS_CUDA = False
_HAS_CUPY = False
try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

try:
    _HAS_CUDA = (cv2.cuda.getCudaEnabledDeviceCount() > 0)
except Exception:
    _HAS_CUDA = False

# ===== 入出力 =====
INPUT_DIR = "/workspace/output/917slices_m0style_rect/"
OUTPUT_LAS_FINAL = "/workspace/output/1014_navspace_centercut_innertrim_gridfill.las"

# ===== パラメータ（緑線生成＝元コード同一） =====
ANGLE_THRESH_DEG = 35.0
LOOKAHEAD_SLICES = 30
LINE_STEP = 0.01
UNION_EPS = 1e-6

# ===== 緑壁（高さ範囲とピッチ） =====
Z_MIN_FOR_NAV = -3.0
Z_MAX_FOR_NAV = 1.9
Z_STEP        = 0.05  # ★ この値が XY グリッド解像度と同じになります

# ===== XYグリッド設定（Z_STEPと同じ） =====
GRID_RES = Z_STEP      # ★ ご要望どおり縦積み間隔と一致
DILATE_ITER = 2        # 緑壁を太らせて連続性を担保（必要に応じて調整）

# ===== 着色 =====
COLOR_INNER = (52000, 52000, 52000)  # 灰（トリミング後）
COLOR_GREEN = (0, 65535, 0)          # 緑（最緩線）

# ===== ユーティリティ =====
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

def angle_turn_deg(p_prev, p_curr, p_next):
    a = np.asarray(p_prev[:2]) - np.asarray(p_curr[:2])
    b = np.asarray(p_next[:2]) - np.asarray(p_curr[:2])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9: return 0.0
    cosv = np.clip(np.dot(a,b)/(na*nb), -1.0, 1.0)
    inner = np.degrees(np.arccos(cosv))
    return abs(inner - 180.0)

def pca_rect_corners_safe(pts):
    """PCAベース＋外接矩形（失敗時はAABB）"""
    if pts.shape[0] < 4: return None
    try:
        xy = pts[:, :2]; c = xy.mean(axis=0)
        X = xy - c; C = np.cov(X.T)
        _, _, VT = np.linalg.svd(C); R = VT.T
        uv = X @ R
        umin, vmin = uv.min(axis=0); umax, vmax = uv.max(axis=0)
        corners_uv = np.array([[umin,vmin],[umin,vmax],[umax,vmin],[umax,vmax]])
        corners_xy = corners_uv @ R.T + c
        z_med = np.median(pts[:,2])
        return np.column_stack([corners_xy, np.full(4,z_med)])
    except:
        xy = pts[:, :2]
        xmin, ymin = np.min(xy, axis=0); xmax, ymax = np.max(xy, axis=0)
        z_med = np.median(pts[:,2])
        return np.array([[xmin,ymin,z_med],[xmin,ymax,z_med],[xmax,ymin,z_med],[xmax,ymax,z_med]])

def rect_polygon_from_corners(c4):
    LL, LU, RL, RU = c4
    ring = [tuple(LL[:2]), tuple(RL[:2]), tuple(RU[:2]), tuple(LU[:2])]
    return Polygon(ring)

def clip_and_sample_inside(p1, p2, poly_union, step):
    line = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
    inter = line.intersection(poly_union)
    if inter.is_empty: return np.empty((0,3), float)
    segs = [inter] if isinstance(inter, LineString) else list(inter.geoms)
    out = []
    v2 = np.asarray(p2[:2]) - np.asarray(p1[:2])
    vv = max(np.dot(v2,v2), 1e-12)
    for seg in segs:
        coords = np.asarray(seg.coords, float)
        for s in range(len(coords)-1):
            a2, b2 = coords[s], coords[s+1]
            d2 = np.linalg.norm(b2-a2)
            if d2<1e-9: continue
            n = max(1,int(np.ceil(d2/step)))
            t = np.linspace(0,1,n+1)
            xy = a2[None,:] + (b2-a2)[None,:]*t[:,None]
            proj = np.dot(xy - np.asarray(p1[:2])[None,:], v2)/vv
            proj = np.clip(proj, 0.0, 1.0)
            z = p1[2] + (p2[2]-p1[2])*proj
            out.append(np.column_stack([xy,z]))
    return np.vstack(out) if out else np.empty((0,3), float)

# ===== OpenCV flood fill（CPU） =====
def flood_fill_union_inside_cpu(wall_bool_np, seeds_xy_idx):
    """
    wall_bool_np: (H,W) np.bool_ True=壁 / False=空き
    seeds_xy_idx: [(xi, yi), ...]
    戻り: inside_bool_np: (H,W) np.bool_（壁に遮られず種から到達できる領域の和）
    """
    H, W = wall_bool_np.shape
    inside = np.zeros((H, W), np.uint8)

    # OpenCV仕様: maskは(H+2,W+2)、非0が障害扱い
    base_mask = np.zeros((H+2, W+2), np.uint8)
    base_mask[1:H+1, 1:W+1][wall_bool_np] = 1

    for (xi, yi) in seeds_xy_idx:
        if not (0 <= xi < W and 0 <= yi < H):
            continue
        if wall_bool_np[yi, xi]:
            # 種が壁上→近傍にずらす
            found = False
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    xj, yj = xi+dx, yi+dy
                    if 0 <= xj < W and 0 <= yj < H and not wall_bool_np[yj, xj]:
                        xi, yi = xj, yj; found = True; break
                if found: break
            if not found:
                continue

        mask = base_mask.copy()
        img_dummy = np.zeros((H, W), np.uint8)
        flags = cv2.FLOODFILL_MASK_ONLY | 4 | (255 << 8)  # 4近傍, fill=255
        cv2.floodFill(img_dummy, mask, (xi, yi), 0, flags=flags)
        filled = (mask[1:H+1, 1:W+1] == 255)
        inside |= filled.astype(np.uint8)

    return (inside > 0)

# ===== GPU/CPU 切替ヘルパ =====
def dilate_bool_mask(mask_np, iterations=1):
    """2D 0/1 mask を膨張。GPU(CUDA)があればcv2.cudaで実行。なければCPU。"""
    if iterations <= 0:
        return (mask_np > 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    if _HAS_CUDA:
        try:
            # CUDA入力は8Uを想定
            src = cv2.cuda_GpuMat()
            src.upload(mask_np.astype(np.uint8))
            er = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8U, kernel, iterations=iterations)
            dst = er.apply(src)
            out = dst.download()
            return (out > 0)
        except Exception as e:
            # フォールバック
            pass

    # CPUフォールバック
    out = cv2.dilate(mask_np.astype(np.uint8), kernel, iterations=iterations)
    return (out > 0)

def to_idx_xy_np(arr_xy, x_min, y_min, nx, ny, grid_res):
    xi = ((arr_xy[:,0] - x_min) / grid_res).astype(np.int32)
    yi = ((arr_xy[:,1] - y_min) / grid_res).astype(np.int32)
    xi = np.clip(xi, 0, nx-1)
    yi = np.clip(yi, 0, ny-1)
    return xi, yi

# ===== メイン処理 =====
def main():
    # 1) スライス矩形（緑エッジ群）読み込み
    slice_files = sorted(
        glob(os.path.join(INPUT_DIR,"slice_*_rect.las")),
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

    # 2) 隅系列・矩形ポリゴン
    connect_to = {c: np.array([i+1 for i in range(N)], int) for c in range(4)}
    series = {c: np.array([corners_seq[i][c] for i in range(N)]) for c in range(4)}
    rect_polys = [rect_polygon_from_corners(corners_seq[k]) for k in range(N)]

    # 3) 初期接続（内部用：水色は出力しない）
    bridge_initial = []
    for i in range(N-1):
        corridor = unary_union([rect_polys[i], rect_polys[i+1]]).buffer(UNION_EPS)
        for c in range(4):
            seg = clip_and_sample_inside(series[c][i], series[c][i+1], corridor, LINE_STEP)
            if seg.size > 0:
                bridge_initial.append(seg)
    bridge_initial = np.vstack(bridge_initial) if bridge_initial else np.empty((0,3), float)

    # 4) 角度急変の先読み再接続（元コード同一）
    disabled = {c: np.zeros(N,bool) for c in range(4)}
    for i in range(1, N-1):
        needs_reconnect = any(
            angle_turn_deg(series[c][i-1], series[c][i], series[c][i+1]) >= ANGLE_THRESH_DEG
            for c in range(4) if not disabled[c][i]
        )
        if not needs_reconnect: continue
        last = min(N-1, i+LOOKAHEAD_SLICES)
        best_j, best_score = i+1, (1e18,1e18,1e18)
        for j in range(i+2, last+1):
            angs, dsum = [], 0.0
            for c in range(4):
                p_prev, p_curr, p_j = series[c][i-1], series[c][i], series[c][j]
                angs.append(angle_turn_deg(p_prev, p_curr, p_j))
                dsum += np.linalg.norm(series[c][j,:2]-series[c][i,:2])
            cand = (np.mean(angs), dsum, j-i)
            if cand < best_score:
                best_score, best_j = cand, j
        if best_j != i+1:
            for c in range(4):
                connect_to[c][i] = best_j
                disabled[c][i+1:best_j] = True

    # 5) 最緩線（緑）
    bridge_pts_list = []
    for i in range(N-1):
        j = int(connect_to[1][i])
        if j <= i or j >= N: continue
        corridor = unary_union([rect_polys[k] for k in range(i, j+1)]).buffer(UNION_EPS)
        for c in range(4):
            seg = clip_and_sample_inside(series[c][i], series[c][j], corridor, LINE_STEP)
            if seg.size > 0:
                bridge_pts_list.append(seg)
    bridge_pts = np.vstack(bridge_pts_list) if bridge_pts_list else np.empty((0,3), float)

    # 6) 航行空間（灰＝元スライスの外周点群を集約）
    map_pts = np.vstack(raw_seq)

    # 7) 緑壁を縦積み（グリッドXY投影にはXYのみ利用するため実体生成は不要）
    # z_layers = np.arange(Z_MIN_FOR_NAV, Z_MAX_FOR_NAV + Z_STEP, Z_STEP)

    # 8) XYグリッド作成（解像度＝Z_STEP）
    x_min, x_max = map_pts[:,0].min(), map_pts[:,0].max()
    y_min, y_max = map_pts[:,1].min(), map_pts[:,1].max()
    nx = int(np.ceil((x_max - x_min) / GRID_RES)) + 1
    ny = int(np.ceil((y_max - y_min) / GRID_RES)) + 1

    # 9) “壁”ラスタ（True=壁セル）を作る：緑線をXY上にマーキング
    wall_bool_np = np.zeros((ny, nx), np.uint8)
    xi, yi = to_idx_xy_np(bridge_pts[:, :2], x_min, y_min, nx, ny, GRID_RES)
    wall_bool_np[yi, xi] = 1

    # 9.5) 膨張（GPU/CPU切替）
    wall_bool_np = dilate_bool_mask(wall_bool_np, iterations=DILATE_ITER)  # boolに変換済み

    # 10) 各スライスの中心（ポリゴン重心）を“内側シード”にして flood fill（CPU）
    seeds = []
    for poly in rect_polys:
        c = poly.centroid
        xi_c = int(round((c.x - x_min) / GRID_RES))
        yi_c = int(round((c.y - y_min) / GRID_RES))
        if 0 <= xi_c < nx and 0 <= yi_c < ny:
            seeds.append((xi_c, yi_c))

    # 10.5) flood fill 実行（CPU）
    inside_bool_np = flood_fill_union_inside_cpu(wall_bool_np.astype(bool), seeds)  # bool

    # 11) “内側”セルに落ちる点だけを残す（外側は完全に削除）
    #     CuPyがあればインデクシング部をGPUでやってからCPUへ戻す最小化
    xi_pts, yi_pts = to_idx_xy_np(map_pts[:, :2], x_min, y_min, nx, ny, GRID_RES)

    if _HAS_CUPY:
        try:
            xi_gpu = cp.asarray(xi_pts, dtype=cp.int32)
            yi_gpu = cp.asarray(yi_pts, dtype=cp.int32)
            inside_gpu = cp.asarray(inside_bool_np, dtype=cp.bool_)
            keep_mask_gpu = inside_gpu[yi_gpu, xi_gpu]
            keep_mask = cp.asnumpy(keep_mask_gpu)
        except Exception:
            # フォールバック（CPU）
            keep_mask = inside_bool_np[yi_pts, xi_pts]
    else:
        keep_mask = inside_bool_np[yi_pts, xi_pts]

    map_pts_trim = map_pts[keep_mask]

    # 12) 出力（灰：内側のみ、緑：最緩線）
    out_xyz = np.vstack([map_pts_trim, bridge_pts])
    color_all = np.vstack([
        np.tile(COLOR_INNER, (len(map_pts_trim), 1)),
        np.tile(COLOR_GREEN, (len(bridge_pts), 1))
    ])

    header = copy_header_with_metadata(laspy.read(slice_files[0]).header)
    las_out = laspy.LasData(header)
    ensure_points_alloc(las_out, len(out_xyz))
    las_out.x, las_out.y, las_out.z = out_xyz[:,0], out_xyz[:,1], out_xyz[:,2]
    las_out.red, las_out.green, las_out.blue = color_all[:,0], color_all[:,1], color_all[:,2]

    os.makedirs(os.path.dirname(OUTPUT_LAS_FINAL), exist_ok=True)
    las_out.write(OUTPUT_LAS_FINAL)

    msg_gpu = []
    msg_gpu.append(f"OpenCV CUDA: {'ON' if _HAS_CUDA else 'OFF'}")
    msg_gpu.append(f"CuPy        : {'ON' if _HAS_CUPY else 'OFF'}")

    print(f"✅ 出力完了: {OUTPUT_LAS_FINAL}")
    print(f"  航行空間(灰, 内側のみ): {len(map_pts_trim):,} 点 / 全体: {len(map_pts):,} 点")
    print(f"  緑線: {len(bridge_pts):,} 点")
    print(f"  グリッド: {nx} x {ny}, 解像度: {GRID_RES} m (Z_STEPと同一)")
    print(f"  壁膨張: {DILATE_ITER} iter")
    print("  GPU利用状況: " + " | ".join(msg_gpu))

if __name__ == "__main__":
    main()
