# -*- coding: utf-8 -*-
"""
M5（3D占有ボクセル接続）単体実行版（白色点群を除外する版）
- 既存の中心線→帯抽出→v–zビットマップ（closing+anchor-downfill）
- 各スライスの free_bitmap を収集
- 3Dボクセル(mask[u,z,v])に積層 → 境界セル中心のみ世界座標で LAS 出力
- 入力LASの白色点群（[65535,65535,65535]）は無視（使わない）
"""

import os
import math
import numpy as np
import laspy
import cv2

# ===== 入出力 =====
INPUT_LAS  = r"/output/0731_suidoubasi_ue.las"    # ←ここを分類済みLASに
OUTPUT_LAS = r"/output/0815no1_20_M5_voxel_only.las"
os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)

# ===== パラメータ（中心線・断面）=====
UKC = -1.0                  # [m] 左右岸抽出に使う水面下閾値（中心線用）
BIN_X = 2.0                 # [m] 中心線作成時の X ビン幅
MIN_PTS_PER_XBIN = 50       # 各 X ビンに必要な最小点数
GAP_DIST = 50.0             # [m] 中心線候補の間引き距離
SECTION_INTERVAL = 0.5      # [m] 断面（中心線内挿）間隔
LINE_LENGTH = 60.0          # [m] 法線方向の全長（±半分使う）
SLICE_THICKNESS = 0.20      # [m] 接線方向の薄さ（u=±厚/2）
MIN_PTS_PER_SLICE = 80      # [点] 各帯の最低点数

# ===== 航行可能空間に使う高さ制限 =====
Z_MAX_FOR_NAV = 1.9         # [m] この高さ以下の点だけで航行空間を判定

# ===== v–z 断面のoccupancy =====
GRID_RES = 0.10             # [m/セル] v,z 解像度
MORPH_RADIUS = 20           # [セル] クロージング構造要素半径
USE_ANCHOR_DOWNFILL = True  # 水面高さ近傍で down-fill を有効化
ANCHOR_Z = 1.50             # [m]
ANCHOR_TOL = 0.5            # [m]
MIN_RECT_SIZE = 5           # [セル] （M5では直接使わないが関数互換のため残置）

# ===== 3Dボクセル =====
VOXEL_RES_V = GRID_RES
VOXEL_RES_S = SECTION_INTERVAL
VOXEL_BORDER_ONLY = True     # 連結自由空間の境界セルのみ出力

# ==== ユーティリティ ====
def copy_header_with_metadata(src_header):
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

def l2(p, q):
    return math.hypot(q[0]-p[0], q[1]-p[1])

def downfill_on_closed(closed_uint8, z_min, grid_res, anchor_z, tol):
    """補間後の占有に対して、アンカー帯にヒットする列を下に埋める"""
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
        if not np.any(col): 
            continue
        if np.any(col[i_lo:i_hi+1]):
            imax = np.max(np.where(col)[0])
            out[:imax+1, j] = True
    return (out.astype(np.uint8) * 255)

def rectangles_and_free(points_vz, grid_res, morph_radius, use_anchor, anchor_z, anchor_tol):
    """
    v–z occupancyを作り、自由空間 free_bitmap（True=自由）とbboxを返す。
    """
    if len(points_vz) == 0:
        return None, None

    v_min, v_max = points_vz[:,0].min(), points_vz[:,0].max()
    z_min, z_max = points_vz[:,1].min(), points_vz[:,1].max()
    gw = max(1, int(np.ceil((v_max - v_min) / grid_res)))
    gh = max(1, int(np.ceil((z_max - z_min) / grid_res)))
    grid_raw = np.zeros((gh, gw), dtype=np.uint8)

    yi = ((points_vz[:,0] - v_min) / grid_res).astype(int)
    zi = ((points_vz[:,1] - z_min) / grid_res).astype(int)
    ok = (yi >= 0) & (yi < gw) & (zi >= 0) & (zi < gh)
    grid_raw[zi[ok], yi[ok]] = 255

    # closing & (optional) anchor down-fill
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*morph_radius+1, 2*morph_radius+1))
    closed0 = cv2.morphologyEx(grid_raw, cv2.MORPH_CLOSE, kernel)
    closed  = downfill_on_closed(closed0, z_min, grid_res, anchor_z, anchor_tol) if use_anchor else closed0

    free_bitmap = ~(closed > 0)  # True=自由
    bbox = (v_min, z_min, gw, gh)
    return free_bitmap, bbox

def vz_to_world_on_slice(vz, c, n_hat):
    """(v,z) -> 世界座標 (x,y,z)  ※u=0（帯の中心線上）"""
    v, z = vz
    p_xy = c + v * n_hat
    return [p_xy[0], p_xy[1], z]

def write_green_las(path, header_src, pts_xyz):
    header = copy_header_with_metadata(header_src)
    las_out = laspy.LasData(header)
    N = len(pts_xyz)
    try:
        las_out.points = laspy.ScaleAwarePointRecord.zeros(N, header=header)
    except AttributeError:
        las_out.points = laspy.PointRecord.zeros(N, header=header)
    pts_xyz = np.asarray(pts_xyz, float)
    las_out.x = pts_xyz[:,0]; las_out.y = pts_xyz[:,1]; las_out.z = pts_xyz[:,2]
    if {"red","green","blue"} <= set(las_out.point_format.dimension_names):
        las_out.red   = np.zeros(N, dtype=np.uint16)
        las_out.green = np.full (N, 65535, dtype=np.uint16)
        las_out.blue  = np.zeros(N, dtype=np.uint16)
    las_out.write(path)
    print(f"✅ 出力: {path}  点数: {N}")

def marching_border_3d(mask):
    """3D bool配列の境界セルをTrueに（6近傍）。scipyが無ければフォールバック"""
    try:
        from scipy.ndimage import binary_erosion
        core = binary_erosion(
            mask,
            structure=np.array([[[0,0,0],[0,1,0],[0,0,0]],
                                [[0,1,0],[1,1,1],[0,1,0]],
                                [[0,0,0],[0,1,0],[0,0,0]]], dtype=bool),
            border_value=False
        )
        return mask & (~core)
    except Exception:
        U,Z,V = mask.shape
        border = np.zeros_like(mask)
        for u in range(U):
            for z in range(Z):
                for v in range(V):
                    if not mask[u,z,v]: 
                        continue
                    nbr = 0
                    for du,dz,dv in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                        uu,zz,vv = u+du, z+dz, v+dv
                        if 0<=uu<U and 0<=zz<Z and 0<=vv<V:
                            if not mask[uu,zz,vv]: nbr += 1
                        else:
                            nbr += 1
                    if nbr>0: border[u,z,v]=True
        return border

# ====== M5（ボクセル接続） ======
def method5_voxel_connect(slices_meta, header_src, out_path):
    """自由空間を3Dグリッド化して連結（境界セル中心を出力）"""
    if not slices_meta:
        write_green_las(out_path, header_src, [])
        return

    grid_res = slices_meta[0]["grid_res"]
    v_all_min = min(s["v_min"] for s in slices_meta)
    z_all_min = min(s["z_min"] for s in slices_meta)
    v_all_max = max(s["v_min"] + s["gw"]*grid_res for s in slices_meta)
    z_all_max = max(s["z_min"] + s["gh"]*grid_res for s in slices_meta)
    gw = int(np.ceil((v_all_max - v_all_min)/grid_res))
    gh = int(np.ceil((z_all_max - z_all_min)/grid_res))
    gu = len(slices_meta)

    # mask[u,z,v] = True（自由）
    mask = np.zeros((gu, gh, gw), dtype=bool)
    for u, s in enumerate(slices_meta):
        fb = s["free_bitmap"]
        if fb is None: 
            continue
        off_v = int(round((s["v_min"] - v_all_min)/grid_res))
        off_z = int(round((s["z_min"] - z_all_min)/grid_res))
        h,w = fb.shape
        mask[u, off_z:off_z+h, off_v:off_v+w] = fb

    # 連結自由空間の境界抽出
    border = marching_border_3d(mask)

    # 境界セル中心を世界座標で出力
    ALL=[]
    for u in range(gu):
        c = slices_meta[u]["c"]; n_hat = slices_meta[u]["n_hat"]
        zz, vv = np.where(border[u])  # 行=Z, 列=V
        for z_i, v_i in zip(zz, vv):
            v = v_all_min + (v_i + 0.5)*grid_res
            z = z_all_min + (z_i + 0.5)*grid_res
            ALL.append(vz_to_world_on_slice([v,z], c, n_hat))

    write_green_las(out_path, header_src, ALL)

# ===== メイン =====
def main():
    las = laspy.read(INPUT_LAS)

    # ==== ここから白色点群を除外 ====
    # RGBが定義されている場合のみ
    if {"red", "green", "blue"} <= set(las.point_format.dimension_names):
        R = np.asarray(las.red)
        G = np.asarray(las.green)
        B = np.asarray(las.blue)
        # 白色点群（65535,65535,65535）は除外
        is_not_white = ~((R==65535) & (G==65535) & (B==65535))
        X = np.asarray(las.x, float)[is_not_white]
        Y = np.asarray(las.y, float)[is_not_white]
        Z = np.asarray(las.z, float)[is_not_white]
    else:
        # RGBがない場合はそのまま
        X = np.asarray(las.x, float)
        Y = np.asarray(las.y, float)
        Z = np.asarray(las.z, float)
    xy  = np.column_stack([X, Y])

    # --- 中心線（UKCで左右岸→中点） ---
    x_min, x_max = xy[:,0].min(), xy[:,0].max()
    edges = np.arange(x_min, x_max + BIN_X, BIN_X)
    through = []
    for i in range(len(edges)-1):
        x0, x1 = edges[i], edges[i+1]
        m = (xy[:,0] >= x0) & (xy[:,0] < x1)
        if np.count_nonzero(m) < MIN_PTS_PER_XBIN:
            continue
        slab_xy = xy[m]; slab_z  = Z[m]
        order = np.argsort(slab_xy[:,1])
        slab_xy = slab_xy[order]; slab_z = slab_z[order]
        under = slab_z <= UKC
        if not np.any(under): 
            continue
        idx = np.where(under)[0]
        left  = slab_xy[idx[0]]
        right = slab_xy[idx[-1]]
        c = 0.5*(left + right)
        through.append(c)
    if len(through) < 2:
        raise RuntimeError("中心線が作れません。UKCやBIN_Xを調整してください。")
    through = np.asarray(through, float)

    # --- gap=50mで間引き ---
    thinned = [through[0]]
    for p in through[1:]:
        if l2(thinned[-1], p) >= GAP_DIST:
            thinned.append(p)
    through = np.asarray(thinned, float)

    # --- 中心線を内挿（断面中心列） ---
    centers = []
    for i in range(len(through)-1):
        p, q = through[i], through[i+1]
        d = l2(p, q)
        if d < 1e-9: 
            continue
        n_steps = int(d / SECTION_INTERVAL)
        for s_i in range(n_steps+1):
            s = min(s_i * SECTION_INTERVAL, d)
            t = s / d
            centers.append((1-t)*p + t*q)
    centers = np.asarray(centers, float)

    # --- 各スライス：free_bitmap を作成 ---
    half_len = LINE_LENGTH * 0.5
    half_th  = SLICE_THICKNESS * 0.5

    slices_meta = []  # per slice: {c, n_hat, free_bitmap, v_min, z_min, gw, gh, grid_res}
    for i in range(len(centers)-1):
        c  = centers[i]
        cn = centers[i+1]
        t_vec = cn - c
        norm = np.linalg.norm(t_vec)
        if norm < 1e-9:
            slices_meta.append({
                "c": c, "n_hat": np.array([1.0,0.0]), "grid_res": GRID_RES,
                "v_min": 0, "z_min": 0, "gw": 0, "gh": 0,
                "free_bitmap": None
            })
            continue
        t_hat = t_vec / norm
        n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)

        # 帯抽出: |u|<=half_th, |v|<=half_len
        dxy = xy - c
        u = dxy @ t_hat
        v = dxy @ n_hat
        m_band = (np.abs(u) <= half_th) & (np.abs(v) <= half_len)

        # 高さ制限（z≤Z_MAX_FOR_NAV）
        m_nav = m_band & (Z <= Z_MAX_FOR_NAV)
        if np.count_nonzero(m_nav) < MIN_PTS_PER_SLICE:
            slices_meta.append({
                "c": c, "n_hat": n_hat, "grid_res": GRID_RES,
                "v_min": 0, "z_min": 0, "gw": 0, "gh": 0,
                "free_bitmap": None
            })
            continue

        points_vz = np.column_stack([v[m_nav], Z[m_nav]])
        free_bitmap, bbox = rectangles_and_free(
            points_vz, GRID_RES, MORPH_RADIUS, USE_ANCHOR_DOWNFILL, ANCHOR_Z, ANCHOR_TOL
        )
        if free_bitmap is None:
            slices_meta.append({
                "c": c, "n_hat": n_hat, "grid_res": GRID_RES,
                "v_min": 0, "z_min": 0, "gw": 0, "gh": 0,
                "free_bitmap": None
            })
            continue

        v_min, z_min, gw, gh = bbox
        slices_meta.append({
            "c": c, "n_hat": n_hat, "grid_res": GRID_RES,
            "v_min": v_min, "z_min": z_min, "gw": gw, "gh": gh,
            "free_bitmap": free_bitmap
        })

    # --- M5のみ実行 ---
    method5_voxel_connect(slices_meta, las.header, OUTPUT_LAS)

    print("✅ 完了: M5（ボクセル接続のみ, 白色点群は除外）")
    print(f"  gap=50適用後 中心線点数: {len(through)}")
    print(f"  断面数（内挿）        : {len(centers)}")

if __name__ == "__main__":
    main()
