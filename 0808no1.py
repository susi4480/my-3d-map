# -*- coding: utf-8 -*-
"""
【機能】中心線に沿って常に垂直な厚みスライス（例:10m）で航行可能空間（緑点）を抽出
- 入力LAS: Z ≤ Z_LIMIT に制限
- 中心線CSV（X,Y）を読み込み、各点の接線→法線を算出
- 各中心線位置で、接線に垂直な帯（±SLAB_THICK/2）に入る点群を抽出
- 帯を (法線N, Z) 平面にビットマップ化 → クロージング → 補間セルを緑点へ
- 緑点は帯中央（中心線位置）に配置し、元座標へ戻す
- 出力LASは元ヘッダ（スケール/オフセット/CRS）を継承
"""

import os
import numpy as np
import laspy
import cv2

# === 入出力 ===
INPUT_LAS      = r"C:\Users\user\Documents\lab\outcome\0731_suidoubasi_ue.las"
CENTERLINE_CSV = r"C:\Users\user\Documents\lab\data\centerline_xy.csv"  # X,Y（ヘッダなし/カンマ区切り）
OUTPUT_LAS     = r"C:\Users\user\Documents\lab\output_las\0808_centerline_perp_10m_green.las"

# === パラメータ ===s
Z_LIMIT      = 1.5
GRID_RES     = 0.1          # N-Z平面ラスタ解像度[m]
MORPH_RADIUS = 3            # クロージング半径[セル]
SLAB_THICK   = 10.0         # 垂直スライス厚み（接線方向に±SLAB_THICK/2）[m]
STEP_EVERY   = 1            # 何点おきに中心線をサンプリングするか（1=全点）
MIN_PIXELS   = 50           # スライス内が少なすぎる時スキップ
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*MORPH_RADIUS+1, 2*MORPH_RADIUS+1))

# === 中心線読込（X,Y） ===
def load_centerline_xy(path):
    # カンマ区切りCSV（ヘッダなし）を想定。頑丈めに読み込む。
    try:
        arr = np.loadtxt(path, delimiter=",", dtype=float)
    except Exception:
        arr = np.genfromtxt(path, delimiter=",", dtype=float, comments=None)
    if arr.ndim == 1:
        if arr.size == 2:
            arr = arr.reshape(1, 2)
        else:
            raise RuntimeError(f"Unexpected centerline CSV shape: {arr.shape}")
    if arr.shape[1] > 2:
        arr = arr[:, :2]
    return arr

# === 接線・法線ベクトル列を作る ===
def centerline_tangent_normal(xy):
    # 前後差分で接線を近似、正規化
    M = xy.shape[0]
    t = np.zeros((M,2), dtype=float)
    if M >= 2:
        t[1:-1] = xy[2:] - xy[:-2]
        t[0]    = xy[1] - xy[0]
        t[-1]   = xy[-1] - xy[-2]
    # 正規化
    norm = np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    t /= norm
    # 法線（左法線）
    n = np.stack([-t[:,1], t[:,0]], axis=1)
    return t, n

# === メイン ===
# 出力フォルダ作成
os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)

# LAS読み込み
las = laspy.read(INPUT_LAS)
pts = np.vstack([las.x, las.y, las.z]).T
rgb = np.vstack([las.red, las.green, las.blue]).T / 65535.0

# Z制限
mask = pts[:,2] <= Z_LIMIT
pts  = pts[mask]
rgb  = rgb[mask]
if len(pts) == 0:
    raise RuntimeError("⚠ Z制限後に点がありません")

# 中心線と接線・法線
cl = load_centerline_xy(CENTERLINE_CSV)        # (M,2)
t, n = centerline_tangent_normal(cl)           # (M,2) each

green_world = []

# 各中心線点で垂直スライス
half_thick = SLAB_THICK * 0.5
for idx in range(0, len(cl), STEP_EVERY):
    c  = cl[idx]   # 中心線位置 [X,Y]
    ti = t[idx]    # 接線 (unit)
    ni = n[idx]    # 法線 (unit)

    # XY平面で点をローカル座標へ投影：s=接線成分, u=法線成分
    dxy = pts[:, :2] - c  # (N,2)
    s = np.einsum('ij,j->i', dxy, ti)  # 接線方向スカラー
    u = np.einsum('ij,j->i', dxy, ni)  # 法線方向スカラー

    # 接線方向 |s| ≤ half_thick の帯だけ採用（“垂直スライス”）
    band_mask = np.abs(s) <= half_thick
    if not np.any(band_mask):
        continue

    slab_pts = pts[band_mask]
    u_slab   = u[band_mask]            # 法線座標
    z_slab   = slab_pts[:,2]

    # N-Z平面へラスタ化
    if u_slab.size < MIN_PIXELS:
        continue

    u_min, u_max = u_slab.min(), u_slab.max()
    z_min, z_max = z_slab.min(), z_slab.max()
    gw = int(np.ceil((u_max - u_min) / GRID_RES))
    gh = int(np.ceil((z_max - z_min) / GRID_RES))
    if gw <= 1 or gh <= 1:
        continue

    grid = np.zeros((gh, gw), dtype=np.uint8)
    ui = ((u_slab - u_min) / GRID_RES).astype(int)
    zi = ((z_slab - z_min) / GRID_RES).astype(int)
    ok = (zi >= 0) & (zi < gh) & (ui >= 0) & (ui < gw)
    grid[zi[ok], ui[ok]] = 255

    # クロージング
    closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, KERNEL)
    diff = (closed > 0) & (grid == 0)
    if not np.any(diff):
        continue

    # 補間セルを緑点へ（セル中心→世界座標）
    ii, jj = np.where(diff)
    u_cent = u_min + (jj + 0.5) * GRID_RES
    z_cent = z_min + (ii + 0.5) * GRID_RES

    # 垂直スライス：中心線上（s=0）に置く → p = C + u*ni + 0*ti
    pxy = c + (u_cent[:, None] * ni[None, :])          # (K,2)
    pz  = z_cent[:, None]
    pw  = np.hstack([pxy, pz])                         # (K,3)
    green_world.append(pw)

# 連結
if len(green_world) == 0:
    raise RuntimeError("⚠ 緑点が生成されませんでした（パラメータや中心線を見直してください）")

green_world = np.vstack(green_world)
green_rgb = np.tile(np.array([[0.0, 1.0, 0.0]]), (len(green_world), 1))

# 出力統合
all_pts   = np.vstack([pts, green_world])
all_rgb   = np.vstack([rgb, green_rgb])
all_rgb16 = (all_rgb * 65535).astype(np.uint16)

# ヘッダ継承（laspy互換）＋ ポイント配列を“必要数で再割当”してから代入
try:
    header = las.header.copy()
except AttributeError:
    header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
    header.scales  = las.header.scales.copy()
    header.offsets = las.header.offsets.copy()

las_out = laspy.LasData(header)

# ★★ ここが重要：points を現在の点数で確保し直す（broadcast エラー対策） ★★
n = all_pts.shape[0]
try:
    las_out.points = laspy.ScaleAwarePointRecord.zeros(n, header=header)
except AttributeError:
    las_out.points = laspy.PointRecord.zeros(n, header=header)

# 代入
las_out.x = all_pts[:,0]
las_out.y = all_pts[:,1]
las_out.z = all_pts[:,2]
las_out.red   = all_rgb16[:,0]
las_out.green = all_rgb16[:,1]
las_out.blue  = all_rgb16[:,2]

# 保存
las_out.write(OUTPUT_LAS)

print(f"✅ 出力: {OUTPUT_LAS}")
print(f"  中心線点数: {len(cl)} / 使用点数: {len(range(0,len(cl),STEP_EVERY))}")
print(f"  元点数: {len(pts)} / 緑点数: {len(green_world)}")
