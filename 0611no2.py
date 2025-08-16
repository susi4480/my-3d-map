# -*- coding: utf-8 -*-
"""
X方向にスライスし、各スライスでY-Z平面にα-shapeを適用。
安全距離チェック後、水色で航行空間をマークしたLASを出力。
"""

import os
import sys
import numpy as np
import laspy
import alphashape
from shapely.geometry import Polygon, mapping, Point
from shapely.ops import unary_union
from tqdm import tqdm
import copy

# === 設定 ===
INPUT_LAS   = "/home/edu3/lab/output/0611_las2_full.las"
OUT_DIR     = "/home/edu3/lab/output/navigable_volume"
os.makedirs(OUT_DIR, exist_ok=True)

X_STEP      = 0.5       # スライス幅 [m]
Z_CUTOFF    = 6.0       # Z制限（この上は除外）
MIN_PTS     = 10000     # 最小点数
ALPHA       = 1.5       # α-shape パラメータ
SAFETY_DIST = 2.0       # 安全距離 [m]
MIN_YZ_STD  = 0.5       # YZの広がりがこれ以下なら無視

# === LAS読み込み ===
print("📥 Loading LAS ...")
las = laspy.read(INPUT_LAS)
pts = np.vstack([las.x, las.y, las.z]).T
if len(pts) == 0:
    sys.exit("❌ LAS に点がありません。")

# === Zフィルタリング（上限制限）===
pts = pts[pts[:, 2] < Z_CUTOFF]
if len(pts) == 0:
    sys.exit("❌ Z < 6.0 の点が存在しません。")

x_min, x_max = np.floor(pts[:, 0].min()), np.ceil(pts[:, 0].max())
x_edges = np.arange(x_min, x_max, X_STEP)

slice_polys = []
safe_polys = []

print("✂️  X-slicing & α-shape ...")
for x0 in tqdm(x_edges):
    x1 = x0 + X_STEP
    m  = (pts[:, 0] >= x0) & (pts[:, 0] < x1)
    num_pts = m.sum()
    if num_pts < MIN_PTS:
        print(f"⚠️ 点数不足 x={x0:.2f}-{x1:.2f} → {num_pts}点")
        continue

    yz = pts[m][:, [1, 2]]  # YZ平面
    if np.std(yz[:, 0]) < MIN_YZ_STD or np.std(yz[:, 1]) < MIN_YZ_STD:
        print(f"⚠️ YZの広がり不足 x={x0:.2f}-{x1:.2f} → std=({np.std(yz[:, 0]):.2f}, {np.std(yz[:, 1]):.2f})")
        continue

    try:
        poly = alphashape.alphashape(yz, ALPHA)
    except Exception as e:
        print(f"⚠️ α-shape 失敗 x={x0:.2f}-{x1:.2f} : {e}")
        continue

    if not isinstance(poly, Polygon) or not poly.is_valid or poly.area < 1.0:
        print(f"⚠️ 無効ポリゴン x={x0:.2f}-{x1:.2f} → 面積={getattr(poly, 'area', 0):.2f}")
        continue

    feat = dict(geometry=mapping(poly), properties=dict(x_min=float(x0), x_max=float(x1)))
    slice_polys.append(feat)

# === 安全距離チェック ===
print("🛟 Filtering by safety distance ...")
for feat in slice_polys:
    poly = Polygon(feat["geometry"]["coordinates"][0])
    if poly.buffer(-SAFETY_DIST).is_empty:
        print(f"🛑 安全距離NG x={feat['properties']['x_min']:.2f}-{feat['properties']['x_max']:.2f}")
        continue
    safe_polys.append(feat)

# === 統計出力 ===
print("📊 処理統計")
print(f"・ Xスライス数           : {len(x_edges)}")
print(f"・ α-shape 成功スライス数: {len(slice_polys)}")
print(f"・ 安全距離通過数        : {len(safe_polys)}")
print(f"・ 通過率                 : {len(safe_polys) / len(x_edges) * 100:.2f}%")

# === マスク作成 & 色付け ===
print("🚢 Embedding space into LAS ...")
las_orig = laspy.read(INPUT_LAS)
pts_orig = np.vstack([las_orig.x, las_orig.y, las_orig.z]).T
mask_inside = np.zeros(len(pts_orig), dtype=bool)

yz_orig = pts_orig[:, [1, 2]]
x_orig  = pts_orig[:, 0]

for feat in safe_polys:
    poly = Polygon(feat["geometry"]["coordinates"][0])
    x0, x1 = feat["properties"]["x_min"], feat["properties"]["x_max"]
    in_poly = np.array([poly.contains(Point(p)) for p in yz_orig])
    in_x = (x_orig >= x0) & (x_orig < x1)
    mask_inside |= (in_poly & in_x)

colors = np.ones((len(pts_orig), 3), dtype=np.uint8) * 255  # 全体白
colors[mask_inside] = [0, 255, 255]  # 航行空間 → 水色

# === 書き出し ===
las_out = laspy.create(point_format=las_orig.header.point_format, file_version=las_orig.header.version)
las_out.header = copy.deepcopy(las_orig.header)
las_out.x = las_orig.x
las_out.y = las_orig.y
las_out.z = las_orig.z
las_out.red   = colors[:, 0]
las_out.green = colors[:, 1]
las_out.blue  = colors[:, 2]

out_path = os.path.join(OUT_DIR, "navigable_space_embedded_xslice.las")
las_out.write(out_path)
print(f"✅ Saved LAS with embedded space: {out_path}")
print("🎉 All done!")
