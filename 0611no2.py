# -*- coding: utf-8 -*-
"""
slice_make_navigable_volume.py（統計ログ付き）
Zスライスとalpha shapeで航行可能空間を生成し、ターミナルに統計出力。
"""
import os
import sys
import numpy as np
import laspy
import open3d as o3d
import alphashape
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
import json
from tqdm import tqdm

# === 設定 ===
INPUT_LAS  = "/home/edu1/miyachi/output/0610suidoubasi_Xslice_full.las"
OUT_DIR    = "/home/edu1/miyachi/output/navigable_volume"
os.makedirs(OUT_DIR, exist_ok=True)

Z_STEP     = 0.5       # スライス間隔[m]
MIN_PTS    = 10000     # スライス最小点数
ALPHA      = 1.5       # alpha shape の alpha
SAFETY_DIST = 2.0      # 安全距離[m]

print("\U0001F4E5 Loading LAS ...")
las = laspy.read(INPUT_LAS)
pts = np.vstack([las.x, las.y, las.z]).T
if len(pts) == 0:
    sys.exit("\u274c LAS に点がありません。")

z_min, z_max = np.floor(pts[:, 2].min()), np.ceil(pts[:, 2].max())
z_edges = np.arange(z_min, z_max, Z_STEP)

slice_polys = []
safe_polys = []
print("\u2702\ufe0f  Z-slicing & \u03b1-shape ...")
for z0 in tqdm(z_edges):
    z1 = z0 + Z_STEP
    m  = (pts[:, 2] >= z0) & (pts[:, 2] < z1)
    num_pts = m.sum()
    if num_pts < MIN_PTS:
        print(f"⚠️  点数不足スライス z={z0:.2f}–{z1:.2f} → {num_pts}点")
        continue

    xy = pts[m, :2]
    try:
        poly = alphashape.alphashape(xy, ALPHA)
    except Exception as e:
        print(f"⚠️  α-shape 失敗 z={z0:.2f}–{z1:.2f} : {e}")
        continue

    if not isinstance(poly, Polygon) or not poly.is_valid or poly.area < 1.0:
        print(f"⚠️  無効ポリゴン z={z0:.2f}–{z1:.2f} → 面積={getattr(poly, 'area', 0):.2f}")
        continue

    feat = dict(geometry=mapping(poly), properties=dict(z_min=float(z0), z_max=float(z1)))
    slice_polys.append(feat)

print("\U0001F6DF Filtering by safety distance ...")
for feat in slice_polys:
    poly = Polygon(feat["geometry"]["coordinates"][0])
    if poly.buffer(-SAFETY_DIST).is_empty:
        print(f"🚫 安全距離NGで除外 z={feat['properties']['z_min']:.2f}–{feat['properties']['z_max']:.2f}")
        continue
    safe_polys.append(feat)

print("\U0001F4CA 処理統計")
print(f"\u30fb 全Zスライス数         : {len(z_edges)}")
print(f"\u30fb α-shape 成功スライス数: {len(slice_polys)}")
print(f"\u30fb 安全距離クリア数      : {len(safe_polys)}")
print(f"\u30fb 除外スライス数        : {len(slice_polys) - len(safe_polys)}")
print(f"\u30fb 通過率                 : {len(safe_polys) / len(z_edges) * 100:.2f}%")

# === GeoJSON 保存関数 ===
def save_geojson(name, feats):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        geojson = dict(type="FeatureCollection", features=feats)
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {path}")

save_geojson("slice_polygons_raw.geojson", slice_polys)
save_geojson("slice_polygons_safe.geojson", safe_polys)

# === union 2D footprint ===
union_poly = unary_union([Polygon(f["geometry"]["coordinates"][0]) for f in safe_polys])
union_feat = dict(geometry=mapping(union_poly), properties=dict(description="Overall navigable 2D footprint"))
save_geojson("navigable_footprint.geojson", [union_feat])

# === 3D メッシュ積層 ===
print("\U0001F6E0  Building 3D mesh ...")
mesh_tris = []
mesh_verts = []
vert_id = 0
for feat in safe_polys:
    z0, z1 = feat["properties"]["z_min"], feat["properties"]["z_max"]
    poly = Polygon(feat["geometry"]["coordinates"][0]).simplify(0.2)
    if not poly.exterior.coords:
        continue
    coords = np.array(poly.exterior.coords)
    lower = np.column_stack([coords, np.full(len(coords), z0)])
    upper = np.column_stack([coords, np.full(len(coords), z1)])
    mesh_verts.extend(lower)
    mesh_verts.extend(upper)
    n = len(coords)
    for i in range(n - 1):
        a, b = vert_id + i, vert_id + (i + 1)
        c, d = vert_id + n + i, vert_id + n + (i + 1)
        mesh_tris.append([a, b, c])
        mesh_tris.append([b, d, c])
    vert_id += 2 * n

if mesh_verts and mesh_tris:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_verts))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh_tris))
    mesh.compute_vertex_normals()
    ply_path = os.path.join(OUT_DIR, "navigable_volume.ply")
    o3d.io.write_triangle_mesh(ply_path, mesh)
    print(f"✅ Saved {ply_path}")
else:
    print("⚠️  Mesh 生成対象ポリゴンがありませんでした")

print("\U0001F389 All done!")
