
# -*- coding: utf-8 -*-
"""
【機能】
- 統合済みLASから「Z≤1.0m」の点を抽出
- 2D α-shapeで河川領域ポリゴンを生成
- GeoJSON + PLY（メッシュ）を出力
---------------------------------------------------------
- 入力: 0925_ue_classified.las
- 出力:
    /output/1007_river_outline_alpha_z1.geojson
    /output/1007_river_outline_alpha_z1.ply
---------------------------------------------------------
必要ライブラリ:
pip install laspy alphashape shapely geopandas trimesh
"""

import numpy as np
import laspy
import alphashape
import geopandas as gpd
import trimesh

# === 入出力 ===
INPUT_LAS = r"/workspace/fulldata/0925_ue_classified.las"
OUTPUT_GEOJSON = r"/workspace/output/1007_river_outline_alpha_z1.geojson"
OUTPUT_PLY = r"/workspace/output/1007_river_outline_alpha_z1.ply"

# === パラメータ ===
Z_MAX = 1.0     # 使用する高さ上限[m]
ALPHA = 2.0     # 大きいほど滑らか、小さいほど粗い外形
SAMPLE_STEP = 2 # 5→約1/5点を使用（高速化）

def main():
    # === LAS読み込み ===
    las = laspy.read(INPUT_LAS)
    x, y, z = las.x, las.y, las.z

    # === Zフィルタ（Z≤1.0m）===
    mask = z <= Z_MAX
    xy = np.vstack([x[mask], y[mask]]).T
    print(f"✅ 入力点数: {len(x):,} → Z≤{Z_MAX}m の点数: {len(xy):,}")

    if len(xy) < 10:
        print("⚠️ 有効点が少なすぎます。α-shapeを実行できません。")
        return

    # === サンプリング ===
    if SAMPLE_STEP > 1:
        xy = xy[::SAMPLE_STEP]
        print(f"➡ {len(xy):,} 点に間引き")

    # === α-shape計算（2D）===
    print("▶ α-shape による外郭ポリゴン生成中 ...")
    shape = alphashape.alphashape(xy, ALPHA)

    # === GeoJSON出力 ===
    gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:32654", geometry=[shape])
    gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    print(f"🗺️ GeoJSON出力完了: {OUTPUT_GEOJSON}")

    # === PLY出力用メッシュ生成 ===
    if shape.geom_type == "Polygon":
        # Polygon外郭座標を取得
        coords = np.array(shape.exterior.coords)
        z_val = np.full(len(coords), Z_MAX, dtype=np.float32)  # 一定高さ
        verts = np.column_stack([coords, z_val])

        # 三角化してメッシュ化
        faces = []
        for i in range(1, len(verts) - 1):
            faces.append([0, i, i + 1])

        mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)
        mesh.export(OUTPUT_PLY)
        print(f"🎉 PLYメッシュ出力完了: {OUTPUT_PLY}")
    else:
        print("⚠️ Polygonでない形状のためPLY出力をスキップしました。")

if __name__ == "__main__":
    main()
