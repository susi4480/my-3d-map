# -*- coding: utf-8 -*-
"""
【機能】
1. 指定フォルダ内の *.xyz（lat  lon  z）をすべて読み込む
2. Z <= 4.5 m の点だけ残す
3. 緯度経度 → UTM Zone 54N（EPSG:32654）に変換
4. Z を削除し、(X, Y) だけの 2D 点群として 1 つの .xyz に書き出す
"""

import os
import glob
import numpy as np
from pyproj import Transformer

# === 設定 ===================================================================
input_dir  = r"C:\Users\user\Documents\lab\data\suidoubasi\lidar_xyz_ue"
output_xyz = r"C:\Users\user\Documents\lab\output_ply\0712_suidoubasi_lidar_ue_2D.xyz"

z_threshold = 3.5                     # m
utm_epsg    = "epsg:32654"            # UTM Zone 54N（東京湾周辺）
transformer = Transformer.from_crs("epsg:4326", utm_epsg, always_xy=True)

# === 1. 入力ファイル収集 =====================================================
xyz_files = glob.glob(os.path.join(input_dir, "*.xyz"))
if not xyz_files:
    raise FileNotFoundError("❌ 指定フォルダに .xyz が見つかりません。")

# === 2. 全ファイル処理 =======================================================
xy_list = []     # 出力用に (X, Y) を溜め込む
total_in, total_out = 0, 0

for path in xyz_files:
    try:
        data = np.loadtxt(path)
        if data.shape[1] < 3:
            print(f"⚠ 列数不足スキップ: {os.path.basename(path)}")
            continue

        lat, lon, z = data[:, 0], data[:, 1], data[:, 2]
        total_in += len(z)

        mask = z <= z_threshold
        if mask.sum() == 0:
            continue

        # 緯度経度 -> UTM
        x, y = transformer.transform(lon[mask], lat[mask])
        xy_list.append(np.vstack([x, y]).T)
        total_out += mask.sum()

    except Exception as e:
        print(f"⚠ 読み込み失敗スキップ: {path} → {e}")

if not xy_list:
    raise RuntimeError("❌ 条件を満たす点が 1 点も得られませんでした。")

# === 3. 結合 & 保存 ==========================================================
points_xy = np.vstack(xy_list)  # (N, 2)
np.savetxt(output_xyz, points_xy, fmt="%.3f")  # 1 mm 単位で保存

print("🎉 完了しました")
print(f"  読み込んだ総点数 : {total_in:,}")
print(f"  出力した点数     : {total_out:,}")
print(f"  出力ファイル     : {output_xyz}")
