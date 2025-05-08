import os
import laspy
import numpy as np

# 対象フォルダ
folder_path = r"C:\Users\user\Documents\lab\data\las2"

# .lasファイルをすべて取得
las_files = [f for f in os.listdir(folder_path) if f.endswith(".las")]

# 各ファイルのZ値範囲を表示
for fname in las_files:
    fpath = os.path.join(folder_path, fname)
    las = laspy.read(fpath)
    z = las.z

    print(f"📁 {fname}")
    print(f"    点数: {len(z)}")
    print(f"    Z値の範囲: {np.min(z):.2f} ～ {np.max(z):.2f}")
    print()
