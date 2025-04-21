import os
import laspy
import numpy as np
import matplotlib.pyplot as plt

# フォルダパス
folder_path = r"C:\Users\owner\ドキュメント\研究１\lasデータ"

# 読み込みファイルをリストアップ（.lasすべて）
las_files = [f for f in os.listdir(folder_path) if f.endswith(".las")]

z_values = []

for fname in las_files:
    try:
        fpath = os.path.join(folder_path, fname)
        las = laspy.read(fpath)
        z = las.z
        z_values.extend(z)
        print(f"✅ {fname}：{len(z)} 点")
    except Exception as e:
        print(f"⚠️ 読み込み失敗 {fname}：{e}")

# ヒストグラム表示
plt.hist(z_values, bins=300, color="skyblue")
plt.xlabel("Z値（高さ）")
plt.ylabel("点数")
plt.title("Z値の分布（点群の高さ）")
plt.grid(True)
plt.show()
