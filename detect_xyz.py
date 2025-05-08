import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ファイルパス（必要に応じて書き換えてください）
file_path = r"C:\Users\user\Documents\lab\devine_data\x0y0search\kandagawa_saigo.asc"

# 数値のみの点群データを読み込み（カンマ区切り）
try:
    data = np.loadtxt(file_path, delimiter=",")
except Exception as e:
    print(f"読み込みエラー: {e}")
    exit()

# DataFrameに変換して列名をつける
df = pd.DataFrame(data, columns=["X", "Y", "Z"])

# 近接点の連続性を確認するため、XとYで並べ替え
df_sorted = df.sort_values(by=["X", "Y"]).reset_index(drop=True)

# Zの差分（絶対値）を計算してdZ列を追加
df_sorted["dZ"] = df_sorted["Z"].diff().abs()

# Zの変化が一定以上（例: 0.1m）なら壁とみなす
threshold = 0.1
wall_candidates = df_sorted[df_sorted["dZ"] > threshold]
floor_points = df_sorted[df_sorted["dZ"] <= threshold]

# 可視化（3D散布図）
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(floor_points["X"], floor_points["Y"], floor_points["Z"], s=1, label='Floor (Low dZ)', alpha=0.5)
ax.scatter(wall_candidates["X"], wall_candidates["Y"], wall_candidates["Z"], s=5, color='red', label='Wall Candidate (High dZ)', alpha=0.7)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('壁候補（Zの急変点）と底面点の分離')
ax.legend()
plt.tight_layout()
plt.show()
