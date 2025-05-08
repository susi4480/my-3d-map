import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_path = r"C:\Users\user\Documents\lab\devine_data\x0y0search\kandagawa_saigo.asc"

# 安全な読み込み処理（Shift_JIS対応）
valid_lines = []
with open(file_path, "rb") as f:
    for line in f:
        try:
            decoded = line.decode("shift_jis").strip()
            parts = decoded.split(",")
            if len(parts) == 3 and all(p.replace('.', '', 1).replace('-', '', 1).isdigit() for p in parts):
                valid_lines.append([float(p) for p in parts])
        except Exception:
            continue

data = np.array(valid_lines)
df = pd.DataFrame(data, columns=["X", "Y", "Z"])

# --- 以下は元のままでOK ---
df_sorted = df.sort_values(by=["X", "Y"]).reset_index(drop=True)
df_sorted["dZ"] = df_sorted["Z"].diff().abs()
threshold = 0.1
wall_candidates = df_sorted[df_sorted["dZ"] > threshold]
floor_points = df_sorted[df_sorted["dZ"] <= threshold]

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
