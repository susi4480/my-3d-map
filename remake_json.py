# -*- coding: utf-8 -*-
"""
【機能】path.jsonを距離間隔で再サンプリング（例：5m間隔）
-------------------------------------------------------
- 入力: path.json （{"path": [[x1, y1], [x2, y2], ...]}）
- 出力: path_resampled.json （等間隔の中心線座標）
- スプラインではなく線形補間で等距離サンプリング
"""

import json
import numpy as np

# ========= 入出力 =========
PATH_IN  = r"C:\Users\user\Documents\lab\data\path.json"
PATH_OUT = r"C:\Users\user\Documents\lab\data\path_resampled.json"
TARGET_INTERVAL = 5.0  # [m] サンプリング間隔

# ========= 再サンプリング処理 =========
with open(PATH_IN, "r", encoding="utf-8") as f:
    data = json.load(f)
path = np.array(data["path"], dtype=np.float64)

# 区間距離
diffs = np.diff(path, axis=0)
dists = np.sqrt((diffs ** 2).sum(axis=1))
cumdist = np.insert(np.cumsum(dists), 0, 0.0)
total_len = cumdist[-1]
print(f"📏 総距離: {total_len:.2f} m")

# 等間隔距離を生成
new_dist = np.arange(0, total_len, TARGET_INTERVAL)
x_new = np.interp(new_dist, cumdist, path[:, 0])
y_new = np.interp(new_dist, cumdist, path[:, 1])
new_path = [[float(x), float(y)] for x, y in zip(x_new, y_new)]

# 出力保存
with open(PATH_OUT, "w", encoding="utf-8") as f:
    json.dump({"path": new_path}, f, ensure_ascii=False, indent=2)

print(f"✅ {len(new_path)} 点を出力 ({TARGET_INTERVAL} m間隔)")
print(f"📂 出力先: {PATH_OUT}")
