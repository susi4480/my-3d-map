# -*- coding: utf-8 -*-
"""
【機能】LAS点群の中心線をGPUで高速抽出（M5スタイル）
------------------------------------------------------
- CuPyを使用してNumPy計算をGPU化
- DGX / CUDA環境対応（CPU版より数倍〜数十倍高速）
------------------------------------------------------
必要:
  pip install cupy-cuda12x laspy
"""

import cupy as cp
import laspy
import csv
import numpy as np  # CSV出力などCPU側に戻すため併用

# ===== パラメータ =====
INPUT_LAS = "/workspace/data/0828_01_500_suidoubasi_ue.las"
OUTPUT_CSV = "/workspace/output/centerline.csv"
BIN_X = 2.0
MIN_PTS_PER_XBIN = 50
UKC = 0.0
GAP_DIST = 50.0
SECTION_INTERVAL = 0.5

# ===== LAS読み込み（CPU）→ GPU転送 =====
las = laspy.read(INPUT_LAS)
X_cpu, Y_cpu, Z_cpu = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)

# GPUメモリに転送
X = cp.asarray(X_cpu)
Y = cp.asarray(Y_cpu)
Z = cp.asarray(Z_cpu)
xy = cp.column_stack([X, Y])

# --- 中心線計算（GPU版） ---
x_min, x_max = float(X.min()), float(X.max())
edges = cp.arange(x_min, x_max+BIN_X, BIN_X)
through = []

for i in range(len(edges)-1):
    x0, x1 = edges[i], edges[i+1]
    m = (X >= x0) & (X < x1)
    if int(cp.count_nonzero(m)) < MIN_PTS_PER_XBIN:
        continue
    slab_xy = xy[m]
    slab_z = Z[m]

    # Y方向ソート
    order = cp.argsort(slab_xy[:,1])
    slab_xy = slab_xy[order]
    slab_z  = slab_z[order]

    under = slab_z <= UKC
    if not bool(cp.any(under)):
        continue
    idx = cp.where(under)[0]
    left = slab_xy[idx[0]]
    right = slab_xy[idx[-1]]
    through.append(0.5*(left+right))

# CuPy配列をまとめる
if len(through) == 0:
    raise RuntimeError("中心線が抽出できませんでした。")

through = cp.stack(through, axis=0)

# --- GAPフィルタ ---
thinned = [through[0]]
for p in through[1:]:
    if cp.linalg.norm(p - thinned[-1]) >= GAP_DIST:
        thinned.append(p)
through = cp.stack(thinned, axis=0)

# --- スライス中心線出力 ---
centers = []
for i in range(len(through)-1):
    p, q = through[i], through[i+1]
    d = cp.linalg.norm(q - p)
    n_steps = int(d / SECTION_INTERVAL)
    for s_i in range(n_steps+1):
        s = s_i * SECTION_INTERVAL
        t = s / d
        centers.append((1-t)*p + t*q)
centers = cp.stack(centers, axis=0)

# --- CPUに戻してCSV保存 ---
centers_cpu = cp.asnumpy(centers)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "X", "Y"])
    for i, (x, y) in enumerate(centers_cpu):
        writer.writerow([i, x, y])

print(f"✅ GPU版中心線出力完了: {OUTPUT_CSV} ({len(centers_cpu)}点)")
