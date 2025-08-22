# -*- coding: utf-8 -*-
"""
【機能】
- LAS点群のランダム1万点から：
    - 最近傍距離（点間隔）の統計を出力
    - 半径内の近傍点数（max_neighborsの目安）を出力
"""

import numpy as np
import laspy
from scipy.spatial import cKDTree

# === 設定 ===
las_path = r"/data/matome/0725_suidoubasi_floor_ue.las"
radius = 1.0        # 半径 [m]
sample_size = 5000  # サンプル点数（重すぎなければ増やしてもOK）

# === LAS読み込み ===
print("📥 LAS読み込み中...")
las = laspy.read(las_path)
points = np.vstack([las.x, las.y, las.z]).T
print(f"✅ 総点数: {len(points):,}")

# === ランダムサンプリング ===
np.random.seed(42)  # 再現性のため固定
indices = np.random.choice(len(points), size=sample_size, replace=False)
sample_points = points[indices]

# === KDTree構築（全点群に対して）
print("🌲 KDTree構築中...")
tree = cKDTree(points)

# === 最近傍距離の計算（2番目が自分以外の最近傍）
print("📏 最近傍距離計算中...")
distances, _ = tree.query(sample_points, k=2)
nearest_dist = distances[:, 1]

# === 指定半径内の点数カウント（自分含む）
print(f"🔍 半径 {radius}m 以内の近傍点数カウント中...")
counts = tree.query_ball_point(sample_points, r=radius)
num_neighbors = np.array([len(c) for c in counts])

# === 統計表示 ===
print("\n📊【最近傍距離（自分を除いた最も近い点との距離）】")
print(f"  平均    : {nearest_dist.mean():.4f} m")
print(f"  中央値  : {np.median(nearest_dist):.4f} m")
print(f"  最小値  : {nearest_dist.min():.4f} m")
print(f"  最大値  : {nearest_dist.max():.4f} m")

print(f"\n📊【半径 {radius}m 以内の近傍点数（自分含む）】")
print(f"  平均    : {num_neighbors.mean():.1f} 点")
print(f"  中央値  : {np.median(num_neighbors):.1f} 点")
print(f"  最小値  : {num_neighbors.min()} 点")
print(f"  最大値  : {num_neighbors.max()} 点")
