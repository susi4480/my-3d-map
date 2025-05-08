import os
import laspy
import numpy as np
import open3d as o3d

# フォルダパス（すべての.lasファイル対象）
folder_path = r"C:\Users\user\Documents\lab\data\las2"

# Z値の異常除去フィルタ（極端な値のみ除外）
z_min, z_max = -1e6, 1e6  # 実際の川底～ビル上くらいまでの幅に設定

all_points = []

# フォルダ内のすべての.lasファイルを処理
for fname in os.listdir(folder_path):
    if fname.endswith(".las"):
        try:
            fpath = os.path.join(folder_path, fname)
            las = laspy.read(fpath)
            pts = np.vstack((las.x, las.y, las.z)).T
            pts = pts[(pts[:,2] > z_min) & (pts[:,2] < z_max)]
            all_points.append(pts)
            print(f"✅ {fname} 読み込み完了（{len(pts)}点）")
        except Exception as e:
            print(f"⚠️ {fname} の読み込みに失敗しました：{e}")

# 点群の統合
if all_points:
    merged = np.vstack(all_points)

    # Open3Dで可視化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged)
    pcd.paint_uniform_color([0.4, 0.6, 1.0])  # 水色系

    print(f"\n🌍 統合点数：{len(merged)} 点を可視化中...")
    o3d.visualization.draw_geometries([pcd])
else:
    print("❌ 表示できる点群が見つかりませんでした。")
