import laspy
import numpy as np

# 入出力ファイルパス
input_path = "/home/edu1/miyachi/data/pond/MBES_02.las"
output_path = "/home/edu1/miyachi/data/pond/MBES_02_deduped.las"

# LASファイル読み込み
las = laspy.read(input_path)
points = np.vstack((las.x, las.y, las.z)).T
print(f"🔢 元の点数: {len(points):,}")

# 重複削除
_, idx = np.unique(points, axis=0, return_index=True)
print(f"✅ 重複除去後の点数: {len(idx):,}")
print(f"🗑️ 重複していた点数: {len(points) - len(idx):,}")

# 新しいLASデータの作成
new_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
new_las.header = las.header

# 必要な属性を設定（最低限 X, Y, Z）
new_las.x = las.x[idx]
new_las.y = las.y[idx]
new_las.z = las.z[idx]

# 必要に応じて他の属性（intensity, return_numberなど）も追加
for dim in las.point_format.dimension_names:
    if dim not in ('X', 'Y', 'Z'):
        try:
            setattr(new_las, dim, getattr(las, dim)[idx])
        except:
            pass  # データが無ければスキップ

# 書き出し
new_las.write(output_path)
print(f"💾 保存完了: {output_path}")
