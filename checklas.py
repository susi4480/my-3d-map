import laspy

# 入力LASファイル
las_path = r"C:\Users\user\Documents\lab\data\fulldata\floor_las\sita\20240627-100310(1)-R20250718-132920.las"

# LASファイルを読み込み
las = laspy.read(las_path)

# 基本情報
print("✅ 点数:", len(las.points))
print("✅ 点フォーマット:", las.header.point_format)
print("✅ 利用可能な属性（dimensions）:")
print(las.point_format.dimension_names)

# 各属性を少しだけ表示
for dim in las.point_format.dimension_names:
    data = getattr(las, dim)
    print(f"--- {dim} ---")
    print(data[:10])  # 先頭10点だけ表示
