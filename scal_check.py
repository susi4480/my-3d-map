import laspy

# LASファイルを読み込み
las = laspy.read(r"C:\Users\user\Documents\lab\data\las2\20240627-095645(1)-R20250425-123250.las")

# X, Y, Z 座標のサンプル表示
print("X:", las.x[:5])
print("Y:", las.y[:5])
print("Z:", las.z[:5])

# スケールとオフセット（地理座標系に関係する）
print("Scale factors:", las.header.scales)
print("Offsets:", las.header.offsets)

# その他のヘッダー情報
print("Point format:", las.header.point_format)
print("Version:", las.header.version)
print("Total points:", las.header.point_count)
