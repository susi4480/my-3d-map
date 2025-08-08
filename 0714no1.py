import os

# 対象フォルダ
folder_path = r"C:\Users\user\Documents\lab\outcome\slice_area_navigation_sita"

# フォルダ内のファイル一覧を取得し、ソート（ファイル名順）
files = sorted(os.listdir(folder_path))

# ファイルを順にリネーム（拡張子維持）
for i, filename in enumerate(files, start=1):
    old_path = os.path.join(folder_path, filename)
    if os.path.isfile(old_path):
        ext = os.path.splitext(filename)[1]  # 拡張子（例: .las）
        new_name = f"{i}{ext}"
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"✅ {filename} → {new_name}")
