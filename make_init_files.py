import os

folders = ['lidar', 'map', 'matching']

for folder in folders:
    init_path = os.path.join(folder, '__init__.py')
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"[作成] フォルダ: {folder}")
    with open(init_path, 'w') as f:
        pass
    print(f"[作成] {init_path}")
