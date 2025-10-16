# map/virtual_scan.py
import numpy as np

def generate_virtual_scans_along_path(path_data):
    scans = []
    for pose in path_data:
        x, y, yaw = pose['x'], pose['y'], pose['yaw']
        angles = np.linspace(-np.pi, np.pi, 360)
        distances = 10 * np.ones_like(angles)  # 仮のスキャン（固定距離）
        scans.append((angles, distances))
    return scans
