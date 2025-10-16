# matching/matcher.py
import numpy as np

def match_scan_to_map(scan, virtual_scans):
    scan_angles, scan_distances = scan
    best_score = -np.inf
    best_index = -1
    for idx, (v_angles, v_distances) in enumerate(virtual_scans):
        score = -np.linalg.norm(scan_distances - v_distances)
        if score > best_score:
            best_score = score
            best_index = idx
    return best_index, best_score
