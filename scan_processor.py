import numpy as np
from ouster.client import ChanField

def lidar_scan_to_distance_vector(scan):
    ranges = scan.field(ChanField.RANGE)
    ranges = np.array(ranges)
    distances = np.mean(ranges, axis=0)
    angles = np.linspace(-np.pi, np.pi, len(distances))
    return angles, distances
