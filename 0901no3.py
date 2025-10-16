# -*- coding: utf-8 -*-
"""
M0（3D連結制約付き）＋最大長方形縁点＋classification（スライス番号）付きLAS出力
"""

import os
import math
import numpy as np
import laspy
import cv2
from scipy.ndimage import label

# === パラメータ ===
INPUT_LAS = "/data/0828_01_500_suidoubasi_ue.las"
OUTPUT_LAS = "/output/0901no3_M0_connected_rect_edges.las"
BIN_X = 2.0
MIN_PTS_PER_XBIN = 50
GAP_DIST = 50.0
SECTION_INTERVAL = 0.5
LINE_LENGTH = 60.0
SLICE_THICKNESS = 0.20
Z_LIMIT = 1.9
MIN_RECT_SIDE = 5

# === 関数 ===
def l2(p, q):
    return math.hypot(q[0] - p[0], q[1] - p[1])

def make_slices(XYZ):
    centers, slices = [], []
    for x in np.arange(XYZ[:, 0].min(), XYZ[:, 0].max(), SECTION_INTERVAL):
        pts = XYZ[(XYZ[:, 0] >= x - BIN_X / 2) & (XYZ[:, 0] <= x + BIN_X / 2) & (XYZ[:, 2] <= Z_LIMIT)]
        if len(pts) < MIN_PTS_PER_XBIN:
            centers.append(None)
            slices.append(None)
            continue
        cx, y_mean = x, np.mean(pts[:, 1])
        c = np.array([cx, y_mean, 0])
        n_hat = np.array([1, 0, 0])
        mask = np.abs(XYZ @ n_hat - c @ n_hat) <= SLICE_THICKNESS / 2
        slices.append(XYZ[mask])
        centers.append(c)
    return centers, slices

def project_to_vz(points, c, n_hat):
    bin_v = int(np.ceil(LINE_LENGTH / 0.05))
    bin_z = int(np.ceil((Z_LIMIT + 2.0) / 0.05))
    grid = np.zeros((bin_z, bin_v), dtype=np.uint8)
    for p in points:
        r = p - c
        v = int(np.floor((r[1] + LINE_LENGTH / 2) / 0.05))
        z = int(np.floor((p[2] + 2.0) / 0.05))
        if 0 <= v < bin_v and 0 <= z < bin_z:
            grid[z, v] = 1
    return grid

def downfill(free):
    filled = np.copy(free)
    for v in range(free.shape[1]):
        col = free[:, v]
        ones = np.where(col)[0]
        if len(ones) < 2:
            continue
        filled[ones[0]:ones[-1] + 1, v] = True
    return filled

def rectangles_on_slice(free_bitmap):
    h, w = free_bitmap.shape
    max_area, best = 0, []
    for top in range(h):
        for left in range(w):
            if not free_bitmap[top, left]:
                continue
            for bottom in range(top + MIN_RECT_SIDE, h):
                for right in range(left + MIN_RECT_SIDE, w):
                    if np.all(free_bitmap[top:bottom, left:right]):
                        area = (bottom - top) * (right - left)
                        if area > max_area:
                            max_area = area
                            best = [(top, left, bottom, right)]
    edges = []
    for top, left, bottom, right in best:
        for v in range(left, right):
            edges.append((v, top))
            edges.append((v, bottom - 1))
        for z in range(top, bottom):
            edges.append((left, z))
            edges.append((right - 1, z))
    return edges

def vz_to_world_on_slice(vz, c, n_hat):
    _, v = vz
    z = vz[1] * 0.05 - 2.0
    y = v * 0.05 - LINE_LENGTH / 2
    x = c[0]
    return np.array([x, y, z])

def write_green_las_with_classification(points_with_class, output_las):
    xyz = np.array([p for p, _ in points_with_class])
    classification = np.array([cls for _, cls in points_with_class], dtype=np.uint8)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = xyz.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.red = np.zeros(len(xyz), dtype=np.uint16)
    las.green = np.full(len(xyz), 65535, dtype=np.uint16)
    las.blue = np.zeros(len(xyz), dtype=np.uint16)
    las.classification = classification
    las.write(output_las)

# === メイン処理 ===
las = laspy.read(INPUT_LAS)
XYZ = np.vstack([las.x, las.y, las.z]).T
centers, slices = make_slices(XYZ)
bitmap_stack = []
for c, pts in zip(centers, slices):
    if c is None or pts is None:
        bitmap_stack.append(None)
        continue
    raw = project_to_vz(pts, c, None)
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    free = np.logical_and(~closed.astype(bool), raw.astype(bool))
    filled = downfill(free)
    bitmap_stack.append(filled)

stack = np.stack([b if b is not None else np.zeros_like(bitmap_stack[0]) for b in bitmap_stack])
labels, num = label(stack)
counts = np.bincount(labels.flatten())
max_label = np.argmax(counts[1:]) + 1

GREEN = []
for u, (c, bmap) in enumerate(zip(centers, bitmap_stack)):
    if bmap is None or np.all(labels[u] != max_label):
        continue
    edges = rectangles_on_slice(bmap)
    for e in edges:
        pt = vz_to_world_on_slice(e, c, None)
        GREEN.append((pt, u))

write_green_las_with_classification(GREEN, OUTPUT_LAS)
