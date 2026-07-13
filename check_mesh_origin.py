#!/usr/bin/env python3
"""
Diagnostic script: checks whether each STL's geometry sits near local origin (0,0,0)
or near its real assembly-position coordinates.

Usage:
    pip install numpy-stl --break-system-packages
    python3 check_mesh_origins.py /path/to/meshes/

Run this on Ubuntu, in the folder containing your .stl files.
"""

import sys
import os
from stl import mesh
import numpy as np

def check_folder(folder):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.stl')])
    if not files:
        print(f"No STL files found in {folder}")
        return

    print(f"{'FILENAME':<55} {'CENTER (mm)':<35} {'NEAR ZERO?':<12}")
    print("-" * 105)

    for fname in files:
        path = os.path.join(folder, fname)
        try:
            m = mesh.Mesh.from_file(path)
            # get bounding box center
            all_points = m.points.reshape(-1, 3)
            mins = all_points.min(axis=0)
            maxs = all_points.max(axis=0)
            center = (mins + maxs) / 2.0
            dist_from_zero = np.linalg.norm(center)

            near_zero = "YES (offset=0)" if dist_from_zero < 20 else "NO (needs offset)"
            print(f"{fname:<55} [{center[0]:8.2f}, {center[1]:8.2f}, {center[2]:8.2f}]   {near_zero}")
        except Exception as e:
            print(f"{fname:<55} ERROR: {e}")

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    check_folder(folder)