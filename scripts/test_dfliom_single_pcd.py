#!/usr/bin/env python3

import sys
from pathlib import Path

# FeatureLIOM Projekt-Root hinzufügen
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Jetzt funktionieren Importe wie im Projektcode:
from keypoint_node.dfliom_single_pcd import dfliom_single_pcd



import os
import open3d as o3d
from pathlib import Path
#from dfliom_single_pcd import dfliom_single_pcd

# Pfad zur Input-Datei
INPUT_PCD = Path("/home/thor_unix_2204/dfliom_ws/src/FeatureLIOM/input_data/dataset_quadhard_full/quadhard_seq_50_79/scan_0054.pcd")
assert INPUT_PCD.is_file(), f"❌ Input file not found: {INPUT_PCD}"

# Output-Ordner definieren
parent_dir = INPUT_PCD.parent.parent  # z.B. .../dataset_quadhard_full
orig_folder_name = INPUT_PCD.parent.name  # z.B. quadhard_seq_50_79
output_folder = parent_dir / f"{orig_folder_name}_dfliom_single_v3"
output_folder.mkdir(exist_ok=True)

# Output-Dateiname
OUTPUT_PCD = output_folder / INPUT_PCD.name.replace(".pcd", "_reduced.pcd")

def main():
    print(f"📥 Loading input PCD: {INPUT_PCD}")
    reduced_pcd = dfliom_single_pcd(str(INPUT_PCD))

    if isinstance(reduced_pcd, o3d.geometry.PointCloud):
        print(f"📤 Saving reduced PCD to: {OUTPUT_PCD}")
        o3d.io.write_point_cloud(str(OUTPUT_PCD), reduced_pcd)
        print("✅ Done.")
    else:
        print("❌ dfliom_single_pcd did not return a valid Open3D PointCloud object!")

if __name__ == "__main__":
    main()
