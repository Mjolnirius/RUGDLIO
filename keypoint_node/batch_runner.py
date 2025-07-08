#!/usr/bin/env python3

import os
import glob
import torch
import open3d as o3d
import numpy as np
from ruamel import yaml

from model.bimodal_compressor import BimodalCompressor
from model.handcrafted_feature_extractor import compute_smoothness
from model.utils import gridSampling


def load_pcd_as_tensor(filepath):
    pcd = o3d.t.io.read_point_cloud(filepath)
    points = pcd.point.positions.numpy()
    rings = pcd.point.rings.numpy() if 'rings' in pcd.point else np.zeros((points.shape[0], 1))
    return torch.from_numpy(points).float().cuda(), torch.from_numpy(rings).float().cuda()

def save_tensor_as_pcd(points_tensor, filepath):
    pcd = o3d.geometry.PointCloud()
    points = points_tensor.cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filepath, pcd)

def process_point_cloud(model, points, rings, mode="bimodal", lidar_range=50.0):
    with torch.no_grad():
        all_indices = torch.arange(len(points), device=points.device)

        if mode == 'handcrafted':
            smoothness = compute_smoothness(points, rings, k=10)
            rough_indices = gridSampling(points, resolution=0.25)
            rough_num = len(rough_indices)

            edge_indices = torch.topk(smoothness, k=int(0.05*rough_num), largest=True).indices.flatten()
            planar_indices = ~torch.isin(all_indices, edge_indices)
            planar_indices = all_indices[planar_indices].flatten()
            indices = torch.randperm(len(planar_indices))[:int(0.15*rough_num)]
            planar_indices = planar_indices[indices]

            direct_kp_indices = torch.cat([edge_indices, planar_indices])
        else:
            match_score, gicp_score, _ = model(points, None)

            if mode == 'gicp_only':
                direct_kp_indices = torch.topk(gicp_score, k=int(0.2*len(gicp_score)), largest=False).indices.flatten()
            elif mode == 'match_only':
                direct_kp_indices = torch.topk(match_score, k=int(0.2*len(match_score)), largest=True).indices.flatten()
            else:
                gicp_indices = torch.topk(gicp_score, k=int(0.1*len(gicp_score)), largest=False).indices.flatten()
                match_score[gicp_indices] = torch.min(match_score)
                match_indices = torch.topk(match_score, k=int(0.1*len(match_score)), largest=True).indices.flatten()
                direct_kp_indices = torch.cat([match_indices, gicp_indices])

                point_norms = torch.norm(points, dim=1)
                coverage_indices = point_norms >= 0.90 * lidar_range
                coverage_indices = coverage_indices.nonzero().flatten()

                if len(coverage_indices) >= 0.01 * points.shape[0]:
                    direct_kp_indices = torch.cat([direct_kp_indices, coverage_indices])

        mask = torch.ones(len(points), dtype=torch.bool, device=points.device)
        mask[direct_kp_indices] = False
        to_be_compressed = points[mask]
        to_be_compressed_indices = all_indices[mask]

        sparse_indices = gridSampling(to_be_compressed, resolution=3.5)

        all_indices = torch.hstack([
            direct_kp_indices,
            to_be_compressed_indices[sparse_indices]
        ])

        return points[all_indices], points[mask]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to directory with input PCD files")
    parser.add_argument("--output_dir", required=True, help="Where to save processed clouds")
    parser.add_argument("--cfg", default="src/FeatureLIOM/config/bimodal_NCL_Pretrained_Match.yaml") # Which model checkpoint to load (weights from training)
    parser.add_argument("--mode", default="bimodal")
    parser.add_argument("--use_model", action="store_true")
    parser.add_argument("--lidar_range", type=float, default=128.0)
    args = parser.parse_args()

    print("### BatchRunner gestartet ###")
    print("Input-Verzeichnis:", args.input_dir)
    print("Output-Verzeichnis:", args.output_dir)
    print("Modus:", args.mode)
    print("Use model:", args.use_model)

    os.makedirs(args.output_dir, exist_ok=True)

    y = yaml.YAML(typ='safe', pure=True)
    bimodal_config = y.load(open(args.cfg, 'r'))
    ckpt_path = "./src/FeatureLIOM/" + bimodal_config['ckpt_dir'] + '/' + bimodal_config['experiment_name'] + "_best.pth"

    model = BimodalCompressor(bimodal_config).cuda().eval()
    if args.use_model and os.path.isfile(ckpt_path):
        print(f"Loaded model from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))

    files = sorted(glob.glob(os.path.join(args.input_dir, '*.pcd')))
    print(f"{len(files)} PCD-Dateien gefunden.")
    for i, file in enumerate(files):
        print(f"[{i+1}/{len(files)}] Processing {file}...")
        points, rings = load_pcd_as_tensor(file)
        keypoints, dropped = process_point_cloud(model, points, rings, args.mode, args.lidar_range)

        out_key = os.path.join(args.output_dir, os.path.basename(file).replace('.pcd', '_keypoints.pcd'))
        out_drop = os.path.join(args.output_dir, os.path.basename(file).replace('.pcd', '_dropped.pcd'))

        save_tensor_as_pcd(keypoints, out_key)
        save_tensor_as_pcd(dropped, out_drop)

        #print("→ Keypoints gespeichert unter:", out_key)
        #print("→ Dropped gespeichert unter:", out_drop)

    print("### BatchRunner abgeschlossen ###")

if __name__ == "__main__":
    main()
