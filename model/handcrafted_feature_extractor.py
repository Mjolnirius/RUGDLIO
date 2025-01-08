import torch
import numpy as np
from scipy.spatial import cKDTree


def compute_smoothness(points, ring_idx, k=10):

    # Extract percentage% edge points and percentage% planar points

    x = points[:, 0].view(-1, 1)
    y = points[:, 1].view(-1, 1)
    
    # phi = torch.atan2(y, x).view(-1, 1) * (180.0 / torch.pi)

    # spherical_coords = torch.hstack([1e10*ring_idx, phi])
    # use ring * 10 as z coord, so closest neighbors are restricted to the points' respective rings
    new_coords = torch.hstack([x, y, ring_idx*10])

    tree = cKDTree(new_coords.cpu().numpy())
    _, indices = tree.query(new_coords.cpu().numpy(), k+1, workers=8)
    indices = torch.tensor(indices).cuda()

    # Gather neighbor points for each point
    neighbor_points = points[indices]  # Shape: (N, k, D)
    
    # Compute the differences between the point and its neighbors
    point_cloud_expanded = points.unsqueeze(1).expand(-1, k+1, -1)  # Shape: (N, k, D)
    differences = neighbor_points - point_cloud_expanded  # Shape: (N, k, D)

    # Compute the sum of differences
    sum_diff = differences.sum(dim=1)  # Shape: (N, D)

    # Compute the norms
    norm_X = points.norm(dim=1)  # Shape: (N,)
    norm_sum_diff = sum_diff.norm(dim=1)  # Shape: (N,)

    # Compute the curvature
    curvatures = norm_sum_diff / (k+1)

    return curvatures