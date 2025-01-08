import torch
from torch.nn.functional import pdist
import numpy as np


# Efficient Voxel Based Sampling from DEPOCO
# https://github.com/PRBonn/deep-point-map-compression/blob/b2e35bb05e70ae28b159c2c602bc187414173c06/depoco/architectures/network_blocks.py#L74
def gridSampling(pcd: torch.tensor, resolution=0.02):

    grid = torch.floor(pcd/resolution)
    center = (grid+0.5)*resolution
    dist = ((pcd-center)**2).sum(dim=1)
    dist = dist/dist.max()*0.7

    max_v_count = max([
        torch.max(pcd[:,0]) - torch.min(pcd[:,0]),
        torch.max(pcd[:,1]) - torch.min(pcd[:,1]),
        torch.max(pcd[:,2]) - torch.min(pcd[:,2])
    ])
    v_size = torch.ceil(max_v_count/resolution)
    grid_idx = grid[:, 0] + grid[:, 1] * \
        v_size + grid[:, 2] * v_size * v_size
    grid_d = grid_idx+dist
    idx_orig = torch.argsort(grid_d)

    # trick from https://github.com/rusty1s/pytorch_unique
    unique, inverse, counts = torch.unique_consecutive(
        grid_idx[idx_orig], return_inverse=True, return_counts=True)
    perm = torch.arange(inverse.size(
        0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])

    """
    HACK: workaround to get the first item. scatter overwrites indices on gpu not sequentially
           -> you get random points in the voxel not the first one
    """ 
    p= perm.cpu()
    i=inverse.cpu()
    idx = torch.empty(unique.shape,dtype=p.dtype).scatter_(0, i, p)
    return idx_orig[idx].tolist()