import torch
import torch.nn as nn
import numpy as np

import octree_handler

from model.kpconv import KPConv


class BimodalCompressor(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.config = config

        self.input_feature_dim = config['input_feature_dim']
        self.output_feature_dim = config['output_feature_dim']
        self.postprocess_hidden_dim = config['postprocessing_hidden_dim']
        self.processed_feature_dim = config['processed_feature_dim']
        self.use_position_embedding = config['use_position_embedding'] if 'use_position_embedding' in config else False

        self.k = config['knn']

        # intialize octree
        self.octree = octree_handler.Octree()

        # initialize sampling resolution
        self.resolution = config['resolution']

        # preactivate to be applied on point covariance matrices
        self.preactivation = nn.Sequential(
            nn.Linear(
                in_features  = self.input_feature_dim,
                out_features = self.output_feature_dim
            ),
            nn.BatchNorm1d(num_features = self.output_feature_dim),
            nn.LeakyReLU()
        )

        # initializing KPConv
        # Kernel radius should be larger than resolution to allow
        # efficient information exchanging between neighboring candidates
        kp_in_dim = self.output_feature_dim if self.input_feature_dim > 1 else self.input_feature_dim
        self.kernel_radius = config['kernel_radius']
        self.num_kernel_points = config['num_kernel_points']
        self.kp_extent = self.kernel_radius / (self.num_kernel_points**(1/3)-1)*1.5
        self.kp_conv = KPConv(
            kernel_size  = self.num_kernel_points,
            p_dim        = 3,
            in_channels  = kp_in_dim,
            out_channels = self.output_feature_dim,
            KP_extent    = self.kp_extent,
            radius       = self.kernel_radius,
            deformable   = config['deformable']
        )
        
        if self.use_position_embedding:
            self.position_embedding_hidden_dim = config['position_embedding_hidden_dim'] if 'position_embedding_hidden_dim' in config else 64
            self.position_embedding_dim = config['position_embedding_dim'] if 'position_embedding_dim' in config else 256
            self.position_embedding = nn.Sequential(
                nn.Linear(
                    in_features = 3, 
                    out_features = self.position_embedding_hidden_dim
                ),
                nn.BatchNorm1d(num_features = self.position_embedding_hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(
                    in_features = self.position_embedding_hidden_dim,
                    out_features = self.position_embedding_dim
                ),
                nn.BatchNorm1d(num_features = self.position_embedding_dim),
                nn.LeakyReLU()
            )

        # feature aggregation layer 
        self.feature_aggregation = nn.Sequential(
            nn.Linear(
                in_features  = self.output_feature_dim,
                out_features = self.postprocess_hidden_dim
            ),
            nn.BatchNorm1d(num_features = self.postprocess_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(
                in_features  = self.postprocess_hidden_dim,
                out_features = self.processed_feature_dim
            ),
            nn.BatchNorm1d(num_features = self.processed_feature_dim),
            nn.LeakyReLU()
        )

        # score prediction layer
        self.keep_score_extraction = nn.Sequential(
            nn.Linear(
                in_features  = self.processed_feature_dim,
                out_features = self.processed_feature_dim // 4
            ),
            nn.BatchNorm1d(num_features = self.processed_feature_dim // 4),
            nn.LeakyReLU(),
            nn.Linear(
                in_features  = self.processed_feature_dim // 4,
                out_features = 1
            )
        )

        if self.use_position_embedding:
            first_input_dim = self.processed_feature_dim + self.position_embedding_dim
        else:
            first_input_dim = self.processed_feature_dim
        self.unique_score_extraction = nn.Sequential(
            nn.Linear(
                in_features  = first_input_dim,
                out_features = first_input_dim // 4
            ),
            nn.BatchNorm1d(num_features = first_input_dim // 4),
            nn.LeakyReLU(),
            nn.Linear(
                in_features  = first_input_dim // 4,
                out_features = 1
            )
        )

        self.sigmoid = nn.Sigmoid()

    
    def forward(self, points, covariances):

        if self.input_feature_dim > 1:

            covariances = covariances.view(-1, 9)

            # Shape: n_s, output_feature_dim
            features = self.preactivation(covariances)
        
        else:

            # Shape: n_s, 1
            features = torch.ones((points.shape[0], 1)).float().cuda()

        # Generate neighbor indices for KPConv
        self.octree.setInput(points.detach().cpu().numpy())
        # Shape: n_s, k
        neighbors_index = self.octree.radiusSearchIndices(
            range(len(points)), self.k, self.kernel_radius
        )
        neighbors_index = torch.from_numpy(neighbors_index).long().to(points.device)

        # Shape: n_s, output_feature_dim
        kpconv_feature = self.kp_conv(
            q_pts       = points,
            s_pts       = points,
            neighb_inds = neighbors_index,
            x           = features 
        )

        # Shape: n_s, processed_feature_dim
        if self.use_position_embedding:
            positional_embedding = self.position_embedding(points)

        aggregated_features = self.feature_aggregation(kpconv_feature)

        # Shape: n_s, 1
        match_score = self.keep_score_extraction(aggregated_features)
        match_score = self.sigmoid(match_score).flatten()

        if self.use_position_embedding:
            unique_score = self.unique_score_extraction(
                torch.hstack([positional_embedding, aggregated_features])
            )
        else:
            unique_score = self.unique_score_extraction(aggregated_features)
        unique_score = self.sigmoid(unique_score).flatten()

        return match_score, unique_score, neighbors_index