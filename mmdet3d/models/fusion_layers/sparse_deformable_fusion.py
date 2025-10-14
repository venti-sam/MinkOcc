# mmdet3d/models/fusion_layers/sparse_deformable_fusion.py

import torch
from torch import nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, force_fp32
from mmdet3d.models.builder import FUSION_LAYERS
from mmdet.models.utils.builder import build_attention
from mmdet3d.core.bbox.structures import points_cam2img

@FUSION_LAYERS.register_module()
class SparseDeformableFusion(BaseModule):
    """
    Fusion module for MinkOccV4 that uses deformable attention to fuse
    sparse LiDAR voxel features with dense, multi-level image features.

    This module iterates through each sample in a batch to handle the variable
    number of voxels, generates reference points by projecting voxel centers
    onto camera views, and applies spatial cross-attention.
    """
    def __init__(self,
                 embed_dims,
                 num_cams,
                 pc_range,
                 voxel_size,
                 deformable_attention,
                 dropout_p=0.1,
                 init_cfg=None):
        super(SparseDeformableFusion, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_cams = num_cams

        # Store pc_range and voxel_size as buffers
        self.register_buffer('pc_range', torch.tensor(pc_range))
        self.register_buffer('voxel_size', torch.tensor(voxel_size))

        # Build the core attention module from config
        self.attention = build_attention(deformable_attention)

        # Output projection layer after attention
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.norm = nn.LayerNorm(embed_dims)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_p)
        
    @force_fp32(apply_to=('occ_feats', 'img_feats'))
    def forward(self, occ_feats, occ_coords, img_feats, img_metas):
        """
        Forward pass for SparseDeformableFusion.

        Args:
            occ_feats (torch.Tensor): Sparse voxel features from the LiDAR branch.
                Shape: (num_total_voxels, C).
            occ_coords (torch.Tensor): Coordinates for the sparse features.
                Shape: (num_total_voxels, 4) -> [batch_idx, z, y, x].
            img_feats (list[torch.Tensor]): List of multi-level dense image
                features from the camera branch.
            img_metas (list[dict]): Meta information for each sample.

        Returns:
            torch.Tensor: Fused sparse features. Shape: (num_total_voxels, C).
        """
        # --- 1. Prepare Image Features (Value for Attention) ---
        bs = occ_coords[:, 0].max().item() + 1
        
        # Get spatial shapes and level start index for multi-level features
        spatial_shapes = []
        for feat in img_feats:
            # Shape is (B*N_cams, C, H, W)
            spatial_shapes.append(feat.shape[-2:])
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=occ_feats.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # Flatten and process image features to create the 'value' tensor
        img_feats_reshaped = []
        for feat in img_feats:
            _, _, H, W = feat.shape
            # Reshape from (B*N, C, H, W) -> (B, N, C, H*W) -> (B, N, H*W, C)
            feat = feat.reshape(bs, self.num_cams, self.embed_dims, H, W)
            feat = feat.permute(0, 1, 3, 4, 2).reshape(bs, self.num_cams, H * W, self.embed_dims)
            img_feats_reshaped.append(feat)
        
        # Concatenate across levels: (B, N_cams, total_pixels, C)
        value = torch.cat(img_feats_reshaped, dim=2)
        
        # --- 2. Iterate Through Batch to Process Each Sample ---
        fused_features_list = []
        for i in range(bs):
            # Isolate data for the current sample
            sample_mask = (occ_coords[:, 0] == i)
            occ_feats_i = occ_feats[sample_mask]      # Query for this sample
            occ_coords_i = occ_coords[sample_mask][:, [3, 2, 1]] # Voxel coords [x, y, z]
            img_metas_i = img_metas[i]

            # --- 3. Generate Reference Points for this Sample ---
            # a. Convert integer voxel coordinates to real-world 3D coordinates
            xyz_coords = occ_coords_i * self.voxel_size + self.pc_range[:3]

            # b. Project 3D points to each camera view
            reference_points_cam_list = []
            bev_mask_list = []
            
            for cam_idx in range(self.num_cams):
                lidar2img = xyz_coords.new_tensor(img_metas_i['lidar2img'][cam_idx])
                
                # Project points and get depth
                uvd_coords = points_cam2img(xyz_coords, lidar2img, with_depth=True)
                
                # Create a mask for valid points (within image bounds and in front of camera)
                img_shape = img_metas_i['img_shape'][cam_idx]
                on_img = (uvd_coords[:, 0] >= 0) & (uvd_coords[:, 0] < img_shape[1]) & \
                         (uvd_coords[:, 1] >= 0) & (uvd_coords[:, 1] < img_shape[0])
                valid_mask = on_img & (uvd_coords[:, 2] > 0)
                bev_mask_list.append(valid_mask)
                
                # Normalize valid coordinates to [0, 1] for attention
                uvd_coords[:, 0] /= img_shape[1]
                uvd_coords[:, 1] /= img_shape[0]
                
                reference_points_cam_list.append(uvd_coords)

            # Stack to create tensors for this sample
            reference_points_cam = torch.stack(reference_points_cam_list, dim=1) # (num_voxels, num_cams, 3)
            bev_mask = torch.stack(bev_mask_list, dim=1) # (num_voxels, num_cams)
            
            # --- 4. Call Attention Module for this Sample ---
            # Reshape query for attention: (1, num_voxels, C)
            query = occ_feats_i.unsqueeze(0)
            
            # SpatialCrossAttention expects batched inputs, so we unsqueeze the sample dimension
            fused_feat = self.attention(
                query=query,
                key=value[i].unsqueeze(0),   # Image features for this specific sample
                value=value[i].unsqueeze(0),
                reference_points_cam=reference_points_cam.unsqueeze(0),
                bev_mask=bev_mask.unsqueeze(0),
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )
            
            # --- 5. Residual Connection and Projection ---
            fused_feat = self.output_proj(fused_feat)
            final_feat = self.norm(occ_feats_i + self.dropout(fused_feat.squeeze(0)))
            
            fused_features_list.append(final_feat)

        # --- 6. Concatenate results and return ---
        return torch.cat(fused_features_list, dim=0)