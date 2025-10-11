import torch
import torch.nn as nn
import torch.nn.functional as F

# Minkowski Version 
# import MinkowskiEngine as ME
# resnet3D version
from mmcv.cnn.bricks.conv_module import ConvModule

from ..builder import FUSION_LAYERS


@FUSION_LAYERS.register_module()
class AdaptiveFusion(nn.Module):
    def __init__(self, out_channels = 18, img_in_channels = 18, pts_in_channels = 18):
        super().__init__()


        self.img_enc = ConvModule(
            img_in_channels, 
            out_channels, 
            kernel_size = 3,
            stride = 1, 
            padding = 1, 
            bias = True, 
            conv_cfg = dict(type='Conv3d'))
        
        self.pts_enc = ConvModule(
            pts_in_channels, 
            out_channels, 
            kernel_size = 3,
            stride = 1, 
            padding = 1, 
            bias = True, 
            conv_cfg = dict(type='Conv3d'))
        
        self.vis_enc = ConvModule(
            out_channels * 2, 
            1, 
            kernel_size = 3,
            stride = 1, 
            padding = 1, 
            bias = True, 
            conv_cfg = dict(type='Conv3d'))

    def forward(self, img_voxel_feats, pts_voxel_feats):
        # Encode both inputs
        img_voxel_feats = self.img_enc(img_voxel_feats)
        pts_voxel_feats = self.pts_enc(pts_voxel_feats)
        # Concatenate the features along channel dimension
        concatenated_feats = torch.cat([img_voxel_feats, pts_voxel_feats], dim=1)  # Shape: [B, 2*C_out, Z, Y, X]




        # Compute adaptive fusion weights
        vis_weight = self.vis_enc(concatenated_feats) # Shape: [B, 1, Z, Y, X]
        vis_weight = torch.sigmoid(vis_weight) # Shape: [B, 1, Z, Y, X]
        
        img_voxel_feats = vis_weight * img_voxel_feats
        pts_voxel_feats = (1 - vis_weight) * pts_voxel_feats
        
        voxel_feats = img_voxel_feats + pts_voxel_feats
        return voxel_feats
        
        