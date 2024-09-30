import MinkowskiEngine as ME
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import FUSION_LAYERS


@FUSION_LAYERS.register_module()
class SparseFusion(nn.Module):
    def __init__(self, out_channels = 16, img_in_channels = 64, pts_in_channels = 32):
        super().__init__()

        # Minkowski Convolution for image and point cloud features
        self.img_enc = nn.Sequential(
            ME.MinkowskiConvolution(
                img_in_channels, out_channels, kernel_size=3, stride=1, dimension=3, bias=False),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True)
        )

        self.pts_enc = nn.Sequential(
            ME.MinkowskiConvolution(
                pts_in_channels, out_channels, kernel_size=3, stride=1, dimension=3, bias=False),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True)
        )

        self.vis_enc = nn.Sequential(
            ME.MinkowskiConvolution(
                out_channels, out_channels, kernel_size=3, stride=1, dimension=3, bias=False),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, stride=1, dimension=3, bias=False),
            ME.MinkowskiSigmoid()
        )

    def forward(self, img_voxel_feats: ME.SparseTensor, pts_voxel_feats: ME.SparseTensor):
        # Encode both inputs
        img_voxel_feats = self.img_enc(img_voxel_feats)
        pts_voxel_feats = self.pts_enc(pts_voxel_feats)
        
        print(pts_voxel_feats.shape)
        ppp


        # Concatenate the features along channel dimension
        concatenated_feats = ME.cat(img_voxel_feats, pts_voxel_feats)
        # concatenated_feats = img_voxel_feats + pts_voxel_feats
        
        print(concatenated_feats.shape)
        ppp


        # Compute adaptive fusion weights
        vis_weight = self.vis_enc(concatenated_feats)

        # Fuse the features with adaptive fusion weights
        voxel_feats = vis_weight * img_voxel_feats + (1 - vis_weight) * pts_voxel_feats

        return voxel_feats
