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
                img_in_channels, out_channels, kernel_size=5, stride=1, dimension=3, bias=False),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True)
        )

        self.pts_enc = nn.Sequential(
            ME.MinkowskiConvolution(
                pts_in_channels, out_channels, kernel_size=5, stride=1, dimension=3, bias=False),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True)
        )

        self.vis_enc = nn.Sequential(
            ME.MinkowskiConvolution(
                out_channels * 2, out_channels, kernel_size=5, stride=1, dimension=3, bias=False),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, stride=1, dimension=3, bias=False),
            ME.MinkowskiSigmoid()
        )

    def forward(self, img_voxel_feats: ME.SparseTensor, pts_voxel_feats: ME.SparseTensor):
        # Encode both inputs
        img_voxel_feats = self.img_enc(img_voxel_feats)
        pts_voxel_feats = self.pts_enc(pts_voxel_feats)


        # Concatenate the features along channel dimension
        concatenated_feats = ME.cat(img_voxel_feats, pts_voxel_feats)
        # concatenated_feats = img_voxel_feats + pts_voxel_feats
        


        # Compute adaptive fusion weights
        vis_weight = self.vis_enc(concatenated_feats)

        # vis weight is a scalar value for each voxel in sparse tensor format
    

        # Fuse the features with adaptive fusion weights
        img_voxel_feats = vis_weight * img_voxel_feats
        
        # create sparse tensor of 1s with same coordinates as vis_weight
        one_sparse_tensor = ME.SparseTensor(
            features=torch.ones_like(vis_weight.F),
            coordinate_map_key=vis_weight.coordinate_map_key,
            coordinate_manager=vis_weight.coordinate_manager,
            device=vis_weight.device
        )
        
        reverse_vis_weight = one_sparse_tensor - vis_weight
        pts_voxel_feats = reverse_vis_weight * pts_voxel_feats
        # voxel_feats = vis_weight * img_voxel_feats + (1 - vis_weight) * pts_voxel_feats
        
        voxel_feats = img_voxel_feats + pts_voxel_feats

        return voxel_feats
