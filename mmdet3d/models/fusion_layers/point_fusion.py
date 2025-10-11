# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core.bbox.structures import (get_proj_mat_by_coord_type,
                                          points_cam2img)
from ..builder import FUSION_LAYERS
from . import apply_3d_transformation


def point_sample(img_meta,
                 img_features,
                 points,
                 proj_mat,
                 coord_type,
                 img_scale_factor,
                 img_crop_offset,
                 img_flip,
                 img_rotate,  # Pass rotation to point_sample
                 img_pad_shape,
                 img_shape,
                 aligned=True,
                 padding_mode='zeros',
                 align_corners=True):
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta info.
        img_features (torch.Tensor): 1 x C x H x W image features.
        points (torch.Tensor): Nx3 point cloud in LiDAR coordinates.
        proj_mat (torch.Tensor): 4x4 transformation matrix.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (torch.Tensor): Scale factor with shape of
            (w_scale, h_scale).
        img_crop_offset (torch.Tensor): Crop offset used to crop
            image during data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (tuple[int]): int tuple indicates the h & w after
            padding, this is necessary to obtain features in feature map.
        img_shape (tuple[int]): int tuple indicates the h & w before padding
            after scaling, this is necessary for flipping coordinates.
        aligned (bool, optional): Whether use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str, optional): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool, optional): Whether to align corners when
            sampling image features for each point. Defaults to True.

    Returns:
        torch.Tensor: NxC image features sampled by point coordinates.
    """

    points = apply_3d_transformation(
        points, coord_type, img_meta, reverse=True)

    pts_2d = points_cam2img(points, proj_mat)

    img_coors = pts_2d[:, 0:2] * img_scale_factor
    img_coors -= img_crop_offset

    orig_h, orig_w = img_shape

    # Handle flip
    if img_flip:
        img_coors[:, 0] = orig_w - img_coors[:, 0]

    # Handle rotation
    if img_rotate != 0.0:
        # Create a counter-clockwise rotation matrix
        angle_rad = torch.deg2rad(
            torch.tensor(img_rotate, device=img_coors.device)
        )
        cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)
        rot_mat = torch.tensor([[cos_a, -sin_a],
                                [sin_a, cos_a]], device=img_coors.device)
        
        # Center of rotation is the center of the cropped image
        center = torch.tensor([orig_w / 2.0, orig_h / 2.0], device=img_coors.device)
        
        # Apply rotation to the coordinates
        img_coors = torch.matmul(img_coors - center, rot_mat.T) + center

    coor_x, coor_y = torch.split(img_coors, 1, dim=1)

    h, w = img_pad_shape
    coor_y = coor_y / h * 2 - 1
    coor_x = coor_x / w * 2 - 1
    grid = torch.cat([coor_x, coor_y],
                     dim=1).unsqueeze(0).unsqueeze(0)

    mode = 'bilinear' if aligned else 'nearest'
    point_features = F.grid_sample(
        img_features,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)

    return point_features.squeeze().t()


@FUSION_LAYERS.register_module()
class PointFusion(BaseModule):
    """Fuse image features from multi-scale features.

    Args:
        img_channels (list[int] | int): Channels of image features.
            It could be a list if the input is multi-scale image features.
        pts_channels (int): Channels of point features
        mid_channels (int): Channels of middle layers
        out_channels (int): Channels of output fused features
        img_levels (int, optional): Number of image levels. Defaults to 3.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
            Defaults to 'LIDAR'.
        conv_cfg (dict, optional): Dict config of conv layers of middle
            layers. Defaults to None.
        norm_cfg (dict, optional): Dict config of norm layers of middle
            layers. Defaults to None.
        act_cfg (dict, optional): Dict config of activatation layers.
            Defaults to None.
        activate_out (bool, optional): Whether to apply relu activation
            to output features. Defaults to True.
        fuse_out (bool, optional): Whether apply conv layer to the fused
            features. Defaults to False.
        dropout_ratio (int, float, optional): Dropout ratio of image
            features to prevent overfitting. Defaults to 0.
        aligned (bool, optional): Whether apply aligned feature fusion.
            Defaults to True.
        align_corners (bool, optional): Whether to align corner when
            sampling features according to points. Defaults to True.
        padding_mode (str, optional): Mode used to pad the features of
            points that do not have corresponding image features.
            Defaults to 'zeros'.
        lateral_conv (bool, optional): Whether to apply lateral convs
            to image features. Defaults to True.
    """

    def __init__(self,
                 img_channels,
                 pts_channels,
                 mid_channels,
                 out_channels,
                 img_levels=3,
                 coord_type='LIDAR',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None,
                 activate_out=True,
                 fuse_out=False,
                 dropout_ratio=0,
                 aligned=True,
                 align_corners=True,
                 padding_mode='zeros',
                 lateral_conv=True):
        super(PointFusion, self).__init__(init_cfg=init_cfg)
        if isinstance(img_levels, int):
            img_levels = [img_levels]
        if isinstance(img_channels, int):
            img_channels = [img_channels] * len(img_levels)
        assert isinstance(img_levels, list)
        assert isinstance(img_channels, list)
        assert len(img_channels) == len(img_levels)

        self.img_levels = img_levels
        self.coord_type = coord_type
        self.act_cfg = act_cfg
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        self.dropout_ratio = dropout_ratio
        self.img_channels = img_channels
        self.aligned = aligned
        self.align_corners = align_corners
        self.padding_mode = padding_mode

        self.lateral_convs = None
        if lateral_conv:
            self.lateral_convs = nn.ModuleList()
            for i in range(len(img_channels)):
                l_conv = ConvModule(
                    img_channels[i],
                    mid_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                self.lateral_convs.append(l_conv)
            self.img_transform = nn.Sequential(
                nn.Linear(mid_channels * len(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        else:
            self.img_transform = nn.Sequential(
                nn.Linear(sum(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        self.pts_transform = nn.Sequential(
            nn.Linear(pts_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        )

        if self.fuse_out:
            self.fuse_conv = nn.Sequential(
                nn.Linear(mid_channels, out_channels),
                # For pts the BN is initialized differently by default
                # TODO: check whether this is necessary
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=False))

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                dict(type='Xavier', layer='Linear', distribution='uniform')
            ]

    def forward(self, img_feats, pts, pts_feats, img_metas):
        """Forward function.

        Args:
            img_feats (list[torch.Tensor]): Image features.
            pts: [list[torch.Tensor]]: A batch of points with shape N x lidar_dim.
            pts_feats (torch.Tensor): A tensor consist of point features of the
                total batch.
            img_metas (list[dict]): Meta information of images.

        Returns:
            torch.Tensor: Fused features of each point.
        """
        img_pts = self.obtain_mlvl_feats(img_feats, pts, img_metas)
        img_pre_fuse = self.img_transform(img_pts)
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio)
            
        
        pts_pre_fuse = self.pts_transform(pts_feats)
        fuse_out = img_pre_fuse + pts_pre_fuse
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)
        return fuse_out

    # def obtain_mlvl_feats(self, img_feats, pts, img_metas):
    #     """Obtain multi-level features for each point.

    #     Args:
    #         img_feats (list(torch.Tensor)): Multi-scale image features produced
    #             by image backbone in shape (N, C, H, W).
    #         pts (list[torch.Tensor]): Points of each sample.
    #         img_metas (list[dict]): Meta information for each sample.

    #     Returns:
    #         torch.Tensor: Corresponding image features of each point.
    #     """
    #     if self.lateral_convs is not None:
    #         img_ins = [
    #             lateral_conv(img_feats[i])
    #             for i, lateral_conv in zip(self.img_levels, self.lateral_convs)
    #         ]
    #     else:
    #         img_ins = img_feats
    #     img_feats_per_point = []
    #     for each in img_ins:
    #         print(each.shape)
    #     # Sample multi-level features
    #     for i in range(len(img_metas)):
    #         mlvl_img_feats = []
    #         for level in range(len(self.img_levels)):
    #             mlvl_img_feats.append(
    #                 self.sample_single(img_ins[level][i:i + 1], pts[i][:, :3],
    #                                    img_metas[i]))
    #         mlvl_img_feats = torch.cat(mlvl_img_feats, dim=-1)
    #         img_feats_per_point.append(mlvl_img_feats)

    #     img_pts = torch.cat(img_feats_per_point, dim=0)
    #     return img_pts
    def obtain_mlvl_feats(self, img_feats, pts, img_metas):
        """Obtain multi-level features for each point.

        Args:
            img_feats (list[torch.Tensor]): Multi-scale image features produced
                by the image backbone. Each element in the list corresponds to
                one feature level. The shape of each element is (B*N, C, H, W),
                where B is batch size and N is number of views per sample.
            pts (list[torch.Tensor]): A list of point tensors, one for each sample.
            img_metas (list[dict]): Meta information for each sample.

        Returns:
            torch.Tensor: Corresponding image features of each point with shape
            (total_points, sum_of_level_features), where total_points is the sum of
            points across all samples.
        """

        num_samples = len(img_metas)
        # Assuming all feature maps have the same (B*N) factor
        B = num_samples
        # Extract N from the shape of the first feature level
        BxN, C, H, W = img_feats[0].shape
        N = BxN // B

        img_ins = []

        # Apply lateral convs if present
        if self.lateral_convs is not None:
            for i, level_idx in enumerate(self.img_levels):
                # img_feats[level_idx]: (B*N, C, H, W)
                feats = self.lateral_convs[i](img_feats[level_idx])
                # Now reshape to (B, N, C, H, W)
                feats = feats.view(B, N, feats.size(1), feats.size(2), feats.size(3))
                img_ins.append(feats)
        else:
            # If no lateral conv, just reshape
            for i, level_idx in enumerate(self.img_levels):
                feats = img_feats[level_idx]  # (B*N, C, H, W)
                feats = feats.view(B, N, feats.size(1), feats.size(2), feats.size(3))
                img_ins.append(feats)

        img_feats_per_point = []
        # Sample multi-level features        
        for i in range(B):
            mlvl_img_feats = []
            for level in range(len(self.img_levels)):
                sample_img_feats = img_ins[level][i]  # (N, C, H, W), h and w changes due to img backbone 
                view_feats = []
                for v in range(N):
                    single_view_feat = sample_img_feats[v:v+1]  # (1, C, H, W)
                    # Pass the view index v here
                    sampled_feats = self.sample_single(single_view_feat, pts[i][:, :3], img_metas[i], view_idx=v)
                    view_feats.append(sampled_feats) # (points_num, img_features_dimension)

                view_feats = torch.stack(view_feats, dim=0) # (no.cams, points_num, C)
                combined_feats = torch.mean(view_feats, dim=0) # (points_num, img_features_dimension)
                mlvl_img_feats.append(combined_feats) # (no. img levels, points_num, img_features_dimension)
            
            mlvl_img_feats = torch.cat(mlvl_img_feats, dim=-1) # (points_num, sum_of_level_features) = (points, img * no.levels)
            img_feats_per_point.append(mlvl_img_feats) # (batch, points, channels * levels)

        img_pts = torch.cat(img_feats_per_point, dim=0) # (points * batch, channels * levels)
        return img_pts

    def sample_single(self, img_feats, pts, img_meta, view_idx=0):
        """Sample features from single level single-view image feature map.

        Args:
            img_feats (torch.Tensor): Image feature map in shape (1, C, H, W).
            pts (torch.Tensor): Points of a single sample.
            img_meta (dict): Meta information of the single sample.
            view_idx (int): Index of the view.

        Returns:
            torch.Tensor: Single level image features of each point.
        """
        img_scale_factor = (
            pts.new_tensor(img_meta['scale_factor'][0][:2])
            if 'scale_factor' in img_meta.keys() else 1)
        img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
        # Get the rotation for the current view
        img_rotate = img_meta.get('rotate', 0.0)
        img_crop_offset = (
            pts.new_tensor(img_meta['img_crop_offset'])
            if 'img_crop_offset' in img_meta.keys() else 0)
        
        proj_mats = get_proj_mat_by_coord_type(img_meta, self.coord_type)
        if isinstance(proj_mats, list):
            # Select the projection matrix for the current view
            proj_mat = pts.new_tensor(proj_mats[view_idx])
        else:
            proj_mat = pts.new_tensor(proj_mats)
        
        img_pts = point_sample(
            img_meta=img_meta,
            img_features=img_feats,
            points=pts,
            proj_mat=proj_mat,
            coord_type=self.coord_type,
            img_scale_factor=img_scale_factor,
            img_crop_offset=img_crop_offset,
            img_flip=img_flip,
            img_rotate=img_rotate,
            img_pad_shape=img_meta['input_shape'][:2],
            img_shape=img_meta['img_shape'][0][:2],
            aligned=self.aligned,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

        return img_pts
