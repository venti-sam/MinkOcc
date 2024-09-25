# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet import BEVStereo4D

import torch
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np

# additional imports
from .. import builder
import MinkowskiEngine as ME



@DETECTORS.register_module()
class BEVStereo4DOCC(BEVStereo4D):

    def __init__(self,
                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 ## add in lidar backbone here
                 lidar_backbone=None,
                 ## add in lidar neck here
                 lidar_neck=None,
                 # add in voxelization method here
                 
                 **kwargs):
        super(BEVStereo4DOCC, self).__init__(**kwargs)
        # add in lidar backbone and necessary inits here
        self.lidar_backbone = builder.build_backbone(lidar_backbone)
        self.lidar_neck = builder.build_neck(lidar_neck)
        ######################################
        
        
        
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
                        self.img_view_transformer.out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))
        self.use_predicter =use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, num_classes),
            )
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transfromation = False

    def loss_single(self,voxel_semantics,mask_camera,preds):
        loss_ = dict()
        voxel_semantics=voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
            loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
            loss_['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics,)
            loss_['loss_occ'] = loss_occ
        return loss_

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)
        # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score=occ_pred.softmax(-1)
        occ_res=occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # init losses dictionary
        losses = dict()

        # process the occupancy grid
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        # print(voxel_semantics.shape) # [batch, 200, 200, 16]
        # remove class 17 for bce supervision in mink-lidarocc, then reshape it to a list[(n,4)]
        # Shape of voxel_semantics_without_free: (4, 200, 200, 16)
        batch_size = voxel_semantics.shape[0]  # 4

        # Initialize an empty list to hold the results for each batch
        coo_list = []

        for b in range(batch_size):
            current_voxel_grid = voxel_semantics[b]
            mask = current_voxel_grid != 17            
            coords = torch.argwhere(mask)
            coo_list.append(coords)
                


        # voxelization of pointcloud
        voxels, num_points, coors = self.voxelize(points)
        # in coors, get the highest coord in column 2 and 3
        # coors shape is (b, z, x, y) -> includes all batches of points
        # Find the maximum value in the 2nd column (x-coordinate)
        # try plotting the first batch coors to see what it looks like

        # voxels shape is (num_voxels, max_points, 5) -> includes all batches of points
        # num_points is the number of points across batches
        # coors is the coordinates of the voxels with batch number included as first column
        
        # using hardsimplevfe from voxel_encoders to get voxel_features
        # basically just averaging the feats from each point 
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
 
        # Create minkowski sparse tensor using coors and voxel_features
        pts_sparse_tensor = ME.SparseTensor(
            features=voxel_features, 
            coordinates=coors, 
            device=voxel_features.device)
        cm = pts_sparse_tensor.coordinate_manager
        target_key, _ = cm.insert_and_map(
                ME.utils.batched_coordinates(coo_list).to(voxel_features.device),
                string_id="target",
        )
        # pass the sparse tensor through the lidar backbone
        pts_feats = self.lidar_backbone(pts_sparse_tensor)

        # pass the pts_feats through the lidar neck
        out_cls, targets, pts_feat = self.lidar_neck(pts_feats, target_key)
        
        #  BCE Loss calculation for scene completion point existence
        bce_loss = self.lidar_neck.get_bce_loss(out_cls, targets)
        losses['loss_bce'] = bce_loss


        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)
        return losses
