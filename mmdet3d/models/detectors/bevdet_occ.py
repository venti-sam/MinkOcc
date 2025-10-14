# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet import BEVStereo4D, BEVDepth4D
from .mvx_faster_rcnn import DynamicMVXFasterRCNN

import torch
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models.backbones.resnet import ResNet
from mmcv.ops import Voxelization
from mmcv.runner import force_fp32
from ..builder import build_renderer


from torch import nn
import numpy as np

# additional imports
from .. import builder
import MinkowskiEngine as ME
import torch.nn.functional as F


from plyfile import PlyData, PlyElement

class MinkowskiSoftplus(nn.Module):
    def forward(self, x):
        return ME.SparseTensor(
            nn.functional.softplus(x.F),
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )
@DETECTORS.register_module()
class BEVStereo4DOCC(BEVStereo4D):

    def __init__(self,
                 loss_occ=None,
                 out_dim=32,
                 dataset_type='nuscenes',
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 lidar_backbone = None,
                 lidar_neck = None,
                 fusion = None,
                 loss_bce_weight = None,
                
                 **kwargs):
        super(BEVStereo4DOCC, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.dataset_type = dataset_type
        self.loss_bce_weight = loss_bce_weight
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
            # self.predicter = nn.Sequential(
            #     nn.Linear(self.out_dim, self.out_dim*2),
            #     nn.Softplus(),
            #     nn.Linear(self.out_dim*2, num_classes),
            # )
            self.predicter = nn.Sequential(
                nn.Linear(num_classes, num_classes*2),
                nn.Softplus(),
                nn.Linear(num_classes*2, num_classes),
            )
            # self.predicter = nn.Sequential(
            #     ME.MinkowskiLinear(self.out_dim, self.out_dim*2),
            #     MinkowskiSoftplus(),
            #     ME.MinkowskiLinear(self.out_dim*2, num_classes),
            # )
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transfromation = False
        
        # Create x, y, z coordinate grids
        x = torch.arange(200)
        y = torch.arange(200)
        z = torch.arange(16)
        # Generate a coordinate grid for each dimension (200, 200, 16)
        mesh_x, mesh_y, mesh_z = torch.meshgrid(x, y, z, indexing='ij')
        # Flatten the coordinate grids and stack them to get (200*200*16, 3)
        self.COO_format_coords = torch.stack([mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten()], dim=1)


        self.lidar_backbone = builder.build_backbone(lidar_backbone)    
        self.lidar_neck = builder.build_neck(lidar_neck)
        self.fusion = builder.build_fusion_layer(fusion)
        # self.occ_backbone = builder.build_backbone(occ_backbone)
        # self.occ_neck = builder.build_neck(occ_neck)

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
        # mainpulate gt to fit supervision for minkowski engine
        voxel_semantics = kwargs['voxel_semantics'] # (b, 200, 200, 16)
        mask_camera = kwargs['mask_camera']
        if self.dataset_type == 'nuscenes':
            assert voxel_semantics[0].min() >= 0 and voxel_semantics[0].max() <= 17
        if self.dataset_type == 'waymo':
            assert voxel_semantics[0].min() >= 0 and voxel_semantics[0].max() <= 15#
        coo_list_gt = [] 
        semantics_list_gt = []
        for b in range(voxel_semantics[0].shape[0]):
            current_voxel_grid = voxel_semantics[b][0]
            if self.dataset_type == 'nuscenes':
                mask = current_voxel_grid != 17
            if self.dataset_type == 'waymo':
                mask = current_voxel_grid != 15
            coords = torch.argwhere(mask) # (dense_points,3)
            coo_list_gt.append(coords)    
            all_feats = current_voxel_grid.view(-1, 1)  # (200*200*16, 1)
            all_coords = self.COO_format_coords.to(current_voxel_grid.device)
            all_coords_and_feats = torch.cat([all_coords, all_feats], dim=1)
            semantics_list_gt.append(all_coords_and_feats) # (200*200*16, 4)

        voxels, num_points, coors = self.voxelize(points)
        if self.dataset_type == 'nuscenes':
            coors[:, 3] = 200 - coors[:, 3]  # Reverse the y direction
            coors = coors[:, [0, 2, 3, 1]] # move b,z,x,y,to b, x, y,z
        if self.dataset_type == 'waymo':
            coors = coors[:, [0, 3, 2, 1]]
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)

        # sparse to dense lidar semantic segmentation
        pts_sparse_tensor = ME.SparseTensor(
            features = voxel_features, 
            coordinates = coors,
            device = voxel_features.device,
        )
        cm = pts_sparse_tensor.coordinate_manager
        target_key, _ = cm.insert_and_map(
            ME.utils.batched_coordinates(coo_list_gt).to(voxel_features.device),
            string_id = "target",
        )
        pts_feats = self.lidar_backbone(pts_sparse_tensor)
        _, _, pts_feat = self.lidar_neck(pts_feats, target_key)
        
        
        # convert pts_feat from list[SparseTensor] to [b, 200, 200, 16] grid format
        pts_coord, pts_feat = pts_feat.decomposed_coordinates_and_features
        batch_grids = []
        for each_coord, each_feat in zip(pts_coord, pts_feat):
            grid = torch.zeros((pts_feat[0].shape[1], 16, 200, 200), dtype=each_feat.dtype, device=each_feat.device)
            x = each_coord[:, 0].long()
            y = each_coord[:, 1].long()
            z = each_coord[:, 2].long()
            mask = (x < 200) & (y < 200) & (z < 16)
            x, y, z = x[mask], y[mask], z[mask]
            each_feat = each_feat[mask]  # Filter features using the mask
            grid[:, z, y, x] = each_feat.t()
            batch_grids.append(grid)
        # Stack the grids along the batch dimension to get [batch, channels, z, y, x]
        occ_pred_lidar = torch.stack(batch_grids, dim=0)
        
        
        
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        occ_pred_cam = self.final_conv(img_feats)# bncdhw->bnwhdc (bczyx -> bxyzc)
        occ_pred = self.fusion(occ_pred_cam, occ_pred_lidar).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc (bczyx -> bxyzc)

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
        losses = dict()
        
        # mainpulate gt to fit supervision for minkowski engine
        voxel_semantics = kwargs['voxel_semantics'] # (b, 200, 200, 16)
        mask_camera = kwargs['mask_camera']
        if self.dataset_type == 'nuscenes':
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        if self.dataset_type == 'waymo':
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 15#
        coo_list_gt = [] 
        semantics_list_gt = []
        for b in range(voxel_semantics.shape[0]):
            current_voxel_grid = voxel_semantics[b]
            if self.dataset_type == 'nuscenes':
                mask = current_voxel_grid != 17
            if self.dataset_type == 'waymo':
                mask = current_voxel_grid != 15
            coords = torch.argwhere(mask) # (dense_points,3)
            coo_list_gt.append(coords)    
            all_feats = current_voxel_grid.view(-1, 1)  # (200*200*16, 1)
            all_coords = self.COO_format_coords.to(current_voxel_grid.device)
            all_coords_and_feats = torch.cat([all_coords, all_feats], dim=1)
            semantics_list_gt.append(all_coords_and_feats) # (200*200*16, 4)

        # voxelization of pointcloud
        voxels, num_points, coors = self.voxelize(points)
        if self.dataset_type == 'nuscenes':
            coors[:, 3] = 200 - coors[:, 3]  # Reverse the y direction
            coors = coors[:, [0, 2, 3, 1]] # move b,z,x,y,to b, x, y,z
        if self.dataset_type == 'waymo':
            coors = coors[:, [0, 3, 2, 1]]
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        
        
        # sparse to dense lidar semantic segmentation
        pts_sparse_tensor = ME.SparseTensor(
            features = voxel_features, 
            coordinates = coors,
            device = voxel_features.device,
        )
        cm = pts_sparse_tensor.coordinate_manager
        target_key, _ = cm.insert_and_map(
            ME.utils.batched_coordinates(coo_list_gt).to(voxel_features.device),
            string_id = "target",
        )
        
        pts_feats = self.lidar_backbone(pts_sparse_tensor)
        out_cls, targets, pts_feat = self.lidar_neck(pts_feats, target_key)

        # bce loss calculation for scene completion point existence
        bce_loss = self.lidar_neck.get_bce_loss(out_cls, targets)   
        losses['loss_bce'] = bce_loss * self.loss_bce_weight
        
        # ce loss calculatiuon for lidar semantic segmentation 
        # ce_loss = self.lidar_neck.get_ce_loss(pts_feat, semantics_list_gt)
        # losses['loss_ce'] = ce_loss
        
        # convert pts_feat from list[SparseTensor] to [b, 200, 200, 16] grid format
        pts_coord, pts_feat = pts_feat.decomposed_coordinates_and_features
        batch_grids = []
        for each_coord, each_feat in zip(pts_coord, pts_feat):
            grid = torch.zeros((pts_feat[0].shape[1], 16, 200, 200), dtype=each_feat.dtype, device=each_feat.device)
            x = each_coord[:, 0].long()
            y = each_coord[:, 1].long()
            z = each_coord[:, 2].long()
            mask = (x < 200) & (y < 200) & (z < 16)
            x, y, z = x[mask], y[mask], z[mask]
            each_feat = each_feat[mask]  # Filter features using the mask
            grid[:, z, y, x] = each_feat.t()
            batch_grids.append(grid)
        # Stack the grids along the batch dimension to get [batch, channels, z, y, x]
        occ_pred_lidar = torch.stack(batch_grids, dim=0)

        # camera lss 3d resnet semantic segmentation
        img_feats, _, depth = self.extract_feat(
        points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth
        occ_pred_cam = self.final_conv(img_feats) # (bczyx)

        # fuse sparse lidar with dense camera features, output should be dense grid (b, 200, 200, 16)
        occ_pred = self.fusion(occ_pred_cam, occ_pred_lidar).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc (bczyx -> bxyzc)

        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
            

        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)
        return losses
    
@DETECTORS.register_module()
class BEVStereo4DOCC_MinkOcc(BEVStereo4D):

    def __init__(self,
                 out_dim=18,
                 dataset_type='nuscenes',
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 lidar_backbone = None,
                 lidar_neck = None, 
                 loss_ce_weight = None,
                 loss_bce_weight = None,       
                 **kwargs):
        
        super(BEVStereo4DOCC_MinkOcc, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.dataset_type = dataset_type
        out_channels = out_dim if use_predicter else num_classes
        self.use_predicter =use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                ME.MinkowskiLinear(self.out_dim, self.out_dim*2),
                MinkowskiSoftplus(),
                ME.MinkowskiLinear(self.out_dim*2, num_classes),
            )
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.class_wise = class_wise
        self.align_after_view_transfromation = False
        self.loss_ce_weight = loss_ce_weight
        self.loss_bce_weight = loss_bce_weight
        
         # Create x, y, z coordinate grids
        x = torch.arange(200)
        y = torch.arange(200)
        z = torch.arange(16)
        # Generate a coordinate grid for each dimension (200, 200, 16)
        mesh_x, mesh_y, mesh_z = torch.meshgrid(x, y, z, indexing='ij')
        # Flatten the coordinate grids and stack them to get (200*200*16, 3)
        self.COO_format_coords = torch.stack([mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten()], dim=1)

        
        
        # build lidar backbone and neck
        self.lidar_backbone = builder.build_backbone(lidar_backbone)    
        self.lidar_neck = builder.build_neck(lidar_neck)
        
        
        # global self count for debugging
        self.global_count = 0
    
    def simple_test(self, 
                    points=None,
                    img_metas=None,
                    img_inputs=None,
                    **kwargs):
        
        # python tools/test.py configs/bevdet_occ/minkocc.py minkocc_epoch_3.pth --eval mAP
        voxel_semantics = kwargs['voxel_semantics'][0] # (b, 200, 200, 16)
        mask_camera = kwargs['mask_camera'][0] # (b, 200, 200, 16)
        if self.dataset_type == 'nuscenes' and self.use_mask:
            # set mask camera hits to 17 in voxel semantics
            voxel_semantics[mask_camera == 0] = 17
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        if self.dataset_type == 'waymo' and self.use_mask:
            # set mask camera hits to 15 in voxel semantics
            voxel_semantics[mask_camera == 0] = 15
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 15
        coo_list_gt = [] 
        for b in range(voxel_semantics.shape[0]):
            current_voxel_grid = voxel_semantics[b]
            if self.dataset_type == 'nuscenes':
                mask = current_voxel_grid != 17
            if self.dataset_type == 'waymo':
                mask = current_voxel_grid != 15
            coords = torch.argwhere(mask) # (dense_points,3)
            coo_list_gt.append(coords)    
            
        # voxelization of pointcloud
        # for nuscenes: take the first 7 elements
        if self.dataset_type == 'nuscenes':
            points = [p[:, :7] for p in points]
        if self.dataset_type == 'waymo':
            points = [p[:, :8] for p in points]
        voxels, num_points, coors = self.voxelize(points)
        if self.dataset_type == 'nuscenes':
            coors[:, 3] = 200 - coors[:, 3]  # Reverse the y direction
            coors = coors[:, [0, 2, 3, 1]] # move b,z,x,y,to b, x, y,z
        if self.dataset_type == 'waymo':
            coors = coors[:, [0, 3, 2, 1]]
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        
         # sparse to dense lidar semantic segmentation
        pts_sparse_tensor = ME.SparseTensor(
            features = voxel_features, 
            coordinates = coors,
            device = voxel_features.device,
        )
        cm = pts_sparse_tensor.coordinate_manager
        target_key, _ = cm.insert_and_map(
            ME.utils.batched_coordinates(coo_list_gt).to(voxel_features.device),
            string_id = "target",
        )
        
        pts_feats = self.lidar_backbone(pts_sparse_tensor)
        _, _, pts_feat = self.lidar_neck(pts_feats, target_key)
        if self.use_predicter:
            pts_feat = self.predicter(pts_feat)
               
        pred_coords, pred_feats = pts_feat.decomposed_coordinates_and_features
        # Initialize a 200x200x16 grid with default class 17 (indicating empty space), 15 for waymo
        if self.dataset_type == 'nuscenes':
            grid = np.full((200, 200, 16), 17, dtype=np.uint8)
        if self.dataset_type == 'waymo':
            grid = np.full((200, 200, 16), 15, dtype=np.uint8)
        for coords, feats in zip(pred_coords, pred_feats):
            # Perform argmax on the features
            
            # move coords to cpu 
            coords = coords.cpu().numpy()
            class_predictions_gpu = feats.argmax(dim=1)
            # Ensure coordinates are within the bounds [0, 200) for x and y, and [0, 16) for z
            in_bounds_mask = (
                (coords[:, 0] >= 0) & (coords[:, 0] < 200) &
                (coords[:, 1] >= 0) & (coords[:, 1] < 200) &
                (coords[:, 2] >= 0) & (coords[:, 2] < 16)
            )
            # Apply the bounds mask
            valid_coords = coords[in_bounds_mask]
            valid_preds = class_predictions_gpu[in_bounds_mask]
            # Assign the class predictions to the corresponding grid locations
            grid[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = valid_preds.cpu().numpy().astype(np.uint8)
        return [grid]
    
    
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
        
        losses = dict()
        
         # mainpulate gt to fit supervision for minkowski engine
        voxel_semantics = kwargs['voxel_semantics'] # (b, 200, 200, 16)
        mask_camera = kwargs['mask_camera'] # (b, 200, 200, 16)
        if self.dataset_type == 'nuscenes' and self.use_mask:
            # set mask camera hits to 17 in voxel semantics
            voxel_semantics[mask_camera == 0] = 17
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        if self.dataset_type == 'waymo' and self.use_mask:
            # set mask camera hits to 15 in voxel semantics
            voxel_semantics[mask_camera == 0] = 15
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 15
            

        coo_list_gt = [] 
        semantics_list_gt = []
        for b in range(voxel_semantics.shape[0]):
            current_voxel_grid = voxel_semantics[b]
            if self.dataset_type == 'nuscenes':
                mask = current_voxel_grid != 17
            if self.dataset_type == 'waymo':
                mask = current_voxel_grid != 15
            coords = torch.argwhere(mask) # (dense_points,3)
            coo_list_gt.append(coords)    
            all_feats = current_voxel_grid.view(-1, 1)  # (200*200*16, 1)
            all_coords = self.COO_format_coords.to(current_voxel_grid.device)
            all_coords_and_feats = torch.cat([all_coords, all_feats], dim=1)
            semantics_list_gt.append(all_coords_and_feats) # (200*200*16, 4)

        # for nuscenes: take xyzintensityrgb only from points list, which is the first 7 elements
        if self.dataset_type == 'nuscenes':
            points = [p[:, :7] for p in points]
        # for waymo: take xyz range intensity rgb, first 8 elements
        if self.dataset_type == 'waymo':
            points = [p[:, :8] for p in points]
        voxels, num_points, coors = self.voxelize(points)
        if self.dataset_type == 'nuscenes':
            coors[:, 3] = 200 - coors[:, 3]  # Reverse the y direction
            coors = coors[:, [0, 2, 3, 1]] # move b,z,x,y,to b, x, y,z
        if self.dataset_type == 'waymo':
            coors = coors[:, [0, 3, 2, 1]] # move b,z,y,x,to b, x, y,z
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        
        
        # save coors as ply together with coo_list_gt
        # first separate coors by its first column (batch index)
        # Separate coordinates by batch index1
        # coors_print = [coors[coors[:, 0] == i][:, 1:] for i in range(coors[:, 0].max() + 1)]
    
        # for i, (each_coors, each_gt) in enumerate(zip(coors_print, coo_list_gt)):
        #     # Define the vertex dtype
        #     dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
            
        #     # Convert coordinates to structured arrays
        #     vertex_coors = np.array([tuple(point) for point in each_coors], dtype=dtype)
        #     vertex_gt = np.array([tuple(point) for point in each_gt], dtype=dtype)
            
        #     # Save each_coors to PLY
        #     ply_coors = PlyElement.describe(vertex_coors, 'vertex')
        #     with open(f"{self.global_count}_batch_{i}_coors.ply", 'wb') as f:
        #         PlyData([ply_coors]).write(f)
            
        #     # Save each_gt to PLY
        #     ply_gt = PlyElement.describe(vertex_gt, 'vertex')
        #     with open(f"{self.global_count}_batch_{i}_gt.ply", 'wb') as f:
        #         PlyData([ply_gt]).write(f)

        # sparse to dense lidar semantic segmentation
        pts_sparse_tensor = ME.SparseTensor(
            features = voxel_features, 
            coordinates = coors,
            device = voxel_features.device,
        )
        cm = pts_sparse_tensor.coordinate_manager
        target_key, _ = cm.insert_and_map(
            ME.utils.batched_coordinates(coo_list_gt).to(voxel_features.device),
            string_id = "target",
        )
        
        pts_feats = self.lidar_backbone(pts_sparse_tensor)
        out_cls, targets, pts_feat = self.lidar_neck(pts_feats, target_key)

        # bce loss calculation for scene completion point existence
        bce_loss = self.lidar_neck.get_bce_loss(out_cls, targets)   
        losses['loss_bce'] = bce_loss * self.loss_bce_weight
        
        # ce loss calculatiuon for lidar semantic segmentation 
        if self.use_predicter:
            pts_feat = self.predicter(pts_feat)
        ce_loss = self.lidar_neck.get_ce_loss(pts_feat, semantics_list_gt)
        losses['loss_ce'] = ce_loss * self.loss_ce_weight
        
        
        self.global_count += 1
        
        return losses
    
@DETECTORS.register_module()
class BEVStereo4DOCC_robotcycle(BEVStereo4D):
    def __init__(self,
                 loss_occ=None,
                 out_dim=18,
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 lidar_backbone = None,
                 lidar_neck = None,        
                 **kwargs):
        
        super(BEVStereo4DOCC_robotcycle, self).__init__(**kwargs)
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.use_predicter =use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                ME.MinkowskiLinear(self.out_dim, self.out_dim*2),
                MinkowskiSoftplus(),
                ME.MinkowskiLinear(self.out_dim*2, num_classes),
            )
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transfromation = False
        
         # Create x, y, z coordinate grids
        # x = torch.arange(200)
        # y = torch.arange(200)
        # z = torch.arange(16)
        # # Generate a coordinate grid for each dimension (200, 200, 16)
        # mesh_x, mesh_y, mesh_z = torch.meshgrid(x, y, z, indexing='ij')
        # # Flatten the coordinate grids and stack them to get (200*200*16, 3)
        # self.COO_format_coords = torch.stack([mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten()], dim=1)

        
        
        # build lidar backbone and neck
        self.lidar_backbone = builder.build_backbone(lidar_backbone)    
        self.lidar_neck = builder.build_neck(lidar_neck)
    
    def loss_single():
        pass
    
    def simple_test():
        pass
    
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

        losses = dict()
        
    
         # voxelization of pointcloud
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        
        
        return losses

@DETECTORS.register_module()
class BEVStereo4DOCC_MinkOccV2(BEVStereo4D):
    def __init__(self, 
                 out_dim=18,
                 dataset_type='nuscenes',
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 img_backbone = None,
                 img_neck = None,
                 img_voxel_encoder = None,
                 occ_backbone = None,
                 occ_neck = None,
                 loss_ce_weight = None,
                 loss_bce_weight = None,
                 loss_lidarseg_weight = None,
                 **kwargs):
        super(BEVStereo4DOCC_MinkOccV2, self).__init__(**kwargs)
        self.dataset_type = dataset_type
        self.use_mask = use_mask
        self.out_dim = out_dim
        # number of cameras 
        if self.dataset_type == 'nuscenes':
            self.num_views = 6
        if self.dataset_type == 'waymo':
            self.num_views = 5

        # loss weights
        self.loss_ce_weight = loss_ce_weight
        self.loss_bce_weight = loss_bce_weight
        self.loss_lidarseg_weight = loss_lidarseg_weight

        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                ME.MinkowskiLinear(self.out_dim, self.out_dim*2),
                MinkowskiSoftplus(),
                ME.MinkowskiLinear(self.out_dim*2, num_classes),
            )
        # Create x, y, z coordinate grids
        x = torch.arange(200)
        y = torch.arange(200)
        z = torch.arange(16)
        # Generate a coordinate grid for each dimension (200, 200, 16)
        mesh_x, mesh_y, mesh_z = torch.meshgrid(x, y, z, indexing='ij')
        # Flatten the coordinate grids and stack them to get (200*200*16, 3)
        self.COO_format_coords = torch.stack([mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten()], dim=1)

        # build minkgenunet
        self.occ_backbone = builder.build_backbone(occ_backbone)
        self.occ_neck = builder.build_neck(occ_neck)

        # new stuff 
        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)
        self.img_voxel_encoder = builder.build_voxel_encoder(img_voxel_encoder)
        
        # global self count for debugging
        # self.global_count = 0
        
    def simple_test(self,
                    points,
                    img_metas,
                    img_inputs = None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        # python tools/test.py configs/bevdet_occ/minkoccv2.py epoch_100.pth --eval mAP

        gpu = points[0].device
        # quantize lidar points and its corresponding image coordinates to sparse tensor to be processed by minkresunet
        lidar_coords = []
        lidar_points = []
        uv_for_each_cam = []
        for each_points in points:
            # for nuscenes its xyzintensitylidarsegrgbuv1uv2uv3uv4uv5uv6
            # each_points = each_points.cpu()
            if self.dataset_type == 'nuscenes':
                each_lidarseg = each_points[:, 4].to(torch.int) # (n,)
                epc = each_points[:, :3]  # (n, 3)
                epf = torch.cat([each_points[:, 3:4], each_points[:, 5:8]], dim=1)  # (n, 4)
                uv = each_points[:, 7:]  # (n, 12)
                epc_epf = torch.cat([epc, epf], dim=1)  # (n, 7)
            # TODO: check waymo
            lidar_points.append(epc_epf)
            uv_for_each_cam.append(uv)
            # lidar_labels.append(each_lidarseg)
            lidar_coords.append(epc)
            
        # voxelization of lidar_points 
        voxels, coors = self.voxelize(lidar_points)
        voxel_feats, voxel_feats_coors = self.pts_voxel_encoder(voxels, coors)

        # camera image extraction 
        img_feats = self.extract_img_feat(img=img_inputs[0])
        # Initialize an empty list to hold the unprojected image features for each batch
        points_img_feats_all_batches = []
        # Loop over each batch
        for each_batch in range(img_feats.shape[0]):
            # Pre-fetch UV and image features for all views in the batch to avoid repeated memory allocations
            uv_batch = uv_for_each_cam[each_batch]
            img_feats_batch = img_feats[each_batch]
            num_points = uv_batch.shape[0]
            # Initialize feature accumulation tensors on device
            points_feat_sum = torch.zeros((num_points, img_feats.shape[2]), device=gpu)
            points_feat_count = torch.zeros(num_points, device=gpu)
            # Unproject image features to the point cloud across all camera views
            for each_camera_view in range(img_feats.shape[1]):
                uv = uv_batch[:, each_camera_view * 2 : each_camera_view * 2 + 2]
                img_feat = img_feats_batch[each_camera_view]
                # Use the helper function to get RGB feature values
                indices, feat_values = self.get_rgb_values(uv, img_feat)
                # Accumulate features only if there are valid points
                if indices.numel() > 0 and feat_values.numel() > 0:
                    points_feat_sum[indices] += feat_values
                    points_feat_count[indices] += 1
            # Avoid division by zero by clamping
            points_feat_avg = points_feat_sum / points_feat_count.unsqueeze(-1).clamp(min=1e-6)
            points_feat_avg[points_feat_count == 0] = 0
            # Concatenate lidar coordinates with their corresponding averaged image features
            # concate with its corresponding lidar coordinates
            points_img_feats_all_batches.append(torch.cat([lidar_coords[each_batch], points_feat_avg], dim=1))

        # voxelization of points img feats (N, Cam feats)
        cam_voxels, cam_coors = self.voxelize(points_img_feats_all_batches)
        cam_feats, cam_feats_coors = self.img_voxel_encoder(cam_voxels, cam_coors)
    
        # merge voxel and cam feats
        merged_coors, merged_feats = self.batchwise_merge(cam_feats, voxel_feats, cam_feats_coors, voxel_feats_coors)
        if self.dataset_type == 'nuscenes':
            merged_coors[:, 3] = 200 - merged_coors[:, 3]  # Reverse the y direction
            merged_coors = merged_coors[:, [0, 2, 3, 1]] # move b,z,x,y,to b, x, y,z
        if self.dataset_type == 'waymo':
            merged_coors = merged_coors[:, [0, 3, 2, 1]]

        # process ground truth 3D semantic occupancy labels 
        voxel_semantics = kwargs['voxel_semantics'][0] # (b, 200, 200, 16)
        mask_camera = kwargs['mask_camera'][0] # (b, 200, 200, 16)
        mask_camera = mask_camera.to(torch.bool)
        if self.dataset_type == 'nuscenes' and self.use_mask:
            voxel_semantics[mask_camera == 0] = 17
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        if self.dataset_type == 'waymo' and self.use_mask:
            voxel_semantics[mask_camera == 0] = 15
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 15
        coo_list_gt = [] 
        for b in range(voxel_semantics.shape[0]):
            current_voxel_grid = voxel_semantics[b]
            if self.dataset_type == 'nuscenes':
                mask = current_voxel_grid != 17
            if self.dataset_type == 'waymo':
                mask = current_voxel_grid != 15
            coords = torch.argwhere(mask) # (dense_points,3)
            coo_list_gt.append(coords)    


        
        # create sparse tensor
        ME_input_tensor = ME.SparseTensor(
            features = merged_feats,
            coordinates = merged_coors,
            device = gpu
        )
        cm = ME_input_tensor.coordinate_manager
        target_key, _ = cm.insert_and_map(
            ME.utils.batched_coordinates(coo_list_gt).to(ME_input_tensor.device),
            string_id = "target",
        )

        # minkoccunet to process combined features 
        pts_feats = self.occ_backbone(ME_input_tensor)
        _, _, pts_feat = self.occ_neck(pts_feats, target_key)
        if self.use_predicter:
            pts_feat = self.predicter(pts_feat)
        
        pred_coords, pred_feats = pts_feat.decomposed_coordinates_and_features
        # Initialize a 200x200x16 grid with default class 17 (indicating empty space)
        grid = np.full((200, 200, 16), 17, dtype=np.uint8)
        for coords, feats in zip(pred_coords, pred_feats):
            # Perform argmax on the features
            
            # move coords to cpu 
            coords = coords.cpu().numpy()
            class_predictions_gpu = feats.argmax(dim=1)
            # Ensure coordinates are within the bounds [0, 200) for x and y, and [0, 16) for z
            in_bounds_mask = (
                (coords[:, 0] >= 0) & (coords[:, 0] < 200) &
                (coords[:, 1] >= 0) & (coords[:, 1] < 200) &
                (coords[:, 2] >= 0) & (coords[:, 2] < 16)
            )
            # Apply the bounds mask
            valid_coords = coords[in_bounds_mask]
            valid_preds = class_predictions_gpu[in_bounds_mask]
            # Assign the class predictions to the corresponding grid locations
            grid[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = valid_preds.cpu().numpy().astype(np.uint8)
        return [grid]

    def extract_img_feat(self, img):
        """Extract features from images."""
        if img is None or self.img_backbone is None:
            return None

        # Get the original image height and width
        img_src_height, img_src_width = img.shape[-2], img.shape[-1]
        B, N, C, H, W = img.shape

        # Reshape image based on batch size
        img = img.view(B * N, C, H, W) if img.dim() == 5 else img.squeeze(0)

        # Extract features using the image backbone
        img_feats = self.img_backbone(img)

        # Process features using the image neck, if available
        if self.img_neck:
            img_feats = self.img_neck(img_feats)
            if isinstance(img_feats, (tuple, list)):
                img_feats = img_feats[0]
        # Reshape features to (B, N, C, H, W)
        _, output_dim, _, _ = img_feats.shape
        # Resize features to the original image dimensions
        img_feats_resized = F.interpolate(
            img_feats,
            size=(img_src_height, img_src_width),
            mode='nearest'
        )
        # Reshape back to (B, N, C, img_src_height, img_src_width)
        img_feats_resized = img_feats_resized.view(B, N, output_dim, img_src_height, img_src_width)
        return img_feats_resized
    
    def get_rgb_values(self, points_img, img):
        """
        Function to get RGB values at the projected 2D points using bilinear interpolation.
        """
        # Get image dimensions
        
        height, width = img.shape[1], img.shape[2]
        device = img.device

        # Normalize coordinates to [-1, 1] for grid_sample
        u = 2 * (points_img[:, 0] / (width - 1)) - 1
        v = 2 * (points_img[:, 1] / (height - 1)) - 1

        # Create a mask for points inside the image boundaries
        mask = (u >= -1) & (u <= 1) & (v >= -1) & (v <= 1)
        indices = mask.nonzero(as_tuple=False).squeeze(-1)

        # Get valid normalized coordinates
        u_valid = u[mask]
        v_valid = v[mask]

        if u_valid.numel() == 0:
            return torch.tensor([], dtype=torch.long, device=device), torch.tensor([], dtype=torch.float32, device=device)

        # Prepare grid for sampling
        grid = torch.stack((u_valid, v_valid), dim=-1).unsqueeze(0).unsqueeze(2)  # Shape: (1, N, 1, 2)

        # Sample RGB values using grid_sample with bilinear interpolation
        rgb_sampled = F.grid_sample(img.unsqueeze(0), grid, align_corners=True, mode='bilinear', padding_mode='zeros')  # Shape: (1, 3, N, 1)
        rgb_values = rgb_sampled.squeeze(0).squeeze(2).permute(1, 0)  # Shape: (N, 3)
    
        return indices, rgb_values

    def batchwise_merge(self, cam_feats, voxel_feats, cam_feats_coors, voxel_feats_coors):
        """
        Merges camera and voxel features batchwise such that any overlapping coordinates
        within the same batch get their features aggregated (summed) in the merged output.
        All unique coordinates are preserved, considering the batch index.

        Args:
            cam_feats (torch.Tensor): Camera features with shape (N2, C).
            voxel_feats (torch.Tensor): Voxel features with shape (N1, C).
            cam_feats_coors (torch.Tensor): Coordinates for camera features (N2, 4),
                where each row is [batch_idx, x, y, z].
            voxel_feats_coors (torch.Tensor): Coordinates for voxel features (N1, 4),
                where each row is [batch_idx, x, y, z].

        Returns:
            merged_coors (torch.Tensor): Merged coordinates with shape (N_merged, 4).
            merged_feats (torch.Tensor): Merged features with shape (N_merged, C).
        """

        # Combine coordinates and features from camera and voxel
        all_coors = torch.cat([voxel_feats_coors, cam_feats_coors], dim=0)  # Shape (N1 + N2, 4)
        all_feats = torch.cat([voxel_feats, cam_feats], dim=0)              # Shape (N1 + N2, C)

        # Get unique batch indices
        batch_indices = all_coors[:, 0].unique()

        merged_coors_list = []
        merged_feats_list = []

        # Process each batch independently
        for batch_idx in batch_indices:
            # Filter coordinates and features for the current batch
            batch_mask = all_coors[:, 0] == batch_idx
            batch_coors = all_coors[batch_mask]
            batch_feats = all_feats[batch_mask]

            # Find unique coordinates within the batch (excluding batch index column)
            unique_batch_coors, inverse_indices = torch.unique(batch_coors[:, 1:], dim=0, return_inverse=True)

            # Aggregate features for unique coordinates
            merged_batch_feats = torch.zeros(
                (unique_batch_coors.shape[0], batch_feats.shape[1]),  # (N_merged_batch, C)
                device=batch_feats.device
            )
            merged_batch_feats.index_add_(0, inverse_indices, batch_feats)

            # Reattach the batch index to coordinates
            batch_indices_column = batch_idx.unsqueeze(0).repeat(unique_batch_coors.shape[0], 1)
            merged_batch_coors = torch.cat([batch_indices_column, unique_batch_coors], dim=1)

            # Append results for the current batch
            merged_coors_list.append(merged_batch_coors)
            merged_feats_list.append(merged_batch_feats)

        # Concatenate results across all batches
        merged_coors = torch.cat(merged_coors_list, dim=0)
        merged_feats = torch.cat(merged_feats_list, dim=0)

        return merged_coors, merged_feats

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    def geometric_augmentation(self, merged_feats, merged_coors, semantics_list_gt, coo_list_gt):
        """
        Apply flips and rotations independently for each batch.
        
        Args:
            merged_feats: (N, C) feature tensors
            merged_coors: (N, 4) coordinates (batch, x, y, z)
            semantics_list_gt: list of (200*200*16, 4) tensors with (x,y,z,semantic_cls), includes zero coordinates
            coo_list_gt: list of (M, 3) tensors with (x,y,z), only non-zero coordinates
        """
        GRID_SIZE = {
            'x': 200,
            'y': 200,
            'z': 16
        }
        
        # Get number of unique batches
        num_batches = int(merged_coors[:, 0].max() + 1)
        
        # Generate random transformations for each batch
        k_per_batch = torch.randint(1, 4, (num_batches,))  # Rotation
        flip_x = torch.randint(0, 2, (num_batches,)).bool()  # Random flip X
        flip_y = torch.randint(0, 2, (num_batches,)).bool()  # Random flip Y
        
        # Create copies for modification
        new_coors = merged_coors.clone()
        new_semantics_list_gt = [sem_gt.clone() for sem_gt in semantics_list_gt]
        new_coo_list_gt = [coo.clone() for coo in coo_list_gt]
        
        # Apply transformations for each batch separately
        for batch_idx in range(num_batches):
            # Get mask for current batch
            batch_mask = merged_coors[:, 0] == batch_idx
            
            # First apply flips if needed
            if flip_x[batch_idx]:
                # Flip merged_coors
                new_coors[batch_mask, 1] = GRID_SIZE['x'] - 1 - new_coors[batch_mask, 1]
                # Flip semantics_list_gt
                new_semantics_list_gt[batch_idx][:, 0] = GRID_SIZE['x'] - 1 - new_semantics_list_gt[batch_idx][:, 0]
                # Flip coo_list_gt
                new_coo_list_gt[batch_idx][:, 0] = GRID_SIZE['x'] - 1 - new_coo_list_gt[batch_idx][:, 0]
                
            if flip_y[batch_idx]:
                # Flip merged_coors
                new_coors[batch_mask, 2] = GRID_SIZE['y'] - 1 - new_coors[batch_mask, 2]
                # Flip semantics_list_gt
                new_semantics_list_gt[batch_idx][:, 1] = GRID_SIZE['y'] - 1 - new_semantics_list_gt[batch_idx][:, 1]
                # Flip coo_list_gt
                new_coo_list_gt[batch_idx][:, 1] = GRID_SIZE['y'] - 1 - new_coo_list_gt[batch_idx][:, 1]
            
            # # Then apply rotation
            # # Get rotation amount for this batch
            # k = k_per_batch[batch_idx].item()
            
            # # Rotate coordinates for this batch
            # x, y = new_coors[batch_mask, 1], new_coors[batch_mask, 2]
            
            # if k == 1:  # 90° clockwise
            #     x_new = y.clone()
            #     y_new = -x.clone() + (GRID_SIZE['y'] - 1)
            # elif k == 2:  # 180°
            #     x_new = -x.clone() + (GRID_SIZE['x'] - 1)
            #     y_new = -y.clone() + (GRID_SIZE['y'] - 1)
            # else:  # 270° clockwise (k=3)
            #     x_new = -y.clone() + (GRID_SIZE['x'] - 1)
            #     y_new = x.clone()
                
            # new_coors[batch_mask, 1] = x_new
            # new_coors[batch_mask, 2] = y_new
            
            # # Rotate corresponding ground truth for this batch (semantics_list_gt)
            # x_gt, y_gt = new_semantics_list_gt[batch_idx][:, 0], new_semantics_list_gt[batch_idx][:, 1]
            
            # if k == 1:  # 90° clockwise
            #     x_gt_new = y_gt.clone()
            #     y_gt_new = -x_gt.clone() + (GRID_SIZE['y'] - 1)
            # elif k == 2:  # 180°
            #     x_gt_new = -x_gt.clone() + (GRID_SIZE['x'] - 1)
            #     y_gt_new = -y_gt.clone() + (GRID_SIZE['y'] - 1)
            # else:  # 270° clockwise (k=3)
            #     x_gt_new = -y_gt.clone() + (GRID_SIZE['x'] - 1)
            #     y_gt_new = x_gt.clone()
                
            # new_semantics_list_gt[batch_idx][:, 0] = x_gt_new
            # new_semantics_list_gt[batch_idx][:, 1] = y_gt_new
            
            # # Rotate corresponding coo_list_gt for this batch
            # x_coo, y_coo = new_coo_list_gt[batch_idx][:, 0], new_coo_list_gt[batch_idx][:, 1]
            
            # if k == 1:  # 90° clockwise
            #     x_coo_new = y_coo.clone()
            #     y_coo_new = -x_coo.clone() + (GRID_SIZE['y'] - 1)
            # elif k == 2:  # 180°
            #     x_coo_new = -x_coo.clone() + (GRID_SIZE['x'] - 1)
            #     y_coo_new = -y_coo.clone() + (GRID_SIZE['y'] - 1)
            # else:  # 270° clockwise (k=3)
            #     x_coo_new = -y_coo.clone() + (GRID_SIZE['x'] - 1)
            #     y_coo_new = x_coo.clone()
                
            # new_coo_list_gt[batch_idx][:, 0] = x_coo_new
            # new_coo_list_gt[batch_idx][:, 1] = y_coo_new
        
        return merged_feats, new_coors, new_semantics_list_gt, new_coo_list_gt
    
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
      
        losses = dict()
        # define gpu device 
        gpu = points[0].device
        # points = list[torch.Tensor], optional): Points of each sample with rgb intensity (n, 8), (xyz, rgbintensityindices) + (n, 12), (uv * num views)
        # lidarseg = list[torch.Tensor], optional): Points of each sample with semantic segmentation labels (n,)
        # Quantize lidar points and its corresponding image coordinates to sparse tensor to be processed by minkresunet 
        lidar_coords = []
        lidar_points = []
        uv_for_each_cam = []
        # lidar_labels = []
        for each_points in points:
            # for nuscenes its xyzintensitylidarsegrgbuv1uv2uv3uv4uv5uv6
            # each_points = each_points.cpu()
            if self.dataset_type == 'nuscenes':
                each_lidarseg = each_points[:, 4].to(torch.int) # (n,)
                epc = each_points[:, :3]  # (n, 3)
                epf = torch.cat([each_points[:, 3:4], each_points[:, 5:8]], dim=1)  # (n, 4)
                uv = each_points[:, 7:]  # (n, 12)
                epc_epf = torch.cat([epc, epf], dim=1)  # (n, 7)
            # TODO: check waymo
            lidar_points.append(epc_epf)
            uv_for_each_cam.append(uv)
            # lidar_labels.append(each_lidarseg)
            lidar_coords.append(epc)

        # voxelization of lidar_points 
        voxels, coors = self.voxelize(lidar_points)
        voxel_feats, voxel_feats_coors = self.pts_voxel_encoder(voxels, coors)


        # camera image extraction 
        img_feats = self.extract_img_feat(img=img_inputs[0])
        # Initialize an empty list to hold the unprojected image features for each batch
        points_img_feats_all_batches = []
        # Loop over each batch
        for each_batch in range(img_feats.shape[0]):
            # Pre-fetch UV and image features for all views in the batch to avoid repeated memory allocations
            uv_batch = uv_for_each_cam[each_batch]
            img_feats_batch = img_feats[each_batch]
            num_points = uv_batch.shape[0]
            # Initialize feature accumulation tensors on device
            points_feat_sum = torch.zeros((num_points, img_feats.shape[2]), device=gpu)
            points_feat_count = torch.zeros(num_points, device=gpu)
            # Unproject image features to the point cloud across all camera views
            for each_camera_view in range(img_feats.shape[1]):
                uv = uv_batch[:, each_camera_view * 2 : each_camera_view * 2 + 2]
                img_feat = img_feats_batch[each_camera_view]
                # Use the helper function to get RGB feature values
                indices, feat_values = self.get_rgb_values(uv, img_feat)
                # Accumulate features only if there are valid points
                if indices.numel() > 0 and feat_values.numel() > 0:
                    points_feat_sum[indices] += feat_values
                    points_feat_count[indices] += 1
            # Avoid division by zero by clamping
            points_feat_avg = points_feat_sum / points_feat_count.unsqueeze(-1).clamp(min=1e-6)
            points_feat_avg[points_feat_count == 0] = 0
            # Concatenate lidar coordinates with their corresponding averaged image features
            # concate with its corresponding lidar coordinates
            points_img_feats_all_batches.append(torch.cat([lidar_coords[each_batch], points_feat_avg], dim=1))
        # stack vertically the points_img_feats_all_batches
        # points_img_feats = torch.cat(points_img_feats_all_batches, dim=0)


        # voxelization of points img feats (N, Cam feats)
        cam_voxels, cam_coors = self.voxelize(points_img_feats_all_batches)
        cam_feats, cam_feats_coors = self.img_voxel_encoder(cam_voxels, cam_coors)
        # if self.dataset_type == 'nuscenes':
        #     cam_feats_coors[:, 3] = 200 - cam_feats_coors[:, 3]  # Reverse the y direction
        #     cam_feats_coors = cam_feats_coors[:, [0, 2, 3, 1]] # move b,z,x,y,to b, x, y,z
        # if self.dataset_type == 'waymo':
        #     cam_feats_coors = cam_feats_coors[:, [0, 3, 2, 1]]
        # merge voxel and cam feats
        merged_coors, merged_feats = self.batchwise_merge(cam_feats, voxel_feats, cam_feats_coors, voxel_feats_coors)
        if self.dataset_type == 'nuscenes':
            merged_coors[:, 3] = 200 - merged_coors[:, 3]  # Reverse the y direction
            merged_coors = merged_coors[:, [0, 2, 3, 1]] # move b,z,x,y,to b, x, y,z
        if self.dataset_type == 'waymo':
            merged_coors = merged_coors[:, [0, 3, 2, 1]]

      

        # process ground truth 3D semantic occupancy labels 
        voxel_semantics = kwargs['voxel_semantics'] # (b, 200, 200, 16)
        mask_camera = kwargs['mask_camera'] # (b, 200, 200, 16)
        # convert mask_camera to torch.bool
        mask_camera = mask_camera.to(torch.bool)
        if self.dataset_type == 'nuscenes' and self.use_mask:
            # set mask camera hits to 17 in voxel semantics
            voxel_semantics[mask_camera == 0] = 17
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        if self.dataset_type == 'waymo' and self.use_mask:
            # set mask camera hits to 15 in voxel semantics
            voxel_semantics[mask_camera == 0] = 15
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 15
        coo_list_gt = [] 
        semantics_list_gt = []
                
        
        for b in range(voxel_semantics.shape[0]):
            current_voxel_grid = voxel_semantics[b]
            if self.dataset_type == 'nuscenes':
                mask = current_voxel_grid != 17
            if self.dataset_type == 'waymo':
                mask = current_voxel_grid != 15
            coords = torch.argwhere(mask) # (dense_points,3)
            coo_list_gt.append(coords)    
            all_feats = current_voxel_grid.view(-1, 1)  # (200*200*16, 1)
            all_coords = self.COO_format_coords.to(gpu)
            all_coords_and_feats = torch.cat([all_coords, all_feats], dim=1)
            semantics_list_gt.append(all_coords_and_feats) # (200*200*16, 4)

        # Apply augmentation
        # merged_feats, merged_coors, semantics_list_gt, coo_list_gt = self.geometric_augmentation(
            # merged_feats, merged_coors, semantics_list_gt, coo_list_gt)

        # save coors as ply together with coo_list_gt
        # first separate coors by its first column (batch index)
        # Separate coordinates by batch index1
        # coors_print = [merged_coors[merged_coors[:, 0] == i][:, 1:] for i in range(merged_coors[:, 0].max() + 1)]
    
        # for i, (each_coors, each_gt) in enumerate(zip(coors_print, coo_list_gt)):
        #     # Define the vertex dtype
        #     dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
            
        #     # Convert coordinates to structured arrays
        #     vertex_coors = np.array([tuple(point) for point in each_coors], dtype=dtype)
        #     vertex_gt = np.array([tuple(point) for point in each_gt[:,:3]], dtype=dtype)
            
        #     # Save each_coors to PLY
        #     ply_coors = PlyElement.describe(vertex_coors, 'vertex')
        #     with open(f"{self.global_count}_batch_{i}_coors.ply", 'wb') as f:
        #         PlyData([ply_coors]).write(f)
            
        #     # Save each_gt to PLY
        #     ply_gt = PlyElement.describe(vertex_gt, 'vertex')
        #     with open(f"{self.global_count}_batch_{i}_gt.ply", 'wb') as f:
        #         PlyData([ply_gt]).write(f)
    
        # self.global_count += 1

        # create sparse tensor
        ME_input_tensor = ME.SparseTensor(
            features = merged_feats,
            coordinates = merged_coors,
            device = gpu
        )
        cm = ME_input_tensor.coordinate_manager
        target_key, _ = cm.insert_and_map(
            ME.utils.batched_coordinates(coo_list_gt).to(ME_input_tensor.device),
            string_id = "target",
        )

        # minkoccunet to process combined features 
        pts_feats = self.occ_backbone(ME_input_tensor)
        out_cls, targets, pts_feat = self.occ_neck(pts_feats, target_key)
        # bce loss
        bce_loss = self.occ_neck.get_bce_loss(out_cls, targets)   
        losses['loss_bce'] = bce_loss * self.loss_bce_weight
        # ce loss
        if self.use_predicter:
            pts_feat = self.predicter(pts_feat)
        ce_loss = self.occ_neck.get_ce_loss(pts_feat, semantics_list_gt)
        losses['loss_ce'] = ce_loss * self.loss_ce_weight

        return losses

@DETECTORS.register_module()
class BEVStereo4DOCC_Gaussian(BEVStereo4D):

    def __init__(self,
                 dataset_type='nuscenes',
                 gaussianinit=None,
                 **kwargs):
        super(BEVStereo4DOCC_Gaussian, self).__init__(**kwargs)
        self.dataset_type = dataset_type
        self.gaussianinit = builder.build_head(gaussianinit)

    def loss_single(self):
        pass

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        pass

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
                each point has shape (N, 8) -> (x, y, z, intensity, index, r, g, b)
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
        losses = dict()
        # lss 
        img_feats, _, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth
        
        # use lidar points and multiview camera features to initialize 3D gaussians 
        gaussians = self.gaussianinit(points, img_feats, **kwargs)
        
        pppp
        return losses

@DETECTORS.register_module()
class MinkOccV3(DynamicMVXFasterRCNN):
    
    def __init__(self, 
                 renderer_cfg = None,
                 occ_backbone = None,
                 occ_neck = None,
                 out_dim = 18,
                 loss_bce_weight = None,
                 loss_ce_weight = None,
                 use_mask = False,
                 dataset_type = 'nuscenes',
                 freeze_layers = False,
                 **kwargs):
        super(MinkOccV3, self).__init__(**kwargs)
        
        self.dataset_type = dataset_type
        self.use_mask = use_mask
        self.out_dim = out_dim
        if self.dataset_type == 'nuscenes':
            self.num_views = 6
            self.num_classes = 18
        if self.dataset_type == 'waymo':
            self.num_views = 5
            self.num_classes = 16
        self.loss_ce_weight = loss_ce_weight
        self.loss_bce_weight = loss_bce_weight
        
        self.predicter = nn.Sequential(
            ME.MinkowskiLinear(self.out_dim, self.out_dim*2),
            MinkowskiSoftplus(),
            ME.MinkowskiLinear(self.out_dim*2, self.num_classes),
        )
    
        # Create x, y, z coordinate grids
        x = torch.arange(200)
        y = torch.arange(200)
        z = torch.arange(16)
        # Generate a coordinate grid for each dimension (200, 200, 16)
        mesh_x, mesh_y, mesh_z = torch.meshgrid(x, y, z, indexing='ij')
        # Flatten the coordinate grids and stack them to get (200*200*16, 3)
        self.COO_format_coords = torch.stack([mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten()], dim=1)
        
        self.occ_backbone = builder.build_backbone(occ_backbone)
        self.occ_neck = builder.build_neck(occ_neck)
        
        # global count is only for visualization checking in coding
        self.global_count = 0
        
        # Build the renderer if config is provided
        if renderer_cfg:
            self.renderer = build_renderer(renderer_cfg)
        else:
            self.renderer = None
            
        
        # for finetuning experiment (train nuscenes, finetune waymo etc)
        if freeze_layers:
            self.freeze_layers()

    def freeze_layers(self):
        """
        Freezes all layers except occ_backbone and occ_neck.
        This means only these two components will be updated during training.
        """
        # Freeze image backbone if it exists
        for param in self.img_backbone.parameters():
            param.requires_grad = False
        
        # Freeze image neck if it exists
        for param in self.img_neck.parameters():
            param.requires_grad = False
                
        # Freeze voxel encoder
        for param in self.pts_voxel_encoder.parameters():
            param.requires_grad = False
            

        # Verify that only occ_backbone and occ_neck and mink predictor are trainable
        def print_trainable_status(module_name, module):
            trainable_params = sum(p.requires_grad for p in module.parameters())
            total_params = sum(1 for _ in module.parameters())
            print(f"{module_name}: {trainable_params}/{total_params} parameters trainable")

        print_trainable_status("occ_backbone", self.occ_backbone)
        print_trainable_status("occ_neck", self.occ_neck)
        print_trainable_status("predicter", self.predicter)
        print_trainable_status("img_backbone", self.img_backbone)
        print_trainable_status("img_neck", self.img_neck)
        print_trainable_status("pts_voxel_encoder", self.pts_voxel_encoder)
        
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, points, img_feats, img_metas):
        """Extract point features."""
        voxels, coors = self.voxelize(points)
        voxel_features, feature_coors = self.pts_voxel_encoder(
            voxels, coors, points, img_feats, img_metas)
        # batch_size = coors[-1, 0] + 1
        return (voxel_features, feature_coors)

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        
        img_feats = self.extract_img_feat(img, img_metas)
        occ_feats, occ_coords = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, occ_feats, occ_coords)
    
    def simple_test(self, 
                    points, 
                    img_metas, 
                    img, 
                    **kwargs):
        
        gpu = points[0].device
        img_feats, occ_feats, occ_coords = self.extract_feat(
            [points], img=img, img_metas=[img_metas])
        
        if self.dataset_type == 'nuscenes':
            occ_coords[:, 3] = 200 - occ_coords[:, 3]  # Reverse the y direction
            occ_coords = occ_coords[:, [0, 2, 3, 1]] # move b,z,x,y,to b, x, y,z
        if self.dataset_type == 'waymo':
            occ_coords = occ_coords[:, [0, 3, 2, 1]]
        
        occ_sparse_tensor = ME.SparseTensor(
            features = occ_feats, 
            coordinates = occ_coords,
            device = occ_feats.device,
        )
        

        
        # process ground truth 3D semantic occupancy labels 
        voxel_semantics = kwargs['voxel_semantics'] # (b, 200, 200, 16)
        mask_camera = kwargs['mask_camera'] # (b, 200, 200, 16)
        
        # convert mask_camera to torch.bool
        mask_camera = mask_camera.to(torch.bool)
        if self.dataset_type == 'nuscenes' and self.use_mask:
            # set mask camera hits to 17 in voxel semantics
            voxel_semantics[mask_camera == 0] = 17
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        if self.dataset_type == 'waymo' and self.use_mask:
            # set mask camera hits to 15 in voxel semantics
            voxel_semantics[mask_camera == 0] = 15
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 15
        coo_list_gt = [] 
        semantics_list_gt = []
        for b in range(voxel_semantics.shape[0]):
            current_voxel_grid = voxel_semantics[b]
            if self.dataset_type == 'nuscenes':
                mask = current_voxel_grid != 17
            if self.dataset_type == 'waymo':
                mask = current_voxel_grid != 15
            coords = torch.argwhere(mask) # (dense_points,3)
            coo_list_gt.append(coords)    
            all_feats = current_voxel_grid.view(-1, 1)  # (200*200*16, 1)
            all_coords = self.COO_format_coords.to(gpu)
            all_coords_and_feats = torch.cat([all_coords, all_feats], dim=1)
            semantics_list_gt.append(all_coords_and_feats) # (200*200*16, 4)
        
        cm = occ_sparse_tensor.coordinate_manager
        target_key, _ = cm.insert_and_map(
            ME.utils.batched_coordinates(coo_list_gt).to(occ_feats.device),
            string_id = "target",
        )
        
        intermediate_feats = self.occ_backbone(occ_sparse_tensor)
        _, _, pts_feat = self.occ_neck(intermediate_feats, target_key)
        pts_feat = self.predicter(pts_feat)
        
        
        pred_coords, pred_feats = pts_feat.decomposed_coordinates_and_features
        # Initialize a 200x200x16 grid with default class 17 (indicating empty space)
        if self.dataset_type == 'nuscenes':
            grid = np.full((200, 200, 16), 17, dtype=np.uint8)
        if self.dataset_type == 'waymo':
            grid = np.full((200, 200, 16), 15, dtype=np.uint8)
            
        for coords, feats in zip(pred_coords, pred_feats):
            # Perform argmax on the features
            
            # move coords to cpu 
            coords = coords.cpu().numpy()
            class_predictions_gpu = feats.argmax(dim=1)
            # Ensure coordinates are within the bounds [0, 200) for x and y, and [0, 16) for z
            in_bounds_mask = (
                (coords[:, 0] >= 0) & (coords[:, 0] < 200) &
                (coords[:, 1] >= 0) & (coords[:, 1] < 200) &
                (coords[:, 2] >= 0) & (coords[:, 2] < 16)
            )
            # Apply the bounds mask
            valid_coords = coords[in_bounds_mask]
            valid_preds = class_predictions_gpu[in_bounds_mask]
            # Assign the class predictions to the corresponding grid locations
            grid[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = valid_preds.cpu().numpy().astype(np.uint8)
        
        return [grid]
    
    @property
    def with_renderer(self):
        return hasattr(self, 'renderer') and self.renderer is not None


    def forward_train(self,
                    points=None,
                    img_metas=None,
                    gt_bboxes_3d=None,
                    gt_labels_3d=None,
                    gt_labels=None,
                    gt_bboxes=None,
                    img=None,
                    supervision_2d_mask=None,
                    supervision_2d_conf=None,
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
            img (torch.Tensor, optional): Images of each sample with shape
                (B, N_views, C, H, W). Defaults to None.
            supervision_2d_mask (torch.Tensor, optional): The 2D supervision
                semantic masks for each camera view. Shape is
                (B, N_views, H_mask, W_mask). Defaults to None.
            supervision_2d_conf (torch.Tensor, optional): The 2D supervision
                confidence scores for each camera view. Shape is
                (B, N_views, H_mask, W_mask). Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Losses of different branches.
            
        """
        
        losses = dict()
        gpu = points[0].device

        # --- Step 1: Extract features for ALL samples ---
        img_feats, occ_feats, occ_coords = self.extract_feat(
            points, img=img, img_metas=img_metas)

        if self.dataset_type == 'nuscenes':
            occ_coords[:, 3] = 200 - occ_coords[:, 3]
            occ_coords = occ_coords[:, [0, 2, 3, 1]]
        if self.dataset_type == 'waymo':
            occ_coords = occ_coords[:, [0, 3, 2, 1]]

        # --- Step 2: Prepare GT for the FULL batch ---
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera'].to(torch.bool)
        if self.dataset_type == 'nuscenes' and self.use_mask:
            voxel_semantics[mask_camera == 0] = 17
        if self.dataset_type == 'waymo' and self.use_mask:
            voxel_semantics[mask_camera == 0] = 15

        coo_list_gt = [] 
        semantics_list_gt = []
        for b in range(voxel_semantics.shape[0]):
            current_voxel_grid = voxel_semantics[b]
            mask = current_voxel_grid != 17
            coords = torch.argwhere(mask)
            coo_list_gt.append(coords)
            
            all_feats = current_voxel_grid.view(-1, 1)
            all_coords = self.COO_format_coords.to(gpu)
            all_coords_and_feats = torch.cat([all_coords, all_feats], dim=1)
            semantics_list_gt.append(all_coords_and_feats)

        # --- Step 3: Unified Network Forward Pass for ALL samples ---
        occ_sparse_tensor = ME.SparseTensor(
            features=occ_feats, 
            coordinates=occ_coords,
            device=occ_feats.device)
        
        cm = occ_sparse_tensor.coordinate_manager
        target_key, _ = cm.insert_and_map(
            ME.utils.batched_coordinates(coo_list_gt).to(occ_feats.device),
            string_id="target",
        )

        intermediate_feats = self.occ_backbone(occ_sparse_tensor)
        out_cls, targets, pts_feat = self.occ_neck(intermediate_feats, target_key)
        
        # --- Step 4: Calculate Losses ---
        bce_loss = self.occ_neck.get_bce_loss(out_cls, targets)
        losses['loss_bce'] = bce_loss * self.loss_bce_weight
        
        pts_feat = self.predicter(pts_feat)

        if self.with_renderer:
            render_losses = self.renderer(
                pts_feat,
                img_metas,
                supervision_2d_mask,
                supervision_2d_conf
            )
            losses.update(render_losses)

        # --- START OF MODIFICATION ---
        # Identify strong samples and compute masked 3D CE loss
        is_strong_sample = [meta.get('is_strong_supervision', False) for meta in img_metas]
        strong_mask = torch.tensor(is_strong_sample, device=gpu)

        # Pass the mask to the loss function. It will be 0 if no strong samples.
        ce_loss = self.occ_neck.get_ce_loss(
            pts_feat, 
            semantics_list_gt,
            loss_mask=strong_mask
        )
        losses['loss_ce'] = ce_loss * self.loss_ce_weight
        # --- END OF MODIFICATION ---

        return losses
    
@DETECTORS.register_module()
class MinkOccV4(DynamicMVXFasterRCNN):
    """
    MinkOccV4 integrates a deformable attention-based fusion mechanism, inspired
    by BEVFormer, to fuse sparse LiDAR features with dense multi-level image
    features. This replaces the point-based fusion in the voxel encoder of
    previous versions.
    """
    def __init__(self,
                 fusion_cfg,  # New required argument for the fusion layer
                 renderer_cfg=None,
                 occ_backbone=None,
                 occ_neck=None,
                 out_dim=18,
                 loss_bce_weight=None,
                 loss_ce_weight=None,
                 use_mask=False,
                 dataset_type='nuscenes',
                 freeze_layers=False,
                 **kwargs):
        # Call the parent constructor from DynamicMVXFasterRCNN
        super(MinkOccV4, self).__init__(**kwargs)

        # --- MinkOccV4 Specific Initializations ---
        self.fusion_layer = builder.build_fusion_layer(fusion_cfg)
        self.dataset_type = dataset_type
        self.use_mask = use_mask
        self.out_dim = out_dim

        if self.dataset_type == 'nuscenes':
            self.num_views = 6
            self.num_classes = 18
        elif self.dataset_type == 'waymo': # Assuming Waymo support might be added
            self.num_views = 5
            self.num_classes = 16
        else:
            raise ValueError(f"Unsupported dataset_type: {self.dataset_type}")

        self.loss_ce_weight = loss_ce_weight
        self.loss_bce_weight = loss_bce_weight

        # The final prediction head after the 3D UNet
        self.predicter = nn.Sequential(
            ME.MinkowskiLinear(self.out_dim, self.out_dim * 2),
            MinkowskiSoftplus(),
            ME.MinkowskiLinear(self.out_dim * 2, self.num_classes),
        )

        # Coordinate grid for creating the target sparse tensor
        x = torch.arange(200)
        y = torch.arange(200)
        z = torch.arange(16)
        mesh_x, mesh_y, mesh_z = torch.meshgrid(x, y, z, indexing='ij')
        self.register_buffer(
            'COO_format_coords',
            torch.stack([mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten()], dim=1),
            persistent=False
        )

        # Build the 3D UNet (backbone and neck for occupancy)
        self.occ_backbone = builder.build_backbone(occ_backbone)
        self.occ_neck = builder.build_neck(occ_neck)

        # Build the renderer for 2D supervision
        self.renderer = builder.build_renderer(renderer_cfg) if renderer_cfg else None

        if freeze_layers:
            self.freeze_layers()

    def freeze_layers(self):
        """Freezes specific layers of the model for fine-tuning."""
        for param in self.img_backbone.parameters():
            param.requires_grad = False
        for param in self.img_neck.parameters():
            param.requires_grad = False
        for param in self.pts_voxel_encoder.parameters():
            param.requires_grad = False
        print("--- Froze img_backbone, img_neck, and pts_voxel_encoder. ---")

    def extract_feat(self, points, img, img_metas):
        """
        Extract features from images and points. This is the core of the
        MinkOccV4 data flow.
        """
        # --- Step 1: Extract Dense Image Features ---
        img_feats = self.extract_img_feat(img, img_metas)

        # --- Step 2: Voxelize LiDAR points and get initial sparse features ---
        voxels, coors = self.voxelize(points)
        # Note: The voxel encoder here should NOT have a fusion layer.
        # It only processes the initial LiDAR point features within each voxel.
        voxel_features, feature_coors = self.pts_voxel_encoder(voxels, coors)

        # --- Step 3: Fuse LiDAR and Image Features using Deformable Attention ---
        # The fusion layer takes the initial sparse features and enhances them
        # with information from the dense image features.
        fused_features = self.fusion_layer(
            occ_feats=voxel_features,
            occ_coords=feature_coors,
            img_feats=img_feats,
            img_metas=img_metas
        )

        # The output of the fusion layer is the final set of features for
        # the sparse tensor, ready for the 3D UNet.
        return (img_feats, fused_features, feature_coors)
    
    @property
    def with_renderer(self):
        return hasattr(self, 'renderer') and self.renderer is not None

    # The forward_train and simple_test methods are nearly identical to MinkOccV3,
    # as the main architectural change is encapsulated within extract_feat.

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      img=None,
                      supervision_2d_mask=None,
                      supervision_2d_conf=None,
                      **kwargs):
        
        losses = dict()
        gpu = points[0].device

        # --- 1. Feature Extraction and Fusion ---
        # This now calls the new MinkOccV4 data flow with deformable fusion.
        img_feats, occ_feats, occ_coords = self.extract_feat(
            points, img=img, img_metas=img_metas)

        # --- 2. Prepare GT Targets ---
        # This logic is identical to MinkOccV3
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera'].to(torch.bool)
        if self.dataset_type == 'nuscenes' and self.use_mask:
            voxel_semantics[mask_camera == 0] = 17
        
        coo_list_gt = [] 
        semantics_list_gt = []
        for b in range(voxel_semantics.shape[0]):
            current_voxel_grid = voxel_semantics[b]
            mask = current_voxel_grid != 17
            coords = torch.argwhere(mask)
            coo_list_gt.append(coords)
            
            all_feats = current_voxel_grid.view(-1, 1)
            all_coords = self.COO_format_coords.to(gpu)
            all_coords_and_feats = torch.cat([all_coords, all_feats], dim=1)
            semantics_list_gt.append(all_coords_and_feats)

        # --- 3. 3D UNet Forward Pass ---
        # The input sparse tensor is now built from the FUSED features
        if self.dataset_type == 'nuscenes':
            occ_coords[:, 3] = 200 - occ_coords[:, 3]
            occ_coords = occ_coords[:, [0, 2, 3, 1]]

        occ_sparse_tensor = ME.SparseTensor(
            features=occ_feats, 
            coordinates=occ_coords,
            device=occ_feats.device)
        
        cm = occ_sparse_tensor.coordinate_manager
        target_key, _ = cm.insert_and_map(
            ME.utils.batched_coordinates(coo_list_gt).to(occ_feats.device),
            string_id="target",
        )

        intermediate_feats = self.occ_backbone(occ_sparse_tensor)
        out_cls, targets, pts_feat = self.occ_neck(intermediate_feats, target_key)
        
        # --- 4. Calculate Losses ---
        # This logic is identical to MinkOccV3
        bce_loss = self.occ_neck.get_bce_loss(out_cls, targets)
        losses['loss_bce'] = bce_loss * self.loss_bce_weight
        
        pts_feat = self.predicter(pts_feat)

        if self.with_renderer:
            render_losses = self.renderer(
                pts_feat,
                img_metas,
                supervision_2d_mask,
                supervision_2d_conf
            )
            losses.update(render_losses)

        is_strong_sample = [meta.get('is_strong_supervision', False) for meta in img_metas]
        strong_mask = torch.tensor(is_strong_sample, device=gpu)

        ce_loss = self.occ_neck.get_ce_loss(
            pts_feat, 
            semantics_list_gt,
            loss_mask=strong_mask
        )
        losses['loss_ce'] = ce_loss * self.loss_ce_weight

        return losses

    def simple_test(self, 
                    points, 
                    img_metas, 
                    img, 
                    **kwargs):
        
        gpu = points[0].device
        # This now calls the new MinkOccV4 data flow with deformable fusion.
        img_feats, occ_feats, occ_coords = self.extract_feat(
            [points], img=img, img_metas=[img_metas])
        
        if self.dataset_type == 'nuscenes':
            occ_coords[:, 3] = 200 - occ_coords[:, 3]
            occ_coords = occ_coords[:, [0, 2, 3, 1]]

        occ_sparse_tensor = ME.SparseTensor(
            features=occ_feats, 
            coordinates=occ_coords,
            device=occ_feats.device)
        
        # The rest of the test logic is identical to MinkOccV3
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera'].to(torch.bool)
        if self.dataset_type == 'nuscenes' and self.use_mask:
            voxel_semantics[mask_camera == 0] = 17
        
        coo_list_gt = [] 
        for b in range(voxel_semantics.shape[0]):
            current_voxel_grid = voxel_semantics[b]
            mask = current_voxel_grid != 17
            coords = torch.argwhere(mask)
            coo_list_gt.append(coords)
        
        cm = occ_sparse_tensor.coordinate_manager
        target_key, _ = cm.insert_and_map(
            ME.utils.batched_coordinates(coo_list_gt).to(occ_feats.device),
            string_id="target",
        )
        
        intermediate_feats = self.occ_backbone(occ_sparse_tensor)
        _, _, pts_feat = self.occ_neck(intermediate_feats, target_key)
        pts_feat = self.predicter(pts_feat)
        
        pred_coords, pred_feats = pts_feat.decomposed_coordinates_and_features
        
        grid = np.full((200, 200, 16), 17, dtype=np.uint8)
            
        for coords, feats in zip(pred_coords, pred_feats):
            coords = coords.cpu().numpy()
            class_predictions_gpu = feats.argmax(dim=1)
            
            in_bounds_mask = (
                (coords[:, 0] >= 0) & (coords[:, 0] < 200) &
                (coords[:, 1] >= 0) & (coords[:, 1] < 200) &
                (coords[:, 2] >= 0) & (coords[:, 2] < 16)
            )
            
            valid_coords = coords[in_bounds_mask]
            valid_preds = class_predictions_gpu[in_bounds_mask]
            
            grid[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = valid_preds.cpu().numpy().astype(np.uint8)
        
        return [grid]