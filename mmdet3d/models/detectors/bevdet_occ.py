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
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 ## add in lidar backbone here
                 lidar_backbone=None,
                 ## add in lidar neck here
                 lidar_neck=None,
                 # add in fuser method here
                 sparse_fusion = None,
                 # add in multi-modal backbone here
                 occ_backbone=None,
                 # add in multi-modal neck here
                 occ_neck=None,
                 
                 **kwargs):
        super(BEVStereo4DOCC, self).__init__(**kwargs)
        # add in lidar backbone and necessary inits here
        self.lidar_backbone = builder.build_backbone(lidar_backbone)
        self.lidar_neck = builder.build_neck(lidar_neck)
        # add in sparse tensor fuser here
        # self.sparse_fusion = builder.build_fusion_layer(sparse_fusion)
        # add in occ backbone and necessary inits here
        # self.occ_backbone = builder.build_backbone(occ_backbone)
        # self.occ_neck = builder.build_neck(occ_neck)
        ######################################
        # declare COO format coordinates from 200x200x16 grid
        # z_voxel, x_voxel, y_voxel = 16, 200, 200
        # Generate the voxel coordinates once (since they are the same for all batches)
        # z_coords, x_coords, y_coords = torch.meshgrid(
            # torch.arange(z_voxel), torch.arange(x_voxel), torch.arange(y_voxel), indexing='ij'
        # )
        # self.COO_format_coords = torch.stack([x_coords, y_coords, z_coords], dim=-1).reshape(-1, 3)  # shape (num_voxels, 3)
        self.GRID_SIZE = (200, 200, 16)
        
        self.out_dim = out_dim
        # out_channels = out_dim if use_predicter else num_classes
        # self.final_conv = ConvModule(
        #                 self.img_view_transformer.out_channels,
        #                 out_channels,
        #                 kernel_size=3,
        #                 stride=1,
        #                 padding=1,
        #                 bias=True,
        #                 conv_cfg=dict(type='Conv3d'))
        self.use_predicter =use_predicter
        # if use_predicter:
        #     self.predicter = nn.Sequential(
        #         nn.Linear(self.out_dim, self.out_dim*2),
        #         nn.Softplus(),
        #         nn.Linear(self.out_dim*2, num_classes),
        #     )
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

    def loss_single(self,voxel_semantics_coo_feats, masked_camera_coo_feats,preds):
        """_summary_

        Args:
            voxel_semantics_coo_feats (torch.tensor): (200x200x16xbatch, 1), voxel semantics class labels
            masked_camera_list_feats (torch.tensor]): (200x200x16xbatch, 1), voxel semantics class labels
            preds (SparseTensor): COO (200x200x16, 3) for coords, (200x200x16, num_classes) for feats

        Returns:
            tensor: cross entropy loss
        """
        loss_ = dict()
        list_pred_coords, list_pred_feats = preds.decomposed_coordinates_and_features
        if self.use_mask:
            # Stack the list of prediction features into a single tensor
            valid_counts = (masked_camera_coo_feats != -100).sum()
            pred_feats_stacked = torch.cat(list_pred_feats, dim=0)
            masked_camera_coo_feats = masked_camera_coo_feats.view(-1).long()
            # Ensure prediction features dtype is correct for loss calculation
            pred_feats_stacked = pred_feats_stacked.float()
            # Calculate the cross-entropy loss
            loss_occ = self.loss_occ(pred_feats_stacked, masked_camera_coo_feats, avg_factor=valid_counts)
            loss_['loss_occ'] = loss_occ
        else:        
            # Stack the list of prediction features into a single tensor
            pred_feats_stacked = torch.cat(list_pred_feats, dim=0)  # Stack along the first dimension (n, num_classes)
            # Ensure voxel_semantics_coo_feats is reshaped to match the flattened predictions
            voxel_semantics_coo_feats = voxel_semantics_coo_feats.view(-1).long()  # Flatten and ensure dtype is long for labels
            # Ensure prediction features dtype is correct for loss calculation
            pred_feats_stacked = pred_feats_stacked.float()  # Convert to float if necessary (for cross-entropy)
            # Calculate the cross-entropy loss
            loss_occ = self.loss_occ(pred_feats_stacked, voxel_semantics_coo_feats)
            loss_['loss_occ'] = loss_occ

        return loss_


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
        voxel_semantics = kwargs['voxel_semantics']
        # mask_camera = kwargs['mask_camera']
        assert voxel_semantics[0].min() >= 0 and voxel_semantics[0].max() <= 17
        
        batch_size = voxel_semantics[0].shape[0]  # 4
        coo_list_gt = []  # List of coordinates (excluding class 17)
        
        for b in range(batch_size):
            current_voxel_grid = voxel_semantics[0][b]
            mask = current_voxel_grid != 17
            coords = torch.argwhere(mask)
            coo_list_gt.append(coords)
        
        voxels, num_points, coors = self.voxelize(points)
        coors[:, 2] = 200 - coors[:, 2]  # Reverse the x direction
        coors = coors[:, [0, 2, 3, 1]]
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        pts_sparse_tensor = ME.SparseTensor(
            features=voxel_features, 
            coordinates=coors, 
            device=voxel_features.device)
        cm = pts_sparse_tensor.coordinate_manager
        target_key, _ = cm.insert_and_map(
                ME.utils.batched_coordinates(coo_list_gt).to(voxel_features.device),
                string_id="target",
        )
        pts_feats = self.lidar_backbone(pts_sparse_tensor)
        out_cls, targets, pts_feat = self.lidar_neck(pts_feats, target_key)


        # convert pts_feat from sparse tensor to 200x200x16 grid format with batch size 1
        pts_coord, pts_feat = pts_feat.decomposed_coordinates_and_features
        pts_coord = pts_coord[0]
        pts_feat = pts_feat[0]
        # print(pts_feat.shape)
        x = pts_coord[:, 0].long()
        y = pts_coord[:, 1].long()
        z = pts_coord[:, 2].long()
        channel_size = pts_feat.shape[1]
        grid = torch.zeros((channel_size, 200, 200, 16), dtype = pts_feat.dtype, device=pts_feat.device)
        grid[:, x, y, z] = pts_feat.t()

        # Step 1: Take softmax over channel 0 (the channel dimension)
        grid_softmax = torch.softmax(grid, dim=0)  # Shape remains (32, 200, 200, 16)

        # Step 2: Take argmax over channel 0
        grid_argmax = grid_softmax.argmax(dim=0)  # Shape becomes (200, 200, 16)
        # make sure argmax is within 0-17
        # grid_argmax = torch.clamp(grid_argmax, 0, 17)

        # Step 3: Convert to uint8 and ensure the shape is (200, 200, 16)
        grid_uint8 = grid_argmax.cpu().numpy().astype(np.uint8)  # Shape: (200, 200, 16)
        
        
        
        
        
        # img_feats, _, _ = self.extract_feat(
        #     points, img=img, img_metas=img_metas, **kwargs)
        # occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)
        # # bncdhw->bnwhdc
        # if self.use_predicter:
        #     occ_pred = self.predicter(occ_pred)
        # occ_score=occ_pred.softmax(-1)
        # occ_res=occ_score.argmax(-1)
        # occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [grid_uint8]

    def pad_lidar_feats(self, pts_feat):
        batch_coords_list, batch_feats_list = pts_feat.decomposed_coordinates_and_features
        padded_coords_list = []
        padded_feats_list = []
        x_min, x_max = 0, 200
        y_min, y_max = 0, 200
        z_min, z_max = 0, 16

        x_size = x_max - x_min  # 200
        y_size = y_max - y_min  # 200
        z_size = z_max - z_min  # 16
        grid_size = x_size * y_size * z_size  # Total number of grid points

        # Step 2: Iterate over each batch independently
        for batch_idx, (batch_valid_coords, batch_valid_feats) in enumerate(zip(batch_coords_list, batch_feats_list)):
            # Apply the filtering mask for valid coordinates
            mask = (batch_valid_coords[:, 0] >= x_min) & (batch_valid_coords[:, 0] < x_max) & \
                (batch_valid_coords[:, 1] >= y_min) & (batch_valid_coords[:, 1] < y_max) & \
                (batch_valid_coords[:, 2] >= z_min) & (batch_valid_coords[:, 2] < z_max)

            # Keep only valid coordinates and features
            batch_valid_coords = batch_valid_coords[mask]
            batch_valid_feats = batch_valid_feats[mask]

            # Compute linear indices of valid coordinates within the grid
            x = batch_valid_coords[:, 0] - x_min
            y = batch_valid_coords[:, 1] - y_min
            z = batch_valid_coords[:, 2] - z_min
            valid_linear_idx = x * (y_size * z_size) + y * z_size + z

            # Create an occupancy mask for all grid points
            occupancy_mask = torch.zeros(grid_size, dtype=torch.bool, device=batch_valid_coords.device)
            occupancy_mask[valid_linear_idx.long()] = True

            # Find missing indices where occupancy is False
            missing_linear_idx = (~occupancy_mask).nonzero(as_tuple=False).squeeze()

            # Convert missing linear indices back to 3D coordinates
            x_missing = missing_linear_idx // (y_size * z_size)
            remainder = missing_linear_idx % (y_size * z_size)
            y_missing = remainder // z_size
            z_missing = remainder % z_size
            missing_coords = torch.stack([x_missing + x_min, y_missing + y_min, z_missing + z_min], dim=1).int()

            # Create zero feature vectors for the missing coordinates
            zero_feats = torch.zeros((missing_coords.shape[0], batch_valid_feats.shape[1]),
                                    dtype=batch_valid_feats.dtype, device=batch_valid_feats.device)

            # Concatenate the valid and missing coordinates and features
            batch_padded_coords = torch.cat([batch_valid_coords, missing_coords], dim=0)
            batch_padded_feats = torch.cat([batch_valid_feats, zero_feats], dim=0)

            # Add the batch index to the coordinates for MinkowskiEngine compatibility
            batch_indices = torch.full((batch_padded_coords.shape[0], 1), batch_idx,
                                    dtype=torch.int32, device=batch_valid_coords.device)
            batch_padded_coords_with_idx = torch.cat([batch_indices, batch_padded_coords], dim=1)

            # Append the results to the list
            padded_coords_list.append(batch_padded_coords_with_idx)
            padded_feats_list.append(batch_padded_feats)

        # Step 3: Concatenate the results across all batches
        all_padded_coords = torch.cat(padded_coords_list, dim=0)
        all_padded_feats = torch.cat(padded_feats_list, dim=0)

        # Step 4: Create a new sparse tensor with the padded coordinates and features
        pts_feats_padded = ME.SparseTensor(
            features=all_padded_feats,
            coordinates=all_padded_coords,
            device = pts_feat.device
        )
        
        return pts_feats_padded
        
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

        # Clone voxel_semantics and apply mask for camera, set class 18 where mask_camera == 0
        masked_semantics = voxel_semantics.clone().to(torch.int8)
        masked_semantics[mask_camera == 0] = -100  # Set class to -100 where mask_camera == 0

        
        # valid_counts = (masked_semantics != -100).sum()
        # print(f"Number of valid voxels: {valid_counts}")
        batch_size = voxel_semantics.shape[0]  # 4

        # Initialize lists to hold the results for each batch
        coo_list_gt = []  # List of coordinates (excluding class 17)
        voxel_semantics_coo_feats = []  # List of features for voxel semantics (corresponding to all coordinates, no mask)
        mask_camera_coo_feats = []  # List of features for mask_camera (only features for valid coords)

        for b in range(batch_size):
            # Process voxel semantics for the current batch
            current_voxel_grid = voxel_semantics[b]
            
            # Mask to remove class 17 and get valid coordinates
            mask = current_voxel_grid != 17
            coords = torch.argwhere(mask)  # (n, 3), get coordinates where class is not 17
            
            # Store valid coordinates in coo_list_gt
            coo_list_gt.append(coords)
            
            # For voxel semantics features, reshape into (200*200*16, 1) and store all features
            all_features = current_voxel_grid.view(-1, 1)  # (200*200*16, 1)
            voxel_semantics_coo_feats.append(all_features)  # Store the full voxel features regardless of mask

            # Now process the mask_camera
            current_masked_semantics = masked_semantics[b]            
            mask_features = current_masked_semantics.view(-1, 1)  # Reshape all features into (200*200*16, 1)
            mask_camera_coo_feats.append(mask_features)  # Store only valid features for mask_camera

        
        # stack voxel_semantics_coo_feats
        voxel_semantics_coo_feats = torch.vstack(voxel_semantics_coo_feats)
        mask_camera_coo_feats = torch.vstack(mask_camera_coo_feats)
        

        
        # Final results:
        # coo_list_gt: Coordinates excluding class 17 (voxel_semantics)
        # voxel_semantics_coo_feats: Full features for voxel semantics (without masking)
        # mask_camera_coo_feats: Full features for mask_camera (-100 for invalid voxels)

        # voxelization of pointcloud
        voxels, num_points, coors = self.voxelize(points)
        coors[:, 2] = 200 - coors[:, 2]  # Reverse the x direction
        # move b,z,x,y,to b, x, y,z
        coors = coors[:, [0, 2, 3, 1]]
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
         # try to visualize the pts_feat in .ply form
       
        # Create minkowski sparse tensor using coors and voxel_features
        pts_sparse_tensor = ME.SparseTensor(
            features=voxel_features, 
            coordinates=coors, 
            device=voxel_features.device)
        cm = pts_sparse_tensor.coordinate_manager
        target_key, _ = cm.insert_and_map(
                ME.utils.batched_coordinates(coo_list_gt).to(voxel_features.device),
                string_id="target",
        )
        # pass the sparse tensor through the lidar backbone
        pts_feats = self.lidar_backbone(pts_sparse_tensor)
        
        # pass the pts_feats through the lidar neck
        out_cls, targets, pts_feat = self.lidar_neck(pts_feats, target_key)

        #  BCE Loss calculation for scene completion point existence
        bce_loss = self.lidar_neck.get_bce_loss(out_cls, targets)
        losses['loss_bce'] = bce_loss

        # return losses
        # constrain pts_feats to within 200x200x16 grid (self.COO_format_coords is the COO format of this) 
        pts_feat = self.pad_lidar_feats(pts_feat)
        # pts_feat is a sparse tensor
        # get the coordinates and features
        # pts_feat_coords, pts_feat_feats = pts_feat.decomposed_coordinates_and_features
        
        
        # input_coords = pts_feat_coords[0].cpu().numpy()
        
        # from plyfile import PlyData, PlyElement
    
        # # Prepare the data for PLY file (only coordinates)
        # vertex_data = [(x, y, z) for x, y, z in input_coords]

        # # Define the PLY elements (only x, y, z coordinates)
        # vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

        # # Create the PLY structure
        # ply_elements = PlyElement.describe(np.array(vertex_data, dtype=vertex_dtype), 'vertex')

        # # Save to a PLY file
        # ply_data = PlyData([ply_elements], text=True)
        # ply_data.write('pts_image.ply')
        
        
        # ppp
       
        img_feats, _, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        
        # img feats in list[(b, c, z_voxel, x_voxel, y_voxel)] where z x and y are fixed at 16x200x200
        # c is given by cfg (prev frame + current_frame if prev_frame is used)
        # depth in (b * ncams, depth_bins, x_resized, y_resized) 
        # convert img feats to sparse tensor format to be fused with pts_feat
        # pts feat COO (n, 3) coords and (n, 32) feats 
        channels = img_feats.shape[1]    # number of channels

        # coo_list = [self.COO_format_coords for _ in range(batch_size)]  # Create a list where each entry is the same `coords` tensor
        feats_list = []
        for b in range(batch_size):
            current_voxel_grid = img_feats[b]  # shape (c, z_voxel, x_voxel, y_voxel)
            feats = current_voxel_grid.view(channels, -1).transpose(0, 1)  # shape (num_voxels, c)
            feats_list.append(feats)

        # Stack features vertically over all batches (num_voxels * batch_size, c)
        stacked_feats = torch.vstack(feats_list)
        
        # create sparse tensor
        img_sparse_tensor = ME.SparseTensor(
            features=stacked_feats, 
            # coordinates=ME.utils.batched_coordinates(coo_list), 
            device=stacked_feats.device,
            coordinate_manager=pts_feat.coordinate_manager,
            coordinate_map_key=pts_feat.coordinate_map_key
        )

        fused_feats = self.sparse_fusion(img_sparse_tensor, pts_feat)
            
        

        
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth
        
        # pass fused feats through Mink-ResUnet for 3D semantic segmentation
        occ_preds = self.occ_backbone(fused_feats)
        _, _, occ_pred = self.occ_neck(occ_preds)
        

        # occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
            

        loss_occ = self.loss_single(voxel_semantics_coo_feats, mask_camera_coo_feats, occ_pred)
        losses.update(loss_occ)
        return losses
