# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from os import path as osp

import mmcv
import numpy as np
import torch
import cv2
from mmcv.utils import print_log
from tqdm import tqdm

from ..core.bbox import Box3DMode, points_cam2img
from .builder import DATASETS
from .waymo_dataset import WaymoDataset
from .occ_metrics import Metric_mIoU, Metric_FScore


@DATASETS.register_module()
class WaymoDatasetOccupancy(WaymoDataset):
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        input_dict = super(WaymoDatasetOccupancy, self).get_data_info(index)
        
        # standard protocol modified from SECOND.Pytorch
        input_dict['occ_gt_path'] = self.data_infos[index]['occ_gt_path']

        return input_dict
    
    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=16,
            use_lidar_mask=False,
            use_image_mask=True)
        show_dir = './vis/' if show_dir is None else show_dir
        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            occ_gt_path = info['occ_gt_path']
        # add ./waymo/kitti_format/ to occ_gt_path
            occ_gt = os.path.join('data/waymo/kitti_format/', occ_gt_path)
            occ_gt = np.load(occ_gt)

            gt_semantics = occ_gt['voxel_label']
            mask_infov = occ_gt['infov'].astype(bool)
            mask_lidar = occ_gt['origin_voxel_state'].astype(bool)
            mask_camera = occ_gt['final_voxel_state'].astype(bool)
            
            mask = np.ones_like(gt_semantics).astype(bool) # 200, 200, 16
            
            mask_camera = (mask & mask_camera).astype(bool)
            mask_lidar = (mask & mask_lidar).astype(bool)

            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)
            
            if index%1==0 and show_dir is not None:
                gt_vis = self.vis_occ(gt_semantics)
                pred_vis = self.vis_occ(occ_pred)
                mmcv.imwrite(np.concatenate([gt_vis, pred_vis], axis=1),
                             os.path.join(show_dir + "%d.jpg"%index))

        return self.occ_eval_metrics.count_miou()
    
    def vis_occ(self, semantics):
        # simple visualization of result in BEV
        semantics_valid = np.logical_not(semantics == 15)
        d = np.arange(14).reshape(1, 1, 14)
        d = np.repeat(d, 200, axis=0)
        d = np.repeat(d, 200, axis=1).astype(np.float32)
        d = d * semantics_valid
        selected = np.argmax(d, axis=2)

        selected_torch = torch.from_numpy(selected)
        semantics_torch = torch.from_numpy(semantics)

        occ_bev_torch = torch.gather(semantics_torch, dim=2,
                                     index=selected_torch.unsqueeze(-1))
        occ_bev = occ_bev_torch.numpy()

        occ_bev = occ_bev.flatten().astype(np.int32)
        occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(200, 200, 4)[::-1, ::-1, :3]
        occ_bev_vis = cv2.resize(occ_bev_vis,(400,400))
        return occ_bev_vis