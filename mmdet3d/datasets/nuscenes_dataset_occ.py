# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm

from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .occ_metrics import (
    Metric_mIoU,
    Metric_FScore,
    Metric_PredictionAccuracy,
    Metric_RayIoU,
)
from pyquaternion import Quaternion 
from torch.utils.data import Dataset, DataLoader\

colors_map = np.array(
    [
        [0, 0, 0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],  # 2 pedestrian  Blue
        [47, 79, 79, 255],  # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],  # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255],  # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],  # 9 motorcycle  Slategrey
        [222, 184, 135, 255],  # 10 building Burlywood
        [0, 175, 0, 255],  # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255],  # 14 walkable, sidewalk
        [255, 0, 0, 255],  # 15 unobsrvd
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 17 undefined
    ]
)

# Define the semantic segmentation colormap
class_colors_3d_vis = {
    0: [255, 0, 0],
    1: [0, 255, 0],
    2: [0, 0, 255],
    3: [255, 255, 0],
    4: [0, 255, 255],
    5: [255, 0, 255],
    6: [128, 128, 0],
    7: [0, 128, 128],
    8: [128, 0, 128],
    9: [255, 128, 0],
    10: [0, 128, 255],
    11: [128, 0, 255],
    12: [128, 255, 0],
    13: [255, 0, 128],
    14: [0, 255, 128],
    15: [128, 128, 128],
    16: [64, 64, 64],
}

# import EgoPoseDataset for RayIoU evaluation
def trans_matrix(T, R):
    tm = np.eye(4)
    tm[:3, :3] = R.rotation_matrix
    tm[:3, 3] = T
    return tm


# A helper dataset for RayIoU. It is NOT used during training.
class EgoPoseDataset(Dataset):
    def __init__(self, data_infos):
        super(EgoPoseDataset, self).__init__()

        self.data_infos = data_infos
        self.scene_frames = {}

        for info in data_infos:
            # Check if 'scene_name' exists, otherwise use 'scene_token'
            scene_key = info.get('scene_name', info.get('scene_token'))
            if scene_key not in self.scene_frames:
                self.scene_frames[scene_key] = []
            self.scene_frames[scene_key].append(info)

    def __len__(self):
        return len(self.data_infos)

    def get_ego_from_lidar(self, info):
        ego_from_lidar = trans_matrix(
            np.array(info['lidar2ego_translation']), 
            Quaternion(info['lidar2ego_rotation']))
        return ego_from_lidar

    def get_global_pose(self, info, inverse=False):
        global_from_ego = trans_matrix(
            np.array(info['ego2global_translation']), 
            Quaternion(info['ego2global_rotation']))
        ego_from_lidar = trans_matrix(
            np.array(info['lidar2ego_translation']), 
            Quaternion(info['lidar2ego_rotation']))
        pose = global_from_ego.dot(ego_from_lidar)
        if inverse:
            pose = np.linalg.inv(pose)
        return pose

    def __getitem__(self, idx):
        info = self.data_infos[idx]
        ref_sample_token = info['token']
        ref_lidar_from_global = self.get_global_pose(info, inverse=True)
        ref_ego_from_lidar = self.get_ego_from_lidar(info)

        scene_key = info.get('scene_name', info.get('scene_token'))
        scene_frame = self.scene_frames[scene_key]
        
        # Find the index of the current info dict in the scene_frame list
        ref_index = -1
        for i, frame_info in enumerate(scene_frame):
            if frame_info['token'] == info['token']:
                ref_index = i
                break
        
        # NOTE: getting output frames
        output_origin_list = []
        for curr_index in range(len(scene_frame)):
            if curr_index == ref_index:
                origin_tf = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else:
                global_from_curr = self.get_global_pose(scene_frame[curr_index], inverse=False)
                ref_from_curr = ref_lidar_from_global.dot(global_from_curr)
                origin_tf = np.array(ref_from_curr[:3, 3], dtype=np.float32)

            origin_tf_pad = np.ones([4])
            origin_tf_pad[:3] = origin_tf
            origin_tf = np.dot(ref_ego_from_lidar[:3], origin_tf_pad.T).T

            if np.abs(origin_tf[0]) < 39 and np.abs(origin_tf[1]) < 39:
                output_origin_list.append(origin_tf)
        
        if len(output_origin_list) > 8:
            select_idx = np.round(np.linspace(0, len(output_origin_list) - 1, 8)).astype(np.int64)
            output_origin_list = [output_origin_list[i] for i in select_idx]

        output_origin_tensor = torch.from_numpy(np.stack(output_origin_list))

        return (ref_sample_token, output_origin_tensor)

@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):
    def __init__(
        self,
        strong_supervision_ratio=1.0,  # Add new argument with a default
        *args,
        **kwargs,
    ):
        # Call the parent class's constructor with all original arguments
        super(NuScenesDatasetOccpancy, self).__init__(*args, **kwargs)

        self.strong_supervision_ratio = strong_supervision_ratio
        # Calculate how many samples are considered "strong"
        if not self.test_mode:
            self.strong_sample_count = int(
                len(self.data_infos) * self.strong_supervision_ratio
            )
            print(
                f"\nINFO: Using the first {self.strong_sample_count} samples "
                f"({self.strong_supervision_ratio * 100:.1f}%) for strong supervision.\n"
            )
        else:
            self.strong_sample_count = 0  # No strong supervision in test mode

    def __getitem__(self, idx):
        """
        Get item from infos and add a supervision flag.
        """
        # Get the data dictionary from the parent class's __getitem__
        # This will call NuScenesDataset -> Custom3DDataset -> __getitem__
        data = super(NuScenesDataset, self).__getitem__(idx)

        # Add the supervision flag to the metadata
        # The `data` can be None if filtering is active, so we check for it
        if data is not None:
            is_strong = (idx < self.strong_sample_count) and not self.test_mode
            data["img_metas"]._data["is_strong_supervision"] = is_strong

        return data

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
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        # standard protocol modified from SECOND.Pytorch
        input_dict["occ_gt_path"] = self.data_infos[index]["occ_path"]
        # input_dict['lidarseg_gt_path'] = self.data_infos[index]['lidarseg_path']
        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        metric_type = eval_kwargs.get("metric_type", "miou")
        print(f"\n[EVALUATION] Using metric: {metric_type.upper()}")

        metric_params = {
            "num_classes": 18,
            "use_lidar_mask": False,
            "use_image_mask": True,  # Typically, evaluation is done within the camera frustum
        }

        # Initialize the chosen metric
        if metric_type.lower() == "miou":
            self.occ_eval_metrics = Metric_mIoU(**metric_params)
        elif metric_type.lower() == "pred_acc":
            self.occ_eval_metrics = Metric_PredictionAccuracy(**metric_params)
        elif metric_type.lower() == "rayiou":
            self.occ_eval_metrics = Metric_RayIoU(**metric_params)
            ego_dataset = EgoPoseDataset(self.data_infos)
            ego_dataloader = DataLoader(ego_dataset, batch_size=1, num_workers=2)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

        if show_dir:
            os.makedirs(show_dir, exist_ok=True)
            print(f"[VISUALIZATION] Saving output visualizations to: {show_dir}")

        # --- Main Evaluation Loop ---
        if metric_type.lower() == "rayiou":
            # Special loop for RayIoU that requires ego poses
            iterator = zip(tqdm(occ_results, desc="Evaluating RayIoU"), ego_dataloader)
            for index, (occ_pred, (token, output_origin)) in enumerate(iterator):
                info = self.data_infos[index]
                try:
                    occ_gt_data = np.load(os.path.join(info["occ_path"], "labels.npz"))
                except FileNotFoundError:
                    print(
                        f"PITFALL: Ground truth file not found for sample index {index}. Skipping."
                    )
                    continue

                gt_semantics = occ_gt_data["semantics"]
                mask_lidar = occ_gt_data["mask_lidar"].astype(bool)
                mask_camera = occ_gt_data["mask_camera"].astype(bool)

                self.occ_eval_metrics.add_batch(
                    semantics_pred=occ_pred,
                    semantics_gt=gt_semantics,
                    mask_lidar=mask_lidar,
                    mask_camera=mask_camera,
                    output_origin=output_origin,
                )

                # Visualization (optional)
                if show_dir and index % 10 == 0:  # Visualize every 10 samples
                    self.vis_occ_3d(gt_semantics, show_dir, index, "gt")
                    self.vis_occ_3d(occ_pred, show_dir, index, "pred")
        else:
            # Standard loop for mIoU and Prediction Accuracy
            for index, occ_pred in enumerate(
                tqdm(occ_results, desc=f"Evaluating {metric_type.upper()}")
            ):
                info = self.data_infos[index]
                try:
                    occ_gt_data = np.load(os.path.join(info["occ_path"], "labels.npz"))
                except FileNotFoundError:
                    print(
                        f"PITFALL: Ground truth file not found for sample index {index}. Skipping."
                    )
                    continue

                gt_semantics = occ_gt_data["semantics"]
                mask_lidar = occ_gt_data["mask_lidar"].astype(bool)
                mask_camera = occ_gt_data["mask_camera"].astype(bool)

                # DOUBLE CHECK: The shapes of your predictions and ground truth must match.
                # pred_shape = occ_pred.shape
                # gt_shape = gt_semantics.shape
                # if pred_shape != gt_shape:
                #     print(f"PITFALL: Shape mismatch at index {index}. Pred: {pred_shape}, GT: {gt_shape}")
                #     continue

                self.occ_eval_metrics.add_batch(
                    semantics_pred=occ_pred,
                    semantics_gt=gt_semantics,
                    mask_lidar=mask_lidar,
                    mask_camera=mask_camera,
                )

                # Visualization (optional)
                if show_dir:  # Visualize every 10 samples
                    print(f"\n[VISUALIZATION] Saving sample index {index}...")
                    self.vis_occ_3d(gt_semantics, show_dir, index, "gt")
                    self.vis_occ_3d(occ_pred, show_dir, index, "pred")

        # --- Final Metric Calculation and Reporting ---
        if metric_type.lower() == "miou":
            return self.occ_eval_metrics.count_miou()
        elif metric_type.lower() == "pred_acc":
            return self.occ_eval_metrics.count_accuracy()
        elif metric_type.lower() == "rayiou":
            return self.occ_eval_metrics.count_rayiou()

    # --- MODIFICATION END ---

    def vis_occ_3d(self, semantics, show_dir, index, prefix):
        """
        Takes in a 200x200x16 grid and saves a 3D visualization as a .ply file.

        Args:
            semantics (numpy.ndarray): 200x200x16 grid containing class indices.
            show_dir (str): Directory to save the .ply file.
            index (int): Index for naming the output file.
            prefix (str): Prefix for distinguishing file types ("gt" or "pred").

        Returns:
            str: The path of the saved .ply file.
        """
        # Ensure the save directory exists
        os.makedirs(show_dir, exist_ok=True)

        # Get the coordinates where the class is not 17 (valid points)
        valid_mask = semantics != 17
        coords = np.array(np.nonzero(valid_mask)).T  # Shape: (n, 3)
        
        # Handle the case where there are no valid points to visualize
        if coords.shape[0] == 0:
            print(f"Warning: No valid points to visualize for {prefix}_{index}.ply. Skipping.")
            return None
            
        classes = semantics[valid_mask]  # Shape: (n,)

        # Get the colors corresponding to each class
        # Add a check for keys that might be missing from the color map
        colors = np.array([class_colors_3d_vis.get(c, [0, 0, 0]) for c in classes])  # Shape: (n, 3)

        # Prepare data to write to the .ply file
        num_points = coords.shape[0]

        # Create the header for the .ply file
        ply_header = f"""ply
            format ascii 1.0
            element vertex {num_points}
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
            """

        # Generate the output file path using the prefix to distinguish files
        output_path = os.path.join(show_dir, f"{prefix}_{index}.ply")
        
        # Write the data to the .ply file
        with open(output_path, "w") as f:
            f.write(ply_header)
            for (x, y, z), color in zip(coords, colors):
                f.write(f"{x} {y} {z} {color[0]} {color[1]} {color[2]}\n")

        return output_path

    def vis_occ_bev(self, semantics):
        # simple visualization of result in BEV
        semantics_valid = np.logical_not(semantics == 17)
        d = np.arange(16).reshape(1, 1, 16)
        d = np.repeat(d, 200, axis=0)
        d = np.repeat(d, 200, axis=1).astype(np.float32)
        d = d * semantics_valid
        selected = np.argmax(d, axis=2)

        selected_torch = torch.from_numpy(selected)
        semantics_torch = torch.from_numpy(semantics)

        occ_bev_torch = torch.gather(
            semantics_torch, dim=2, index=selected_torch.unsqueeze(-1)
        )
        occ_bev = occ_bev_torch.numpy()

        occ_bev = occ_bev.flatten().astype(np.int32)
        occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(200, 200, 4)[::-1, ::-1, :3]
        occ_bev_vis = cv2.resize(occ_bev_vis, (400, 400))
        return occ_bev_vis
