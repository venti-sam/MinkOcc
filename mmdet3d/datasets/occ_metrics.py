import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import pickle as pkl
import argparse
import time
import torch
import sys, platform
import math
from sklearn.neighbors import KDTree
from termcolor import colored
from pathlib import Path
from copy import deepcopy
from functools import reduce
from torch.utils.cpp_extension import load

np.seterr(divide="ignore", invalid="ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pcolor(string, color, on_color=None, attrs=None):
    """
    Produces a colored string for printing

    Parameters
    ----------
    string : str
        String that will be colored
    color : str
        Color to use
    on_color : str
        Background color to use
    attrs : list of str
        Different attributes for the string

    Returns
    -------
    string: str
        Colored string
    """
    return colored(string, color, on_color, attrs)


def getCellCoordinates(points, voxelSize):
    return (points / voxelSize).astype(np.int)


def getNumUniqueCells(cells):
    M = cells.max() + 1
    return np.unique(cells[:, 0] + M * cells[:, 1] + M**2 * cells[:, 2]).shape[0]


class BaseMetric:
    """
    Base class for all occupancy metrics. Handles common initialization and mask application.
    """

    def __init__(
        self, save_dir=".", num_classes=18, use_lidar_mask=False, use_image_mask=False
    ):
        # Class names for semantic labels
        self.class_names = [
            "others",
            "barrier",
            "bicycle",
            "bus",
            "car",
            "construction_vehicle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "trailer",
            "truck",
            "driveable_surface",
            "other_flat",
            "sidewalk",
            "terrain",
            "manmade",
            "vegetation",
            "free",
        ]

        # Configuration parameters
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes
        self.cnt = 0

        # Point cloud and voxel parameters
        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4

        # Calculate dimensions
        self.occ_xdim = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0])
            / self.occupancy_size[0]
        )
        self.occ_ydim = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1])
            / self.occupancy_size[1]
        )
        self.occ_zdim = int(
            (self.point_cloud_range[5] - self.point_cloud_range[2])
            / self.occupancy_size[2]
        )
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim

    def _apply_masks(self, semantics_pred, semantics_gt, mask_lidar, mask_camera):
        """
        Apply visibility masks while preserving 3D structure.
        All inputs should be shape (200, 200, 16)
        """
        if self.use_image_mask:
            masked_gt = semantics_gt.copy()
            masked_pred = semantics_pred.copy()
            masked_gt[~mask_camera] = self.num_classes - 1
            # masked_pred[~mask_camera] = self.num_classes - 1
            return masked_gt, masked_pred
        elif self.use_lidar_mask:
            masked_gt = semantics_gt.copy()
            masked_pred = semantics_pred.copy()
            masked_gt[~mask_lidar] = self.num_classes - 1
            # masked_pred[~mask_lidar] = self.num_classes - 1
            return masked_gt, masked_pred
        return semantics_gt, semantics_pred


class Metric_mIoU(BaseMetric):
    """
    Mean Intersection over Union (mIoU) metric for semantic occupancy evaluation.
    Considers all voxels when computing the metric.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hist = np.zeros((self.num_classes, self.num_classes))

    def hist_info(self, n_cl, pred, gt):
        """Build confusion matrix for IoU calculation"""
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # Exclude invalid labels
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        # Create confusion matrix
        hist = np.bincount(
            n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl**2
        ).reshape(n_cl, n_cl)

        return hist, correct, labeled

    def per_class_iu(self, hist):
        """Calculate per-class IoU from confusion matrix"""
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        """Compute mean IoU for a single batch"""
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(
            n_classes, pred.flatten(), label.flatten()
        )
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        return round(np.nanmean(mIoUs) * 100, 2), hist

    def add_batch(self, semantics_pred, semantics_gt, mask_lidar, mask_camera):
        """Process a batch of predictions"""
        self.cnt += 1
        masked_semantics_gt, masked_semantics_pred = self._apply_masks(
            semantics_pred, semantics_gt, mask_lidar, mask_camera
        )
        _, _hist = self.compute_mIoU(
            masked_semantics_pred, masked_semantics_gt, self.num_classes
        )
        self.hist += _hist

    def count_miou(self):
        """Calculate final mIoU results"""
        mIoU = self.per_class_iu(self.hist)
        print(f"===> per class IoU of {self.cnt} samples:")
        for ind_class in range(self.num_classes - 1):
            print(
                f"===> {self.class_names[ind_class]} - IoU = "
                + str(round(mIoU[ind_class] * 100, 2))
            )
        print(
            f"===> mIoU of {self.cnt} samples: "
            + str(round(np.nanmean(mIoU[: self.num_classes - 1]) * 100, 2))
        )
        return self.class_names, mIoU, self.cnt


class Metric_PredictionAccuracy(BaseMetric):
    """
    Prediction Accuracy metric that only considers predicted occupied voxels.
    Focuses on the accuracy of positive predictions without penalizing false negatives.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hist = np.zeros((self.num_classes, self.num_classes))

    def compute_prediction_accuracy(self, pred, gt):
        """Compute accuracy only for predicted occupied voxels"""
        # Only consider non-free predictions
        pred_occupied = pred != self.num_classes - 1
        valid_pred = pred[pred_occupied]
        corresponding_gt = gt[pred_occupied]

        # Initialize confusion matrix
        hist = np.zeros((self.num_classes, self.num_classes))

        # Build confusion matrix for predicted voxels
        for pred_class in range(self.num_classes):
            pred_mask = valid_pred == pred_class
            if not np.any(pred_mask):
                continue
            for gt_class in range(self.num_classes):
                hist[pred_class, gt_class] = np.sum(
                    (valid_pred == pred_class) & (corresponding_gt == gt_class)
                )

        per_class_acc = np.diag(hist) / (hist.sum(axis=1) + 1e-6)
        overall_acc = np.sum(np.diag(hist)) / (np.sum(hist) + 1e-6)

        return per_class_acc, overall_acc, hist

    def add_batch(self, semantics_pred, semantics_gt, mask_lidar, mask_camera):
        """Process a batch of predictions"""
        self.cnt += 1
        # print(semantics_pred.shape, semantics_gt.shape)
        # print(mask_lidar.shape, mask_camera.shape)
        masked_semantics_gt, masked_semantics_pred = self._apply_masks(
            semantics_pred, semantics_gt, mask_lidar, mask_camera
        )
        # print(masked_semantics_pred.shape, masked_semantics_gt.shape)
        _, _, batch_hist = self.compute_prediction_accuracy(
            masked_semantics_pred, masked_semantics_gt
        )
        self.hist += batch_hist

    def count_accuracy(self):
        """Calculate final accuracy results"""
        per_class_acc = np.diag(self.hist) / (self.hist.sum(axis=1) + 1e-6)
        overall_acc = np.sum(np.diag(self.hist)) / (np.sum(self.hist) + 1e-6)

        print(f"===> Per class accuracy of {self.cnt} samples:")
        for ind_class in range(self.num_classes - 1):
            print(
                f"===> {self.class_names[ind_class]} - Accuracy = "
                + str(round(per_class_acc[ind_class] * 100, 2))
            )
        print(
            f"===> Overall accuracy of {self.cnt} samples: "
            + str(round(overall_acc * 100, 2))
        )

        return self.class_names, per_class_acc, overall_acc, self.cnt


# extra imports:
# dvr, which is included as a lib in main folder
dvr = load(
    "dvr",
    sources=["lib/dvr/dvr.cpp", "lib/dvr/dvr.cu"],
    verbose=True,
    extra_cuda_cflags=["-allow-unsupported-compiler"],
)
# Constants for voxel grid configuration
_pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
_voxel_size = 0.4


class Metric_RayIoU(BaseMetric):
    """
    RayIoU metric evaluates 3D semantic predictions by simulating LiDAR ray casting.
    Compares predicted and ground truth ray intersections at multiple distance thresholds.
    """

    def __init__(
        self, save_dir=".", num_classes=18, use_lidar_mask=False, use_image_mask=False
    ):
        super().__init__(save_dir, num_classes, use_lidar_mask, use_image_mask)

        # Distance thresholds for evaluation (in meters)
        self.thresholds = [1, 2, 4]

        # Storage for evaluation results
        self.pcd_pred_list = []
        self.pcd_gt_list = []

        # Generate LiDAR ray patterns once during initialization
        self.lidar_rays = self._generate_lidar_rays()

    def _generate_lidar_rays(self):
        """Generate ray directions simulating a LiDAR sensor's scanning pattern"""
        # Calculate pitch angles
        pitch_angles = []
        for k in range(10):
            angle = math.pi / 2 - math.atan(k + 1)
            pitch_angles.append(-angle)

        # Add upward angles up to NuScenes LiDAR FOV limit
        while pitch_angles[-1] < 0.21:
            delta = pitch_angles[-1] - pitch_angles[-2]
            pitch_angles.append(pitch_angles[-1] + delta)

        # Generate rays for each angle combination
        lidar_rays = []
        for pitch_angle in pitch_angles:
            for azimuth_angle in np.arange(0, 360, 1):
                azimuth_angle = np.deg2rad(azimuth_angle)
                x = np.cos(pitch_angle) * np.cos(azimuth_angle)
                y = np.cos(pitch_angle) * np.sin(azimuth_angle)
                z = np.sin(pitch_angle)
                lidar_rays.append((x, y, z))

        return torch.tensor(lidar_rays, dtype=torch.float32)

    def _process_sample(self, semantics, output_origin):
        """
        Process a single sample through ray casting and return point cloud with semantic labels.

        Args:
            semantics: Semantic predictions/ground truth (shape: [200, 200, 16])
            output_origin: LiDAR sensor positions (shape: [1, T, 3])
        """
        T = output_origin.shape[1]  # Number of time steps (8)
        pred_pcds_t = []

        # Convert semantic grid to occupancy grid
        sem = (
            torch.from_numpy(semantics)
            if isinstance(semantics, np.ndarray)
            else semantics
        )
        occ_pred = deepcopy(sem)
        free_id = self.num_classes - 1
        occ_pred[sem < free_id] = 1
        occ_pred[sem == free_id] = 0

        # Prepare grid for DVR module
        occ_pred = occ_pred.permute(2, 1, 0)
        occ_pred = occ_pred[None, None, :].contiguous().float()

        # Setup coordinate transformation
        offset = torch.Tensor(_pc_range[:3])[None, None, :]
        scaler = torch.Tensor([_voxel_size] * 3)[None, None, :]

        # Process each time step
        for t in range(T):
            # Get LiDAR position for current time step
            lidar_origin = output_origin[:, t : t + 1, :]  # [1, 1, 3]

            # Expand rays to match batch dimension
            rays_batch = self.lidar_rays[None, :, :]  # [1, 14040, 3]

            lidar_endpts = rays_batch + lidar_origin  # Broadcasting: [1, 14040, 3]

            # Transform to voxel grid coordinates
            output_origin_render = ((lidar_origin - offset) / scaler).float()
            output_points_render = ((lidar_endpts - offset) / scaler).float()

            # Create time index tensor of correct shape
            lidar_tindex = torch.zeros([1, self.lidar_rays.shape[0]])

            # Perform ray casting
            with torch.no_grad():
                pred_dist, _, coord_index = dvr.render_forward(
                    occ_pred.cuda(),
                    output_origin_render.cuda(),
                    output_points_render.cuda(),
                    lidar_tindex.cuda(),
                    [1, 16, 200, 200],
                    "test",
                )
                pred_dist *= _voxel_size

            # Extract semantic labels at intersection points
            coord_index = coord_index[0].long().cpu()
            sem_label = sem[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]][
                :, None
            ]
            pred_dist = pred_dist[0, :, None].cpu()

            # Combine semantic labels with distances
            pcds = torch.cat([sem_label.float(), pred_dist], dim=-1)
            pred_pcds_t.append(pcds)

        return torch.cat(pred_pcds_t, dim=0)

    def add_batch(
        self, semantics_pred, semantics_gt, mask_lidar, mask_camera, output_origin
    ):
        """Add a batch of predictions and ground truth for evaluation"""
        self.cnt += 1

        # Apply visibility masks
        masked_semantics_gt, masked_semantics_pred = self._apply_masks(
            semantics_pred, semantics_gt, mask_lidar, mask_camera
        )

        # Process both prediction and ground truth
        pcd_pred = self._process_sample(masked_semantics_pred, output_origin)
        pcd_gt = self._process_sample(masked_semantics_gt, output_origin)

        # Filter out free space rays
        valid_mask = pcd_gt[:, 0].int() != self.num_classes - 1
        pcd_pred = pcd_pred[valid_mask]
        pcd_gt = pcd_gt[valid_mask]

        # Store results for this batch
        self.pcd_pred_list.append(pcd_pred)
        self.pcd_gt_list.append(pcd_gt)

    def _calc_rayiou(self):
        """Calculate RayIoU metrics across all processed samples"""
        gt_cnt = np.zeros([self.num_classes])
        pred_cnt = np.zeros([self.num_classes])
        tp_cnt = np.zeros([len(self.thresholds), self.num_classes])

        for pcd_pred, pcd_gt in zip(self.pcd_pred_list, self.pcd_gt_list):
            for j, threshold in enumerate(self.thresholds):
                # Calculate L1 distance error
                depth_pred = pcd_pred[:, 1]
                depth_gt = pcd_gt[:, 1]
                l1_error = torch.abs(depth_pred - depth_gt)
                tp_dist_mask = l1_error < threshold

                # Calculate per-class metrics
                for i in range(self.num_classes - 1):
                    cls_mask_pred = pcd_pred[:, 0] == i
                    cls_mask_gt = pcd_gt[:, 0] == i

                    if j == 0:
                        gt_cnt[i] += cls_mask_gt.sum().item()
                        pred_cnt[i] += cls_mask_pred.sum().item()

                    tp_cls = cls_mask_gt & cls_mask_pred
                    tp_mask = tp_cls & tp_dist_mask
                    tp_cnt[j][i] += tp_mask.sum().item()

        # Calculate IoU for each threshold
        iou_list = []
        for j in range(len(self.thresholds)):
            iou = tp_cnt[j] / (gt_cnt + pred_cnt - tp_cnt[j])
            iou_list.append(iou[:-1])  # Exclude free class

        return iou_list

    def count_rayiou(self):
        """Calculate and report final RayIoU metrics"""
        iou_list = self._calc_rayiou()
        rayiou = np.nanmean(iou_list)
        rayiou_by_threshold = [
            np.nanmean(iou_list[i]) for i in range(len(self.thresholds))
        ]

        print(f"\n===> RayIoU results for {self.cnt} samples:")
        for i in range(self.num_classes - 1):
            print(f"===> {self.class_names[i]}:")
            for j, threshold in enumerate(self.thresholds):
                print(f"    RayIoU@{threshold}m = {iou_list[j][i]:.3f}")

        print(f"\n===> Mean RayIoU = {rayiou:.3f}")
        for j, threshold in enumerate(self.thresholds):
            print(f"===> Mean RayIoU@{threshold}m = {rayiou_by_threshold[j]:.3f}")

        return self.class_names, iou_list, rayiou_by_threshold, self.cnt


class Metric_FScore:
    def __init__(
        self,
        leaf_size=10,
        threshold_acc=0.6,
        threshold_complete=0.6,
        voxel_size=[0.4, 0.4, 0.4],
        range=[-40, -40, -1, 40, 40, 5.4],
        void=[17, 255],
        use_lidar_mask=False,
        use_image_mask=False,
    ) -> None:

        self.leaf_size = leaf_size
        self.threshold_acc = threshold_acc
        self.threshold_complete = threshold_complete
        self.voxel_size = voxel_size
        self.range = range
        self.void = void
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.cnt = 0
        self.tot_acc = 0.0
        self.tot_cmpl = 0.0
        self.tot_f1_mean = 0.0
        self.eps = 1e-8

    def voxel2points(self, voxel):
        # occIdx = torch.where(torch.logical_and(voxel != FREE, voxel != NOT_OBSERVED))
        # if isinstance(voxel, np.ndarray): voxel = torch.from_numpy(voxel)
        mask = np.logical_not(
            reduce(
                np.logical_or, [voxel == self.void[i] for i in range(len(self.void))]
            )
        )
        occIdx = np.where(mask)

        points = np.concatenate(
            (
                occIdx[0][:, None] * self.voxel_size[0]
                + self.voxel_size[0] / 2
                + self.range[0],
                occIdx[1][:, None] * self.voxel_size[1]
                + self.voxel_size[1] / 2
                + self.range[1],
                occIdx[2][:, None] * self.voxel_size[2]
                + self.voxel_size[2] / 2
                + self.range[2],
            ),
            axis=1,
        )
        return points

    def add_batch(self, semantics_pred, semantics_gt, mask_lidar, mask_camera):

        # for scene_token in tqdm(preds_dict.keys()):
        self.cnt += 1

        if self.use_image_mask:

            semantics_gt[mask_camera == False] = 255
            semantics_pred[mask_camera == False] = 255
        elif self.use_lidar_mask:
            semantics_gt[mask_lidar == False] = 255
            semantics_pred[mask_lidar == False] = 255
        else:
            pass

        ground_truth = self.voxel2points(semantics_gt)
        prediction = self.voxel2points(semantics_pred)
        if prediction.shape[0] == 0:
            accuracy = 0
            completeness = 0
            fmean = 0

        else:
            prediction_tree = KDTree(prediction, leaf_size=self.leaf_size)
            ground_truth_tree = KDTree(ground_truth, leaf_size=self.leaf_size)
            complete_distance, _ = prediction_tree.query(ground_truth)
            complete_distance = complete_distance.flatten()

            accuracy_distance, _ = ground_truth_tree.query(prediction)
            accuracy_distance = accuracy_distance.flatten()

            # evaluate completeness
            complete_mask = complete_distance < self.threshold_complete
            completeness = complete_mask.mean()

            # evalute accuracy
            accuracy_mask = accuracy_distance < self.threshold_acc
            accuracy = accuracy_mask.mean()

            fmean = 2.0 / (1 / (accuracy + self.eps) + 1 / (completeness + self.eps))

        self.tot_acc += accuracy
        self.tot_cmpl += completeness
        self.tot_f1_mean += fmean

    def count_fscore(
        self,
    ):
        base_color, attrs = "red", ["bold", "dark"]
        print(
            pcolor(
                "\n######## F score: {} #######".format(self.tot_f1_mean / self.cnt),
                base_color,
                attrs=attrs,
            )
        )
