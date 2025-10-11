# mmdet3d/models/renderers/pulsar_renderer.py

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import cv2  # Using OpenCV for image I/O

from mmcv.runner import BaseModule
from ..builder import RENDERERS

# Ensure PyTorch3D is available
try:
    from pytorch3d.renderer import (
        PerspectiveCameras,
        PointsRasterizationSettings,
        PointsRenderer,
        PointsRasterizer,
        AlphaCompositor,
        PulsarPointsRenderer,
    )
    from pytorch3d.structures import Pointclouds

    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False


@RENDERERS.register_module()
class PulsarRenderer(BaseModule):
    """
    Differentiable renderer using the PyTorch3D backend.

    This class wraps PyTorch3D's rendering capabilities to project a 3D
    semantic occupancy grid into 2D semantic masks. This process is
    differentiable, allowing for end-to-end training with a 2D semantic
    loss. It supports two rendering backends: the standard `PointsRenderer`
    with alpha compositing and the faster `PulsarPointsRenderer`.
    """

    def __init__(
        self,
        width,
        height,
        voxel_size,
        point_cloud_range,
        loss_2d_weight=1.0,
        backend="points_renderer",
        pulsar_cfg=None,
        init_cfg=None,
    ):
        """
        Initialize the PulsarRenderer module.

        Args:
            width (int): The width of the rendered output image.
            height (int): The height of the rendered output image.
            voxel_size (list[float]): The size of each voxel in the 3D grid.
            point_cloud_range (list[float]): The range of the 3D point cloud.
            loss_2d_weight (float, optional): The weight for the rendering loss. Defaults to 1.0.
            backend (str, optional): The rendering backend ('points_renderer' or 'pulsar').
            pulsar_cfg (dict, optional): Configuration for the Pulsar backend.
            init_cfg (dict, optional): Initialization config dict.
        """
        super(PulsarRenderer, self).__init__(init_cfg)
        if not PYTORCH3D_AVAILABLE:
            raise ImportError("Please install PyTorch3D to use this module.")

        self.width = width
        self.height = height
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.point_cloud_range_min = torch.tensor(
            point_cloud_range[:3], dtype=torch.float32
        )

        self.loss_2d = nn.CrossEntropyLoss(reduction="none", ignore_index=0)
        self.loss_2d_weight = loss_2d_weight

        self.backend = backend
        self.pulsar_cfg = pulsar_cfg if pulsar_cfg is not None else {}
        if self.backend not in ["points_renderer", "pulsar"]:
            raise ValueError(f"Unknown renderer backend: {self.backend}")

        # --- CLASS MAPPING SETUP ---
        class_names_3d = [
            "void/other",
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
            "drivable_surface",
            "other_flat",
            "sidewalk",
            "terrain",
            "manmade",
            "vegetation",
            "free",
        ]
        class_names_2d = [
            "background",
            "barrier",
            "bicycle",
            "bus",
            "sedan",
            "motorcycle",
            "crane",
            "highway",
            "people",
            "traffic_cone",
            "sidewalk",
            "truck",
            "building",
            "overhead bridge",
            "pole",
            "billboard",
            "tree",
            "sky",
        ]
        mapping_2d_to_3d_names = {
            "background": ["void/other"],
            "barrier": ["barrier"],
            "bicycle": ["bicycle"],
            "bus": ["bus"],
            "sedan": ["car"],
            "motorcycle": ["motorcycle"],
            "crane": ["construction_vehicle"],
            "highway": ["drivable_surface"],
            "people": ["pedestrian"],
            "traffic_cone": ["traffic_cone"],
            "sidewalk": ["sidewalk"],
            "truck": ["truck", "trailer"],
            "building": ["manmade"],
            "overhead bridge": ["manmade"],
            "pole": ["manmade"],
            "billboard": ["manmade"],
            "tree": ["vegetation"],
        }

        d_3d = {name: i for i, name in enumerate(class_names_3d)}
        d_2d = {name: i for i, name in enumerate(class_names_2d)}

        self.map_2d_to_3d = torch.full((len(d_2d),), -1, dtype=torch.long)
        for name_2d, names_3d in mapping_2d_to_3d_names.items():
            self.map_2d_to_3d[d_2d[name_2d]] = d_3d[names_3d[0]]

        self.sky_channel_2d = d_2d["sky"]
        self.background_channel_2d = d_2d["background"]

    def _get_camera_components(self, img_meta, view_idx, device):
        """Extracts individual camera parameter tensors for later batching."""
        lidar2cam = torch.from_numpy(img_meta["lidar2cam"][view_idx]).float().to(device)
        K = torch.from_numpy(img_meta["cam_intrinsic"][view_idx]).float().to(device)
        R_nusc, T_nusc = lidar2cam[:3, :3], lidar2cam[:3, 3]
        C_cam = torch.tensor(
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32, device=device
        )
        R_view, T_view = C_cam @ R_nusc, C_cam @ T_nusc
        fx, fy, px, py = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        image_height, image_width = img_meta["ori_shape"][:2]

        return {
            "R": R_view.unsqueeze(0),
            "T": T_view.unsqueeze(0),
            "focal_length": torch.tensor([[fx, fx]], device=device),
            "principal_point": torch.tensor([[px, py]], device=device),
            "image_size": torch.tensor([[image_height, image_width]], device=device),
        }

    def forward(self, sparse_occ_grid, img_metas, gt_2d_masks, gt_2d_conf):
        """
        Forward pass for rendering and loss calculation.
        """
        coords, logits, device = (
            sparse_occ_grid.C,
            sparse_occ_grid.F,
            sparse_occ_grid.F.device,
        )
        batch_size, num_views = len(img_metas), gt_2d_masks.shape[1]

        # --- 1. Prepare Features: Use raw logits directly for both backends ---
        features_for_render = logits

        # --- 2. Create Batched Point Cloud ---
        points_list, features_list = [], []
        for b in range(batch_size):
            batch_mask = coords[:, 0] == b
            points_b_xyz = coords[batch_mask, 1:].float() * self.voxel_size.to(
                device
            ) + self.point_cloud_range_min.to(device)
            if points_b_xyz.shape[0] > 0:
                points_list.append(points_b_xyz)
                features_list.append(features_for_render[batch_mask])

        if not points_list:
            return dict(
                loss_render_2d=torch.tensor(0.0, device=device, requires_grad=True)
            )

        point_cloud_batch = Pointclouds(points=points_list, features=features_list)

        # --- 3. Create Batched Camera ---
        cam_params = [
            self._get_camera_components(img_metas[b], i, device)
            for b in range(batch_size)
            for i in range(num_views)
        ]
        camera_batch = PerspectiveCameras(
            R=torch.cat([p["R"] for p in cam_params], dim=0),
            T=torch.cat([p["T"] for p in cam_params], dim=0),
            focal_length=torch.cat([p["focal_length"] for p in cam_params], dim=0),
            principal_point=torch.cat(
                [p["principal_point"] for p in cam_params], dim=0
            ),
            image_size=torch.cat([p["image_size"] for p in cam_params], dim=0),
            device=device,
            in_ndc=False,
        )

        # --- 4. Render the Logits ---
        point_cloud_expanded = point_cloud_batch.extend(num_views)
        radii = self.voxel_size.min().item() / 2.0
        raster_settings = PointsRasterizationSettings(
            image_size=(self.height, self.width), radius=radii, points_per_pixel=10
        )
        rasterizer = PointsRasterizer(
            cameras=camera_batch, raster_settings=raster_settings
        )

        if self.backend == "pulsar":
            n_channels = logits.shape[1]
            background_features = torch.zeros(n_channels, device=device)
            renderer = PulsarPointsRenderer(
                rasterizer=rasterizer, n_channels=n_channels
            ).to(device)

            n_instances = len(point_cloud_expanded)
            pulsar_kwargs = {
                "gamma": tuple(
                    self.pulsar_cfg.get("gamma", 1e-4) for _ in range(n_instances)
                ),
                "znear": tuple(
                    self.pulsar_cfg.get("znear", 0.1) for _ in range(n_instances)
                ),
                "zfar": tuple(
                    self.pulsar_cfg.get("zfar", 100.0) for _ in range(n_instances)
                ),
                "bg_col": background_features,
            }
            rendered_maps_raw = renderer(point_cloud_expanded, **pulsar_kwargs)
        else:  # points_renderer
            background_color = torch.zeros(logits.shape[1], device=device)
            renderer = PointsRenderer(
                rasterizer=rasterizer,
                compositor=AlphaCompositor(background_color=background_color),
            )
            rendered_maps_raw = renderer(point_cloud_expanded)

        # --- 5. Calculate Loss Directly on Rendered Logits ---
        # output is (B, H, W, 18).
        rendered_logits = rendered_maps_raw.permute(0, 3, 1, 2).contiguous()

        gt_masks_flat = gt_2d_masks.view(
            batch_size * num_views, self.height, self.width
        )
        gt_conf_flat = gt_2d_conf.view(batch_size * num_views, self.height, self.width)

        self.map_2d_to_3d = self.map_2d_to_3d.to(device)
        gt_masks_remapped = self.map_2d_to_3d[gt_masks_flat.long()]
        gt_masks_remapped[gt_masks_flat == self.sky_channel_2d] = (
            self.background_channel_2d
        )

        raw_loss = self.loss_2d(rendered_logits, gt_masks_remapped.long())
        weighted_loss = (raw_loss * gt_conf_flat).mean()
        final_loss = weighted_loss * self.loss_2d_weight

        return dict(loss_render_2d=final_loss)
