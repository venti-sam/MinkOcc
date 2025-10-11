# mmdet3d/models/renderers/pulsar_renderer.py (DEBUG VERSION)

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


def print_tensor_stats(name, tensor):
    """Helper function to print detailed statistics of a tensor."""
    if tensor is None:
        print(f"  - {name}: None")
        return
    if not isinstance(tensor, torch.Tensor):
        print(f"  - {name}: Not a tensor (type: {type(tensor)})")
        return
    
    grad_fn_status = "PRESENT" if tensor.grad_fn is not None else "--- ABSENT ---"
    if not tensor.is_leaf and tensor.grad_fn is None and tensor.requires_grad:
        grad_fn_status = "!!! CRITICAL: MISSING grad_fn ON NON-LEAF TENSOR !!!"

    print(
        f"  - {name}:\n"
        f"    - Shape: {tensor.shape}\n"
        f"    - Dtype: {tensor.dtype}\n"
        f"    - Device: {tensor.device}\n"
        f"    - grad_fn: {grad_fn_status}\n"
        f"    - Stats: Min={tensor.min().item():.4f}, Max={tensor.max().item():.4f}, "
        f"Mean={tensor.mean().item():.4f}, Std={tensor.std().item():.4f}"
    )

@RENDERERS.register_module()
class PulsarRenderer(BaseModule):
    """
    Differentiable renderer using the PyTorch3D backend. (DEBUG VERSION)
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
        super(PulsarRenderer, self).__init__(init_cfg)
        if not PYTORCH3D_AVAILABLE:
            raise ImportError("Please install PyTorch3D to use this module.")

        print("\n[DEBUG] Initializing PulsarRenderer...")
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
        self.class_names_3d = [
            "void/other","barrier","bicycle","bus","car","construction_vehicle",
            "motorcycle","pedestrian","traffic_cone","trailer","truck",
            "drivable_surface","other_flat","sidewalk","terrain","manmade",
            "vegetation","free",
        ]
        self.class_names_2d = [
            "background","barrier","bicycle","bus","sedan","motorcycle","crane",
            "highway","people","traffic_cone","sidewalk","truck","building",
            "overhead bridge","pole","billboard","tree","sky",
        ]
        mapping_2d_to_3d_names = {
            "background": ["void/other"],"barrier": ["barrier"],"bicycle": ["bicycle"],
            "bus": ["bus"],"sedan": ["car"],"motorcycle": ["motorcycle"],
            "crane": ["construction_vehicle"],"highway": ["drivable_surface"],
            "people": ["pedestrian"],"traffic_cone": ["traffic_cone"],
            "sidewalk": ["sidewalk"],"truck": ["truck", "trailer"],"building": ["manmade"],
            "overhead bridge": ["manmade"],"pole": ["manmade"],"billboard": ["manmade"],
            "tree": ["vegetation"],
        }

        self.channel_mapping = []
        d_3d = {name: i for i, name in enumerate(self.class_names_3d)}
        d_2d = {name: i for i, name in enumerate(self.class_names_2d)}
        for target_name_2d, source_names_3d in mapping_2d_to_3d_names.items():
            self.channel_mapping.append(
                (d_2d[target_name_2d], [d_3d[name] for name in source_names_3d])
            )
        self.free_channel_3d = d_3d["free"]
        self.sky_channel_2d = d_2d["sky"]
        self.background_channel_2d = d_2d["background"]

    def _get_camera_components(self, img_meta, view_idx, device):
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
            "R": R_view.unsqueeze(0),"T": T_view.unsqueeze(0),
            "focal_length": torch.tensor([[fx, fx]], device=device),
            "principal_point": torch.tensor([[px, py]], device=device),
            "image_size": torch.tensor([[image_height, image_width]], device=device),
        }

    def forward(self, sparse_occ_grid, img_metas, gt_2d_masks, gt_2d_conf):
        print("\n" + "="*80)
        print("--- ENTERING PULSAR RENDERER FORWARD PASS (DEBUG MODE) ---")
        print("="*80)

        coords, logits, device = (
            sparse_occ_grid.C,
            sparse_occ_grid.F,
            sparse_occ_grid.F.device,
        )
        batch_size, num_views = len(img_metas), gt_2d_masks.shape[1]

        # --- 1. Input Validation ---
        print("\n--- SECTION 1: INPUT VALIDATION ---")
        print(f"Batch Size: {batch_size}, Num Views: {num_views}")
        print_tensor_stats("Input Logits (sparse_occ_grid.F)", logits)
        
        # --- 2. Prepare Features and Batched Point Cloud ---
        print("\n--- SECTION 2: FEATURE PREPARATION ---")
        probs_3d = F.softmax(logits, dim=1)
        print_tensor_stats("3D Probs (after softmax)", probs_3d)
        
        opacity = 1.0 - probs_3d[:, self.free_channel_3d]
        print_tensor_stats("Opacity (1 - P(free))", opacity)
        
        num_points, num_2d_classes = probs_3d.shape[0], len(self.class_names_2d)
        probs_2d_for_render = torch.zeros(num_points, num_2d_classes, device=device)
        for target_idx, src_indices in self.channel_mapping:
            probs_2d_for_render[:, target_idx] = torch.sum(
                probs_3d[:, src_indices], dim=1
            )
        print_tensor_stats("2D Probs (after mapping)", probs_2d_for_render)
        
        features_for_render = torch.cat(
            [probs_2d_for_render, opacity.unsqueeze(1)], dim=1
        )
        print_tensor_stats("Final Features for Render (2D Probs + Opacity)", features_for_render)

        points_list = []
        features_list = []
        for b in range(batch_size):
            batch_mask = coords[:, 0] == b
            points_b_xyz = coords[batch_mask, 1:].float() * self.voxel_size.to(
                device
            ) + self.point_cloud_range_min.to(device)
            if points_b_xyz.shape[0] > 0:
                points_list.append(points_b_xyz)
                features_list.append(features_for_render[batch_mask])

        if not points_list:
            print("WARNING: No points to render. Returning zero loss.")
            return dict(
                loss_render_2d=torch.tensor(0.0, device=device, requires_grad=True)
            )
        
        point_cloud_batch = Pointclouds(points=points_list, features=features_list)
        print(f"Created Pointclouds object with {len(point_cloud_batch)} clouds.")

        # --- 3. Create a Single Batched Camera Object ---
        print("\n--- SECTION 3: CAMERA PREPARATION ---")
        cam_params = [
            self._get_camera_components(img_metas[b], i, device)
            for b in range(batch_size)
            for i in 
    range(num_views)
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
        print(f"Created batched PerspectiveCameras object for {len(camera_batch)} cameras.")

        # --- 4. Single, Batched Rendering Call ---
        print("\n--- SECTION 4: RENDERING ---")
        point_cloud_expanded = point_cloud_batch.extend(num_views)
        n_channels = features_for_render.shape[1]
        background_features = torch.zeros(n_channels, device=device)
        background_features[self.sky_channel_2d] = 1.0
        radii = self.voxel_size.min().item() / 2.0
        print(f"Using point radius: {radii:.4f}")

        raster_settings = PointsRasterizationSettings(
            image_size=(self.height, self.width), radius=radii, points_per_pixel=10
        )
        rasterizer = PointsRasterizer(
            cameras=camera_batch, raster_settings=raster_settings
        )

        if self.backend == "pulsar":
            renderer = PulsarPointsRenderer(
                rasterizer=rasterizer, n_channels=n_channels
            ).to(device)
            n_instances = len(point_cloud_expanded)
            pulsar_kwargs = {
                "gamma": tuple(self.pulsar_cfg.get("gamma", 1e-4) for _ in range(n_instances)),
                "znear": tuple(self.pulsar_cfg.get("znear", 0.1) for _ in range(n_instances)),
                "zfar": tuple(self.pulsar_cfg.get("zfar", 100.0) for _ in range(n_instances)),
                "bg_col": background_features,
            }
            print(f"Calling Pulsar backend with gamma={pulsar_kwargs['gamma'][0]:.1e}...")
            rendered_maps_raw = renderer(point_cloud_expanded, **pulsar_kwargs)
        else:
            renderer = PointsRenderer(
                rasterizer=rasterizer,
                compositor=AlphaCompositor(background_color=background_features),
            )
            print("Calling standard PointsRenderer backend...")
            rendered_maps_raw = renderer(point_cloud_expanded)
        
        print_tensor_stats("Rendered Maps (Raw Output)", rendered_maps_raw)

        # --- 5. Vectorized Loss Calculation ---
        print("\n--- SECTION 5: LOSS CALCULATION ---")
        rendered_maps = rendered_maps_raw.permute(0, 3, 1, 2)[:, :-1, :, :]
        gt_masks_flat = gt_2d_masks.view(
            batch_size * num_views, self.height, self.width
        )
        gt_conf_flat = gt_2d_conf.view(batch_size * num_views, self.height, self.width)

        gt_masks_for_loss = gt_masks_flat.clone()
        gt_masks_for_loss[gt_masks_for_loss == self.sky_channel_2d] = (
            self.background_channel_2d
        )
        
        print_tensor_stats("Rendered Maps for Loss (final channels)", rendered_maps)
        print_tensor_stats("GT Masks for Loss", gt_masks_for_loss.float())
        print_tensor_stats("GT Confidence for Loss", gt_conf_flat)

        raw_loss = self.loss_2d(rendered_maps, gt_masks_for_loss.long())
        print_tensor_stats("Raw Loss (per-pixel)", raw_loss)

        weighted_loss = (raw_loss * gt_conf_flat).mean()
        final_loss = weighted_loss * self.loss_2d_weight

        print_tensor_stats("Weighted Loss (scalar)", weighted_loss)
        print_tensor_stats("Final Loss (scalar)", final_loss)
        print("="*80)
        print("--- EXITING PULSAR RENDERER FORWARD PASS ---")
        print("="*80 + "\n")

        return dict(loss_render_2d=final_loss)