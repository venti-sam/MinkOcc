# configs/bevdet_occ/minkoccv4_full.py

# Inherit all settings from the MinkOccV3 configuration.
# This includes dataset paths, data pipelines, training schedules, etc.
_base_ = './minkoccv3_mini.py'

# ===================================================================
#                       USER-DEFINED VARIABLES
# ===================================================================

# =========== Model Settings ===========
# The main model type.
model_type = 'MinkOccV4'
# The dataset name, used internally by the model for specific logic.
dataset_name = 'nuscenes'
# Voxel and point cloud range settings.
voxel_size = [0.4, 0.4, 0.4]
point_cloud_range = [-40.0, -40.0, -2.6, 40.0, 40.0, 5.4]

# --- Loss Weights ---
# Weight for the Binary Cross-Entropy loss in the 3D occupancy neck.
loss_bce_weight = 1.0
# Weight for the Cross-Entropy loss in the 3D occupancy neck.
loss_ce_weight = 0.5
# Weight for the 2D rendering loss from the Pulsar renderer.
loss_2d_weight = 0.5  # Set to 0.0 to disable rendering loss for debugging.

# --- Model Flags ---
# Whether to use the visibility mask during loss calculation.
use_mask = True
# Whether to freeze the image backbone and neck (useful for fine-tuning).
freeze_layers = False

# --- Renderer Settings ---
# The backend for the differentiable renderer ('pulsar' or 'points_renderer').
renderer_backend = 'points_renderer'
# Width of the rendered image.
render_width = 1600
# Height of the rendered image.
render_height = 900
# Pulsar-specific setting for differentiability.
pulsar_gamma = 1e-4
# Near clipping plane for the renderer.
pulsar_znear = 0.1
# Far clipping plane for the renderer.
pulsar_zfar = 50.0

# =========== Data Augmentation Settings ===========
data_config = {
    'input_size': (384, 704),
    # Augmentation
    'resize': (-0.03, 0.03),
    'rot': (-0.0, 0.0),
    'flip': True,
    'crop_h': (0.0, 0.0),
    # Photometric Distortion
    'pmd': dict(
        brightness_delta=16,
        contrast_lower=0.8,
        contrast_upper=1.2,
        saturation_lower=0.8,
        saturation_upper=1.2,
        hue_delta=9,
        rate=0.5
    )
}

# =========== Training Settings ===========
# Batch size per GPU.
samples_per_gpu = 2
# Number of workers per GPU.
workers_per_gpu = 2
# Ratio of data used for strong supervision (e.g., with 3D labels).
strong_supervision_ratio = 1.0
# Maximum number of training epochs.
max_epochs = 50
# Learning rate for the optimizer.
learning_rate = 1e-4
# Weight decay for the optimizer.
weight_decay = 1e-2

# =========== Logging Settings ===========
# Project name for Weights & Biases logging.
wandb_project = 'MinkOccV4-nuScenes'
# Run name for Weights & Biases logging.
wandb_run_name = 'minkoccv4_mini_exp_strong_sup'

# ===================================================================

# --- 2. Override the model dictionary to define MinkOccV4 ---
model = dict(
    # Change the detector type to our new MinkOccV4 class.
    type='MinkOccV4',

    # The image backbone and neck are inherited from the base config.
    # The FPN from MinkOccV3 produces 5 feature levels, which matches the
    # num_levels required by the new fusion layer.

    # Modify the point cloud branch:
    pts_voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=4,
        feat_channels=[32, 32],  # Output dimension is 32.
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        # CRITICAL CHANGE: The fusion_layer is REMOVED from the voxel encoder.
        # Fusion will now be handled by a separate, top-level module.
    ),

    # CRITICAL CHANGE: Add the new deformable attention fusion configuration.
    # This module will be instantiated in the MinkOccV4 detector.
    fusion_cfg=dict(
        type='SparseDeformableFusion',
        embed_dims=32,  # Must match the output of pts_voxel_encoder (feat_channels[-1])
        num_cams=6,
        pc_range=point_cloud_range,
        voxel_size=voxel_size,
        dropout_p=0.1,
        deformable_attention=dict(
            type='SpatialCrossAttention',
            embed_dims=32,
            num_cams=6,
            deformable_attention=dict(
                type='MSDeformableAttention3D',
                embed_dims=32,
                num_heads=8,
                num_levels=5,  # Matches the 5 levels from the FPN in img_neck
                num_points=4,
                batch_first=True
            )
        )
    ),
    # Disable the renderer for initial experiments; can be re-enabled later.
    renderer_cfg = None,

    # The 3D UNet (occ_backbone, occ_neck), renderer, and loss weights
    # are inherited directly from the base minkoccv3_full.py config.
)

# --- 3. Update Logistics ---
# Change the run name to distinguish this experiment from V3 runs.

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project=wandb_project,
                name=wandb_run_name
            )
        )
    ]
)