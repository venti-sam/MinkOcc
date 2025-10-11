# =================================================================_base_
_base_ = ['../_base_/schedules/cosine.py', '../_base_/default_runtime.py']

# ===================================================================
#                       USER-DEFINED VARIABLES
# ===================================================================

# =========== General Settings ===========
# The name of the dataset type being used.
dataset_type = 'NuScenesDatasetOccpancy'
# The root directory of your dataset.
data_root = 'data/nuscenes/'
# The prefix for the generated .pkl info files.
info_prefix = 'minkoccv3-nuscenes'
# The list of class names used in the dataset.
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# Path to the pretrained ResNet-50 backbone.
pretrained_backbone = 'pretrained_checkpoints/resnet50-19c8e357.pth'

# =========== Model Settings ===========
# The main model type.
model_type = 'MinkOccV3'
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
loss_2d_weight = 1.0  # Set to 0.0 to disable rendering loss for debugging.

# --- Model Flags ---
# Whether to use the visibility mask during loss calculation.
use_mask = True
# Whether to freeze the image backbone and neck (useful for fine-tuning).
freeze_layers = False

# --- Renderer Settings ---
# The backend for the differentiable renderer ('pulsar' or 'points_renderer').
renderer_backend = 'pulsar'
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
strong_supervision_ratio = 0.1
# Maximum number of training epochs.
max_epochs = 50
# Learning rate for the optimizer.
learning_rate = 1e-4
# Weight decay for the optimizer.
weight_decay = 1e-2

# =========== Logging Settings ===========
# Project name for Weights & Biases logging.
wandb_project = 'MinkOccV3-nuScenes'
# Run name for Weights & Biases logging.
wandb_run_name = 'minkoccv3_exp_weak_sup'

# ===================================================================
#                       DERIVED CONFIGURATIONS
# ===================================================================

# --- Model Configuration ---
model = dict(
    type=model_type,
    dataset_type=dataset_name,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        with_cp=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained_backbone)
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=128,
        num_outs=5
    ),
    pts_voxel_layer=dict(
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(-1, -1),
    ),
    pts_voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=4,
        feat_channels=[32, 32],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        fusion_layer=dict(
            type='PointFusion',
            img_channels=128,
            pts_channels=32,
            mid_channels=64,
            out_channels=32,
            img_levels=[0, 1, 2, 3, 4],
            align_corners=False,
            activate_out=True,
            fuse_out=False
        )
    ),
    occ_backbone=dict(
        type='TR3DMinkResNet',
        in_channels=32,
        depth=18,
        pool=False,
        num_stages=4,
        in_planes=32,
        norm='batch',
        num_planes=(64, 128, 256, 512)
    ),
    occ_neck=dict(
        type='TR3DNeck',
        in_channels=(64, 128, 256, 512),
        out_channels=18,
        strides=(4, 8, 16, 32),
        is_generative=True
    ),
    renderer_cfg=dict(
        type='PulsarRenderer',
        backend=renderer_backend,
        width=render_width,
        height=render_height,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        loss_2d_weight=loss_2d_weight,
        pulsar_cfg=dict(
            gamma=pulsar_gamma,
            znear=pulsar_znear,
            zfar=pulsar_zfar,
        )
    ),
    out_dim=18,
    loss_bce_weight=loss_bce_weight,
    loss_ce_weight=loss_ce_weight,
    use_mask=use_mask,
    freeze_layers=freeze_layers,
)

# --- Data Pipeline Configuration ---
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=4, file_client_args=dict(backend='disk')),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='LoadOccGTFromFile'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='RandomAugMultiViewImage', data_config=data_config, is_train=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.0),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='RandomJitterPoints', jitter_std=[0.01, 0.01, 0.01], clip_range=[-0.05, 0.05]),
    dict(type='NormalizeMultiImage', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='PadMultiImages', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img', 'voxel_semantics', 'mask_camera', 'supervision_2d_mask', 'supervision_2d_conf']),
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=4, file_client_args=dict(backend='disk')),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='LoadOccGTFromFile'),
    dict(type='RandomAugMultiViewImage', data_config=data_config, is_train=False),
    dict(type='NormalizeMultiImage', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='PadMultiImages', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(
        type='Collect3D',
        keys=[
            'points', 'img', 'voxel_semantics', 'mask_camera',
            'supervision_2d_mask', 'supervision_2d_conf'
        ]
    )
]


# --- Dataset Configuration ---
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + f'{info_prefix}_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=dict(use_lidar=True, use_camera=True),
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR',
        strong_supervision_ratio=strong_supervision_ratio,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + f'{info_prefix}_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=dict(use_lidar=True, use_camera=True),
        test_mode=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR'
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + f'{info_prefix}_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=dict(use_lidar=True, use_camera=True),
        test_mode=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR'
    )
)

# --- Optimizer and Scheduler Configuration ---
optimizer = dict(
    type='AdamW',
    lr=learning_rate,
    weight_decay=weight_decay
)

optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.9,
    min_lr_ratio=5e-6
)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

# --- Hooks and Logging Configuration ---
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

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