_base_ = ['../_base_/schedules/cosine.py', '../_base_/default_runtime.py']

# model settings
voxel_size = [0.4, 0.4, 0.4]
point_cloud_range = [-40.0, -40.0, -2.6, 40.0, 40.0, 5.4]

checkpoint = 'pretrained_checkpoints/resnet50-19c8e357.pth'
model = dict(
    type='MinkOccV3',
    dataset_type = 'nuscenes',
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
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=128,
        num_outs=5),

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
            fuse_out=False)),
    occ_backbone = dict(
        type='TR3DMinkResNet',
        in_channels=32,
        depth=18,
        pool = False,
        num_stages = 4,
        in_planes = 32,
        norm='batch',
        num_planes=(64, 128, 256, 512)
    ),
    occ_neck = dict(
        type='TR3DNeck',
        in_channels=(64, 128, 256, 512),
        out_channels=18, # 18 classes in cvpr-nuscenes
        strides=(4, 8, 16, 32),  # Strides from the backbone
        is_generative=True 
    ),
    renderer_cfg=dict(
        type='PulsarRenderer',
        # --- CHOOSE YOUR BACKEND HERE ---
        # backend='points_renderer',  # The original AlphaCompositor renderer
        backend='pulsar',           # The new, fast Pulsar renderer
        width=1600,  
        height=900,  
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        loss_2d_weight=0.5,
        pulsar_cfg=dict(
            gamma=1e-4, 
            znear=0.1,
            zfar=50.0,
        )
    ),
    out_dim = 18,
    loss_bce_weight = 1.0,
    loss_ce_weight = 0.5,
    use_mask=True,
    freeze_layers = False,
)

# dataset settings
dataset_type = 'NuScenesDatasetOccpancy'
data_root = '/home/users/astar/ares/samuelsz/scratch/data/nuscenes_mini/'
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
file_client_args = dict(backend='disk')
input_modality = dict(use_lidar=True, use_camera=True)

data_config = {
    'input_size': (384, 704),
    # Augmentation
    'resize': (-0.03, 0.03),  # Reduced from (-0.06, 0.11)
    'rot': (-2.0, 2.0),        # Reduced from (-5.4, 5.4)
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
        rate=0.5  # Apply pmd with a 50% probability
    )
}
# train dataloader
train_pipeline = [
    dict(type='LoadPointsFromFile', 
         coord_type='LIDAR', 
         load_dim=5, 
         use_dim=4, 
         file_client_args = file_client_args),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='LoadOccGTFromFile'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(
    #     type='ResizeMultiImage',
    #     img_scale = [(225, 400), (450, 800)], # nuscenes resolution
    #     multiscale_mode='range',
    #     keep_ratio=True),
    dict(type='RandomAugMultiViewImage', data_config=data_config, is_train=True),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.78539816, 0.78539816],
    #     scale_ratio_range=[0.95, 1.05],
    #     translation_std=[0.2, 0.2, 0.2]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.0),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='RandomJitterPoints', 
         jitter_std=[0.01, 0.01, 0.01], 
         clip_range=[-0.05, 0.05]),
    dict(
        type='NormalizeMultiImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True
        ),
    dict(type='PadMultiImages', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['points', 'img', 'voxel_semantics', 'mask_camera', 'supervision_2d_mask', 'supervision_2d_conf']),
]


test_pipeline = [
    dict(type='LoadPointsFromFile', 
         coord_type='LIDAR', 
         load_dim=5, 
         use_dim=4,
         file_client_args = file_client_args),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(900, 1600),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='ResizeMultiImage',
                img_scale=data_config['input_size'], # Use the fixed input_size
                multiscale_mode='value', # Use 'value' for a fixed scale, not 'range'
                keep_ratio=True)
            ,
            # dict(type='RandomAugMultiViewImage', data_config=data_config, is_train=False),
            # dict(
            #     type='GlobalRotScaleTrans',
            #     rot_range=[0, 0],
            #     scale_ratio_range=[1., 1.],
            #     translation_std=[0, 0, 0]),
            # dict(type='RandomFlip3D'),
            dict(
                type='NormalizeMultiImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True
            ),
            dict(type='PadMultiImages', size_divisor=32),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img', 'voxel_semantics', 'mask_camera', 'supervision_2d_mask', 'supervision_2d_conf'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points', 'img'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        data_root=data_root,
        type = dataset_type,
        ann_file=data_root + 'bevdetv3-nuscenes-mini_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt = False,
        box_type_3d='LiDAR'),
    val=dict(
        data_root=data_root,
        type = dataset_type,
        ann_file=data_root + 'bevdetv3-nuscenes-mini_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        filter_empty_gt = False,
        box_type_3d='LiDAR'),
    test=dict(
        data_root=data_root,
        type = dataset_type,
        ann_file=data_root + 'bevdetv3-nuscenes-mini_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        filter_empty_gt = False,
        box_type_3d='LiDAR')
)

# Optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-4,  # Half of original lr for continued training
    weight_decay=1e-2
)

optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=200,  # Reduced warmup since model is already trained
    warmup_ratio=0.9,  # Higher ratio since we're already at a good point
    min_lr_ratio=5e-6  # Keeping your original setting
)

runner = dict(type='EpochBasedRunner', max_epochs=50)  # Reduced from 30

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
                project='MinkOccV3-nuScenes-mini',  # Your W&B project name
                name='minkoccv3_mini_exp_run1'      # A name for this specific run
            )
        )
    ]
)
# load_from = "minkoccv3_waymo_epoch_19.pth"
# evaluation = dict(interval=1, pipeline=eval_pipeline)

# You may need to download the model first is the network is unstable
# load_from = 'https://download.openmmlab.com/mmdetection3d/pretrain_models/mvx_faster_rcnn_detectron2-caffe_20e_coco-pretrain_gt-sample_kitti-3-class_moderate-79.3_20200207-a4a6a3c7.pth'  # noqa
