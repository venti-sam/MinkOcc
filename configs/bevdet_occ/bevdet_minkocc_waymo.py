# Copyright (c) Phigent Robotics. All rights reserved.

_base_ = ['../_base_/default_runtime.py']
# Global
# # For nuScenes we usually do 10-class detection
# class_names = [
#     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
#     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
# ]
class_names = ['Car', 'Pedestrian', 'Cyclist']
data_config = {
    'input_size': (256, 704),
    'src_size': (1280, 1920),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}
# GT occuapncy configs
use_infov_mask = True
use_lidar_mask = False
use_camera_mask = True
FREE_LABEL = 23
num_classes = 16

# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

voxel_size = [0.1, 0.1, 0.2]
# z axis is a bit hard to adjust if ground is not flat, set as -2.6 for now, which means z axis is 0 - 20
point_cloud_range = [-40.0, -40.0, -2.6, 40.0, 40.0, 5.4]

numC_Trans = 32

# multi_adj_frame_id_cfg = (1, 0, 1)

model = dict(
    type='BEVStereo4DOCC',
    align_after_view_transfromation=False,
    dataset_type = 'waymo',
    num_adj=0,
    # add in lidar minkunet here 
    lidar_backbone = dict(
        type='TR3DMinkResNet',
        in_channels=6,
        depth=18,
        pool = False,
        num_stages = 4,
        in_planes = 16,
        norm='batch',
        num_planes=(32, 64, 128, 256)
    ),
    # add in lidar minkunet neck here 
    lidar_neck = dict(
        type='TR3DNeck',
        in_channels=(32, 64, 128, 256),
        out_channels=32,
        strides=(4, 8, 16, 32),  # Strides from the backbone
        loss_bce_weight = 1.0,
        is_generative=True
    ),
    
    # add in voxelization code here, the method is derived from centerpoint.py (second)
    pts_voxel_layer=dict(
        max_num_points=20, 
        point_cloud_range=point_cloud_range,
        voxel_size=[0.4, 0.4, 0.4],  # xy size follow centerpoint
        # max_voxels=(90000, 120000)),
    ),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=6),

    
    # add in sparse fusion (modelled from openoccupancy adaptive fusion)
    # sparse_fusion=dict(
    #     type='SparseFusion',
    #     out_channels=16,
    #     img_in_channels=64,
    #     pts_in_channels=32
    # ),
    
    
    
    
    
    ###########################################
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVStereo',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=False,
        loss_depth_weight=0.05,
        depthnet_cfg=dict(use_dcn=False,
                          aspp_mid_channels=96,
                          stereo=True,
                          bias=5.),
        downsample=16),
    ###########################################
    
    
    ######################################################
    # Multo- modal semantic segmentation
    # add in lidar minkunet here 
    # occ_backbone = dict(
    #     type='TR3DMinkResNet',
    #     in_channels=16,
    #     depth=18,
    #     pool = False,
    #     num_stages = 4,
    #     in_planes = 32, 
    #     norm='batch',
    #     num_planes=(64, 128, 256, 512)
    # ),
    # # add in lidar minkunet neck here 
    # occ_neck = dict(
    #     type='TR3DNeck',
    #     in_channels=(64, 128, 256, 512),
    #     out_channels=32,
    #     strides=(4, 8, 16, 32),  # Strides from the backbone
    #     # loss_bce_weight = 1.0,
    #     is_generative=False
    # ),
    
    # img_bev_encoder_backbone=dict(
    #     type='CustomResNet3D',
    #     numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
    #     num_layer=[1, 2, 4],
    #     with_cp=False,
    #     num_channels=[numC_Trans,numC_Trans*2,numC_Trans*4],
    #     stride=[1,2,2],
    #     backbone_output_ids=[0,1,2]),
    # img_bev_encoder_neck=dict(type='LSSFPN3D',
    #                           in_channels=numC_Trans*7,
    #                           out_channels=numC_Trans),
    # pre_process=dict(
    #     type='CustomResNet3D',
    #     numC_input=numC_Trans,
    #     with_cp=False,
    #     num_layer=[1,],
    #     num_channels=[numC_Trans,],
    #     stride=[1,],
    #     backbone_output_ids=[0,]),
    loss_occ=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0),
    use_mask=True,
)

# Data
dataset_type = 'WaymoDatasetOccupancy'
data_root = 'data/waymo/kitti_format/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(type='WaymoLoadMultiViewImageFromFiles', 
         data_config=data_config,
         to_float32=True, 
         img_scale=(1280, 1920),
         is_train = True,
         scales = [0.1],
    ),
    dict(
        type='WaymoLoadOccGTFromFile',
        crop_x=False,
        use_infov_mask=use_infov_mask,
        use_camera_mask=use_camera_mask,
        use_lidar_mask=use_lidar_mask,
        FREE_LABEL=FREE_LABEL,
        num_classes=num_classes,
    ),
    dict(type='WaymoLoadAnnotations'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=6,
        file_client_args=file_client_args),
    dict(
        type='WaymoBEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(type='WaymoPointToMultiViewDepthFusion', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'points', 'gt_depth', 'voxel_semantics', 'mask_infov',
                                'mask_lidar','mask_camera'])
]

test_pipeline = [
     dict(type='WaymoLoadMultiViewImageFromFiles', 
         data_config=data_config,
         to_float32=True, 
         img_scale=(1280, 1920),
         is_train = False,
         scales = [0.1],
    ),
    dict(type='WaymoLoadAnnotations'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=6,
        file_client_args=file_client_args),
    dict(type='WaymoBEVAug',
         bda_aug_conf=bda_aug_conf,
         classes=class_names,
         is_train=False),
    dict(
        type='WaymoLoadOccGTFromFile',
        crop_x=False,
        use_infov_mask=use_infov_mask,
        use_camera_mask=use_camera_mask,
        use_lidar_mask=use_lidar_mask,
        FREE_LABEL=FREE_LABEL,
        num_classes=num_classes,
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 'voxel_semantics', 'mask_camera'])
        ])
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

# share_data_config = dict(
#     type=dataset_type,
#     classes=class_names,
#     modality=input_modality,
#     stereo=True,
#     filter_empty_gt=False,
#     img_info_prototype='bevdet4d',
#     multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
# )

# test_data_config = dict(
#     pipeline=test_pipeline,
#     ann_file=data_root + 'bevdetv3-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=1, # batch size
    workers_per_gpu=4,
    train=dict(
        type = dataset_type,
        data_root = data_root,
        ann_file = data_root + 'waymo_infos_train.pkl', 
        pose_file = data_root + 'cam_infos.pkl',
        split = 'training',
        pipeline = train_pipeline,
        modality = input_modality,
        classes = class_names,
        test_mode = False,
        box_type_3d = 'LiDAR',
        load_interval = 1,
        filter_empty_gt = False,
    ),
    val = dict(
        type = dataset_type,
        data_root = data_root,
        ann_file = data_root + 'waymo_infos_val.pkl',
        pose_file = data_root + 'cam_infos_val.pkl',
        split = 'training',
        pipeline = test_pipeline,
        modality = input_modality,
        classes = class_names,
        test_mode = True,
        box_type_3d = 'LiDAR',
        filter_empty_gt = False,
    ),
    test = dict(
        type = dataset_type,
        data_root = data_root,
        ann_file = data_root + 'waymo_infos_val.pkl',
        pose_file = data_root + 'cam_infos_val.pkl',
        split = 'training',
        pipeline = test_pipeline,
        modality = input_modality,
        classes = class_names,
        test_mode = True,
        box_type_3d = 'LiDAR',
        filter_empty_gt = False,
    ))
        
       


# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[100,])
runner = dict(type='EpochBasedRunner', max_epochs=100)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

# load_from="bevdet-r50-4d-stereo-cbgs.pth"
# fp16 = dict(loss_scale='dynamic')
