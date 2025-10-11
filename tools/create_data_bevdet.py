# Copyright (c) OpenMMLab. All rights reserved.
import os
import pickle
import argparse

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

# Assuming these modules are in your project's PYTHONPATH
from tools.data_converter import nuscenes_converter as nuscenes_converter
from tools.data_converter.create_gt_database import create_groundtruth_database

# --- This section remains unchanged ---
map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# --- All functions remain unchanged ---
def get_gt(info):
    """Generate gt labels from info."""
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT']['ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    for ann_info in info['ann_infos']:
        if (map_name_from_general_to_detection[ann_info['category_name']] not in classes
                or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue
        box = Box(
            ann_info['translation'], ann_info['size'], Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(classes.index(
            map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels

def nuscenes_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to nuScenes dataset."""
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

def add_ann_adj_info(data_root, version, extra_tag):
    """Adds annotation and adjacent frame info to the generated .pkl files."""
    nuscenes = NuScenes(version=version, dataroot=data_root)
    for split in ['train', 'val']:
        info_file = os.path.join(data_root, f'{extra_tag}_infos_{split}.pkl')
        if not os.path.exists(info_file):
            print(f"Info file not found: {info_file}, skipping split: {split}")
            continue
        print(f"Loading info file: {info_file}")
        with open(info_file, 'rb') as f:
            dataset = pickle.load(f)

        print(f"Processing {split} split...")
        for id in range(len(dataset['infos'])):
            if id % 100 == 0:
                print(f'{id}/{len(dataset["infos"])}')
            info = dataset['infos'][id]
            sample = nuscenes.get('sample', info['token'])
            ann_infos = [nuscenes.get('sample_annotation', ann) for ann in sample['anns']]
            for ann_info in ann_infos:
                velocity = nuscenes.box_velocity(ann_info['token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
            dataset['infos'][id]['ann_infos'] = ann_infos
            dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
            dataset['infos'][id]['scene_token'] = sample['scene_token']
            scene = nuscenes.get('scene', sample['scene_token'])
            dataset['infos'][id]['occ_path'] = f'{data_root}/gts/{scene["name"]}/{info["token"]}'

        with open(info_file, 'wb') as fid:
            print(f"Saving updated info file to: {info_file}")
            pickle.dump(dataset, fid)

# --- Main execution block with argparse ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Preparation for nuScenes dataset.')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory of the nuScenes data.')
    parser.add_argument('--version', type=str, required=True, choices=['v1.0-mini', 'v1.0-trainval'],
                        help='nuScenes version to process.')
    parser.add_argument('--info-prefix', type=str, required=True,
                        help='Prefix for the generated info files (e.g., "bevdetv3-nuscenes").')
    parser.add_argument('--max-sweeps', type=int, default=10,
                        help='Maximum number of sweeps to use.')
    parser.add_argument('--steps', type=str, default='123',
                        help="Which steps to run. '1' for initial prep, '2' for adding annotations, "
                             "'3' for creating GT database. E.g., '123' runs all, '23' runs the last two.")
    args = parser.parse_args()

    # Ensure the root directory exists.
    os.makedirs(args.data_root, exist_ok=True)

    if '1' in args.steps:
        print(f"\n--- Step 1: Creating initial nuScenes info files for version {args.version} ---")
        nuscenes_data_prep(
            root_path=args.data_root,
            info_prefix=args.info_prefix,
            version=args.version,
            max_sweeps=args.max_sweeps)
    
    if '2' in args.steps:
        print("\n--- Step 2: Adding extended annotation info ---")
        add_ann_adj_info(
            data_root=args.data_root,
            version=args.version,
            extra_tag=args.info_prefix)

    if '3' in args.steps:
        print("\n--- Step 3: Creating ground truth database ---")
        train_info_file = os.path.join(args.data_root, f'{args.info_prefix}_infos_train.pkl')
        if os.path.exists(train_info_file):
            create_groundtruth_database(
                'NuScenesDataset',
                args.data_root,
                info_prefix=args.info_prefix,
                info_path=train_info_file
            )
        else:
            print(f"Error: Training info file not found at {train_info_file}. Skipping Step 3.")

    print("\n--- Data preparation script finished. ---")