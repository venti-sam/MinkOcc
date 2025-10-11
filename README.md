
# MinkOcc: A 3D Occupancy Prediction Framework

MinkOcc is a high-performance, [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)-based framework for **3D Occupancy Prediction** from a **Bird's-Eye-View (BEV)** perspective. It is designed for multi-camera 3D perception tasks and builds upon the robust foundations of BEV-style object detection.

This guide provides a streamlined setup process using Docker, which is the recommended method for ensuring a consistent and reproducible environment.

---

### Quick Start Guide

1.  **Setup:** [Use Docker (Recommended)](#1-setup-with-docker-recommended)
2.  **Data:** [Prepare Datasets & Checkpoints](#2-data-and-model-preparation)
3.  **Training:** [Run Training](#3-training)
4.  **Evaluation:** [Evaluate a Model](#4-evaluation-and-visualization)

---

## 1. Setup with Docker (Recommended)

Using Docker is highly recommended as it automatically handles all CUDA, PyTorch, and other dependency conflicts.

#### Useful Scripts: Managing the Container

These scripts, located in the `MinkOcc/docker/` directory, are your primary tools for managing the development environment.

```shell
# Start the container in detached mode
./start_container.sh

# Join the container's interactive shell
./join_container.sh

# Stop the container when you are finished
./stop_container.sh
```

### Option A: Pull Pre-built Image

A pre-built image is available on Docker Hub, allowing you to skip the build process entirely.

```shell
# Pull the pre-built image from Docker Hub
docker pull cyadestiny/minkocc-dev:v3.0

# Proceed to 'Starting and Managing the Container'
```

### Option B: Build Docker Image from Source

If you need to make custom modifications, you can build the Docker image yourself.

```shell
# (Run on your host machine, inside the 'MinkOcc/docker' directory)
cd docker
./build_image.sh
```

#### Final Installation Steps (Inside the Container)

After building the image and entering the container with `./join_container.sh`, you must run two final installation scripts. These scripts are located in the `/workspace/` directory.

1.  **Install MinkowskiEngine**
    This script clones and installs the specific version of MinkowskiEngine required for this project.
    ```bash
    # (Run this command INSIDE the container)
    /workspace/necessary_patches/install_me_cuda.sh
    ```

2.  **Apply PyTorch3D Pulsar Patch**
    The Pulsar renderer used by PyTorch3D requires a patch to function correctly. This script applies the necessary changes.
    ```bash
    # (Run this command INSIDE the container)
    /workspace/necessary_patches/patch_pytorch3d_pulsar.sh
    ```

---

### Common Issues & Fixes

#### Issue: `mmcv` CUDA implementation not found

If you encounter an error like `RuntimeError: nms_impl: implementation for device cuda:0 not found.`, it means `mmcv` was not installed with the correct CUDA support.

#### Fix:

Run the following commands inside the container to reinstall it with the proper compilation flags.

```bash
# 1. Uninstall any existing mmcv versions
pip uninstall -y mmcv mmcv-full

# 2. Reinstall mmcv-full with the correct CUDA and PyTorch versions
pip install mmcv-full==1.5.2 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.1/index.html
```

---

## 2. Data and Model Preparation

### Step 2.1: Dataset Setup (nuScenes)

First, download the official **nuScenes dataset** (v1.0-trainval or v1.0-mini) and the pre-computed **Ground Truth (GTS) voxels** from the CVPR2023-3D-Occupancy-Prediction challenge.

Arrange your data using the following directory structure:

```
MinkOcc/
├── data/
│   └── nuscenes/
│       ├── v1.0-trainval/  # <- nuScenes data (maps, samples, sweeps, etc.)
│       └── gts/            # <- Downloaded Ground Truth voxels
└── ...
```

#### Mounting External Datasets

For large datasets, it is recommended to mount them from an external drive instead of copying them into the Docker container. To do this, modify the `volumes` section of your `docker-compose.yml` file.

**Example:**

```yaml
# In docker-compose.yml
services:
  minkocc-dev:
    # ... other configurations
    volumes:
      # Mounts your external drive's nuScenes path to the expected location inside the container
      - /media/your_user/external_drive/nuscenes:/workspace/data/nuscenes:rw
      # ... other volumes
```

### Step 2.2: Generate 2D Pseudo-Labels

You can enhance training with 2D pseudo-labels generated using a segmentation model.

1.  **Generate 2D Labels:** Use the provided Hugging Face script based on Grounded-Segment-Anything to generate segmentation masks. The labels should correspond to the following classes:
    ```python
    class_names_2d = [
        "background", "barrier", "bicycle", "bus", "sedan",
        "motorcycle", "crane", "highway", "people", "traffic_cone",
        "sidewalk", "truck", "building", "overhead bridge", "pole",
        "billboard", "tree", "sky",
    ]
    ```

2.  **Hugging Face Notebook:** The generation script can be found in the official repository: [grounded_sam.ipynb](https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/grounded_sam.ipynb).

3.  **Directory Structure:** Place the generated 2D pseudo-labels in a corresponding `*_SAM` directory for each camera. For example, labels for `CAM_BACK` should be placed in `CAM_BACK_SAM`.

    ```
    MinkOcc/
    └── data/
        └── nuscenes/
            ├── samples/
            │   ├── CAM_BACK/
            │   └── CAM_BACK_SAM/      # <- Pseudo-labels for CAM_BACK
            │   ├── CAM_FRONT/
            │   └── CAM_FRONT_SAM/     # <- Pseudo-labels for CAM_FRONT
            │   # ... and so on for all camera angles
            ├── v1.0-trainval/
            └── gts/
    ```

### Step 2.3: Download Pretrained Backbone

The model requires a ResNet50 backbone pretrained on ImageNet for image feature extraction.

```shell
# (Run on your host machine)
mkdir -p pretrained_checkpoints
wget https://download.pytorch.org/models/resnet50-0676ba61.pth -O pretrained_checkpoints/resnet50-0676ba61.pth
```

### Step 2.4: Generate Dataset Info Files (.pkl)

Finally, preprocess the nuScenes data to generate the pickle files required for training and evaluation.

```shell
# (Run this command INSIDE the container)

# For the full v1.0-trainval dataset
python tools/create_data_bevdet.py --data-root ./data/nuscenes --version v1.0-trainval --info-prefix minkoccv3-nuscenes-full

# For the mini dataset
python tools/create_data_bevdet.py --data-root ./data/nuscenes --version v1.0-mini --info-prefix minkoccv3-nuscenes-mini
```

---

## 3. Training

Use the `tools/train.py` script to start training. All configurations are located in the `configs/bevdet_occ/` directory. Before starting, review the config file (`minkoccv3_full.py`) to adjust parameters like supervision level, batch size, and learning rate.

```shell
# (Run all training commands INSIDE the container)

# Train MinkOccV3 on a single GPU
python tools/train.py configs/bevdet_occ/minkoccv3_full.py

# Resume training from a checkpoint
python tools/train.py configs/bevdet_occ/minkoccv3_full.py --resume-from /path/to/your/checkpoint.pth

# Train on multiple GPUs (e.g., 2 GPUs)
./tools/dist_train.sh configs/bevdet_occ/minkoccv3_full.py 2 --autoscale-lr
```

---

## 4. Evaluation and Visualization

### Evaluation

Evaluate the model's performance using standard occupancy prediction metrics like Mean Intersection over Union (mIoU).

```shell
# (Run this command INSIDE the container)
# Usage: python tools/test.py [config_file] [checkpoint_file] --eval [metric] --metric-type [type]

# Example evaluation
python3 tools/test.py configs/bevdet_occ/minkoccv3_full.py \
    ./work_dirs/minkoccv3_full/latest.pth \
    --eval miou --metric-type rayiou
```

### Visualization

To generate and save prediction visualizations, add the `--show-dir` flag to the test command.

```shell
# (Run this command INSIDE the container)
python3 tools/test.py configs/bevdet_occ/minkoccv3_mini.py \
    ./minkocc_checkpoint/epoch.pth \
    --show-dir ./prediction_visuals \
    --eval miou
```

---

## 5. Pretrained Models

| Model | Config File | Checkpoint |
|:---|:---|:---|
| MinkOccV3 | `minkoccv3_full.py` | [Coming soon] |
| MinkOccV3-mini | `minkoccv3_mini.py` | [Coming soon]|

---

## 6. Acknowledgements

This project would not be possible without leveraging multiple great open-source codebases. We extend our gratitude to the developers of:

*   [open-mmlab](https://github.com/open-mmlab)
*   [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)

---

## 7. BibTeX

```bibtex
@misc{sze2025minkoccrealtimelabelefficientsemantic,
      title={MinkOcc: Towards real-time label-efficient semantic occupancy prediction}, 
      author={Samuel Sze and Daniele De Martini and Lars Kunze},
      year={2025},
      eprint={2504.02270},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.02270}, 
}
```

If this work is helpful for your research, please also consider citing the original mmmdet, BEVDet as well.

```bibtex
@article{huang2023dal,
  title={Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection},
  author={Huang, Junjie and Ye, Yun and Liang, Zhujin and Shan, Yi and Du, Dalong},
  journal={arXiv preprint arXiv:2311.07152},
  year={2023}
}

@article{huang2021bevdet,
  title={BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View},
  author={Huang, Junjie and Huang, Guan and Zhu, Zheng and Yun, Ye and Du, Dalong},
  journal={arXiv preprint arXiv:2112.11790},
  year={2021}
}

@misc{mmdet3d2020,
    title={{MMDetection3D: OpenMMLab} next-generation platform for general {3D} object detection},
    author={MMDetection3D Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmdetection3d}},
    year={2020}
}
```