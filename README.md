# MinkOcc

## Get Started

#### Installation and Data Preparation

a. Create a conda virtual environment and activate it.

```shell script
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the official instructions.

```shell script
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu114 -f https://download.pytorch.org/whl/torch_stable.html
Recommended torch>=1.12
```

c. Install mmcv-full.

```shell script
pip install mmcv-full==1.5.2
```

d. Install mmdet and mmseg.

```shell script
pip install mmdet==2.24.0
pip install mmsegmentation==0.24.0
```

e. Prepare MinkOcc repo by.

```shell script
git clone https://github.com/venti-sam/MinkOcc.git
cd MinkOcc
pip install -v -e .
```

f. Download Nuscenes Mini dataset:

```shell script
https://www.nuscenes.org/nuscenes#download
```

step 3. Prepare nuScenes dataset as introduced in [nuscenes_det.md](docs/en/datasets/nuscenes_det.md) and create the pkl for MinkOcc by running:

```shell
python tools/create_data_bevdet.py
```

g. For Occupancy Prediction task, download the mini and (only) the 'gts' from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) and arrange the folder as:

```shell script
└── nuscenes
    ├── v1.0-mini (existing)
    ├── sweeps  (existing)
    ├── samples (existing)
    └── gts (new)
```

#### Train model

```shell script
# single gpu
python tools/train.py configs/bevdet_occ/bevdet_minkocc.py
```

#### Test model

# single gpu

```shell script
python tools/test.py $config $checkpoint --eval mAP
```

## Acknowledgement

This project is not possible without multiple great open-sourced code bases. We list some notable examples below.

- [open-mmlab](https://github.com/open-mmlab)
- [CenterPoint](https://github.com/tianweiy/CenterPoint)
- [Lift-Splat-Shoot](https://github.com/nv-tlabs/lift-splat-shoot)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)

Beside, there are some other attractive works extend the boundary of BEVDet.

- [BEVerse](https://github.com/zhangyp15/BEVerse) for multi-task learning.
- [BEVStereo](https://github.com/Megvii-BaseDetection/BEVStereo) for stero depth estimation.

## Bibtex

If this work is helpful for your research, please consider citing the following BibTeX entries.

````

@article{huang2023dal,
title={Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection},
author={Huang, Junjie and Ye, Yun and Liang, Zhujin and Shan, Yi and Du, Dalong},
journal={arXiv preprint arXiv:2311.07152},
year={2023}
}

@article{huang2022bevpoolv2,
title={BEVPoolv2: A Cutting-edge Implementation of BEVDet Toward Deployment},
author={Huang, Junjie and Huang, Guan},
journal={arXiv preprint arXiv:2211.17111},
year={2022}
}

@article{huang2022bevdet4d,
title={BEVDet4D: Exploit Temporal Cues in Multi-camera 3D Object Detection},
author={Huang, Junjie and Huang, Guan},
journal={arXiv preprint arXiv:2203.17054},
year={2022}
}

@article{huang2021bevdet,
title={BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View},
author={Huang, Junjie and Huang, Guan and Zhu, Zheng and Yun, Ye and Du, Dalong},
journal={arXiv preprint arXiv:2112.11790},
year={2021}
}

```

```
````
