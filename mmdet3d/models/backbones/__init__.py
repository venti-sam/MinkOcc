# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.backbones import SSDVGG, HRNet, ResNet, ResNetV1d, ResNeXt
from .mink_resnet import MinkResNet
from .minkocc_resnet import TR3DMinkResNet
from .multi_backbone import MultiBackbone
from .nostem_regnet import NoStemRegNet
from .resnet import CustomResNet, CustomResNet3D
from .second import SECOND
from .swin import SwinTransformer

__all__ = [
    'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'NoStemRegNet',
    'SECOND',
    'MultiBackbone','MinkResNet', 'CustomResNet', 'CustomResNet3D',
    'SwinTransformer', 'TR3DMinkResNet'
]
