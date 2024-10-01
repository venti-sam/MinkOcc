# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/tr3d/blob/master/mmdet3d/models/necks/tr3d_neck.py # noqa
from typing import List, Tuple

try:
    import MinkowskiEngine as ME
    from MinkowskiEngine import SparseTensor
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    ME = SparseTensor = None
    pass

from mmcv.runner import BaseModule
import torch
from torch import nn


from mmdet3d.models.builder import NECKS


@NECKS.register_module()
class TR3DNeck(BaseModule):
    r"""U-Net-like Neck for TR3D.

    Args:
        in_channels (tuple[int]): Number of channels in input tensors.
        out_channels (int): Number of channels in output tensors (e.g., 32).
        strides (tuple[int]): Strides for each input tensor.
    """

    def __init__(self, 
                 in_channels: Tuple[int], 
                 out_channels: int, 
                 strides: Tuple[int], 
                 loss_bce_weight=1.0,
                 is_generative=False):
        super(TR3DNeck, self).__init__()
        # Store the strides for each level
        self.strides = strides
        # Store the desired output channels
        self.out_channels = out_channels
        # if use generative convolution transpose
        self.is_generative = is_generative
        # BCE loss weight
        self.loss_bce_weight = loss_bce_weight
        self.bce_criterion = nn.BCEWithLogitsLoss()
        # Initialize layers
        self._init_layers(in_channels)

    def _init_layers(self, in_channels: Tuple[int]):
        """Initialize layers.

        Args:
            in_channels (tuple[int]): Number of channels in input tensors.
        """
        num_levels = len(in_channels)
        # Lists to hold upsampling layers and fusion convolutions
        self.upsample_layers = nn.ModuleList()
        # List to hold classification layers
        if self.is_generative:
            self.classification_layer = nn.ModuleList()
        # self.fuse_convs = nn.ModuleList()
        # Loop over levels from deepest to the highest resolution
        for i in range(num_levels - 1, 0, -1):
            # Kernel size is 4 for the deepest two levels, rest is 2
            kernel_size = 4 if i >= num_levels - 2 else 2
    
            # Compute upsampling factor between current level and higher resolution level
            up_factor = self.strides[i] // self.strides[i - 1]
            # Create upsample layer using _make_block
            self.upsample_layers.append(
                self._make_block(
                    in_channels=in_channels[i],         # Input channels from current level
                    out_channels=in_channels[i - 1],    # Output channels to match higher level
                    generative=self.is_generative,      # Use generative convolution transpose
                    stride=up_factor,                  # Upsampling factor
                    kernel_size = kernel_size
                )
            )
            if self.is_generative:
                self.classification_layer.append(
                    ME.MinkowskiConvolution(
                        in_channels[i - 1],     # Input channels from the higher resolution level
                        1,                     # Output channels (1 for binary classification)
                        kernel_size=1,
                        stride=1,
                        dimension=3
                    )
                )
        conv = ME.MinkowskiGenerativeConvolutionTranspose if self.is_generative \
            else ME.MinkowskiConvolutionTranspose
        # Final convolution to adjust channels to out_channels (e.g., 32)
        self.final_conv = nn.Sequential(
            conv(
                in_channels[0],     # Input channels from the highest resolution level
                self.out_channels,  # Desired number of output channels
                kernel_size=2,
                stride=2,
                dimension=3
            ),
            ME.MinkowskiBatchNorm(self.out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                self.out_channels,   
                self.out_channels, 
                kernel_size=3,
                stride=1,
                dimension=3
            ),
            ME.MinkowskiBatchNorm(self.out_channels),
            ME.MinkowskiReLU(inplace=True)
        )
        
        if self.is_generative:
            self.final_cls = ME.MinkowskiConvolution(
                self.out_channels,     # Input channels from the highest resolution level
                1,                     # Output channels (1 for binary classification)
                kernel_size=1,
                stride=1,
                dimension=3
            )
            # Pruning layer
            self.pruning = ME.MinkowskiPruning()

    def init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (ME.MinkowskiConvolution, ME.MinkowskiConvolutionTranspose)):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
            cm = out.coordinate_manager
            strided_target_key = cm.stride(
                target_key,
                out.tensor_stride[0],
            )
            kernel_map = cm.kernel_map(
                out.coordinate_map_key,
                strided_target_key,
                kernel_size=kernel_size,
                region_type=1,
            )
            for k, curr_in in kernel_map.items():
                target[curr_in[0].long()] = 1
        return target


    def forward(self, x: List[SparseTensor], target_key=None) -> SparseTensor:
        """Forward pass.

        Args:
            x (list[SparseTensor]): Features from the backbone.
            target_key : The coordinate key of the targets in the batch (always provided if is_generative).

        Returns:
            SparseTensor: Output feature from the neck.
        """
        # Start from the deepest level's features
        out = x[-1]
        out_cls, targets = [], []
        
        # Iterate from deepest level to highest resolution level
        for i in range(len(x) - 1, 0, -1):
            idx = len(x) - 1 - i  # Indexing for upsample_layers and fuse_convs
            
            # Upsample current output to match resolution and channels of higher level
            up_feat = self.upsample_layers[idx](out)
            
            # Fuse upsampled features with features from higher resolution level
            out = up_feat + x[i - 1]
            
            # Add classification pruning logic for generative models
            if self.is_generative:
                # Compute classification for the current level
                out_curr_cls = self.classification_layer[idx](out)
                keep = (out_curr_cls.F > 0).squeeze()  # Binary mask for pruning
                
                # Target is always available if the model is generative
                target = self.get_target(out, target_key)
                targets.append(target)
                out_cls.append(out_curr_cls)
                
                # During training, add target information to the keep mask
                if self.training:
                    keep = keep + target
                
                # Prune features based on the keep mask
                if keep.sum() > 0:
                    out = self.pruning(out, keep)
            
            # Apply fusion convolution (if needed)
            # out = self.fuse_convs[idx](out)
        
        # Apply final convolution to adjust channels to out_channels
        out = self.final_conv(out)
        
        # Final classification pruning (for generative models)
        if self.is_generative:
            out_curr_cls = self.final_cls(out)
            keep = (out_curr_cls.F > 0).squeeze()
            
            # Target is always available for final pruning step
            target = self.get_target(out, target_key)
            targets.append(target)
            out_cls.append(out_curr_cls)
            
            # Prune features based on final keep mask
            if keep.sum() > 0:
                out = self.pruning(out, keep)
        
        # If not generative, out_cls and targets will remain as empty lists
        return out_cls, targets, out

    def get_bce_loss(self, out_cls, targets):
        device = out_cls[0].device
        num_layers, bce_loss = len(out_cls), 0
        bce_losses = []
        for out_cl, target in zip(out_cls, targets):
            curr_loss = self.bce_criterion(
                out_cl.F.squeeze(), target.type(out_cl.F.dtype).to(device)
            )
            bce_losses.append(curr_loss.item())
            bce_loss += curr_loss / num_layers
        return bce_loss * self.loss_bce_weight

    @staticmethod
    def _make_block(in_channels: int,
                    out_channels: int,
                    generative: bool = True,
                    stride: int = 1,
                    kernel_size: int = 2,
        ) -> nn.Module:
        """Construct upsample block using Minkowski Convolution Transpose.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (match higher level channels).
            generative (bool): Use generative convolution transpose if True.
            stride (int): Upsampling factor.

        Returns:
            nn.Module: Upsampling block.
        """
        # Choose the appropriate convolution transpose based on the generative flag
        conv = ME.MinkowskiGenerativeConvolutionTranspose if generative \
            else ME.MinkowskiConvolutionTranspose
        # Create the upsample block
        return nn.Sequential(
            conv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dimension=3
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                dimension=3
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
        )