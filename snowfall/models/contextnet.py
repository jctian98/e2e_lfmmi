#!/usr/bin/env python3

# Copyright (c)  2021  University of Chinese Academy of Sciences (author: Han Zhu)
# Apache 2.0


import torch
from torch import Tensor, nn
from typing import List
from snowfall.models import AcousticModel
from snowfall.models.conformer import Swish


class ContextNet(AcousticModel):
    """ContextNet. Reference: https://arxiv.org/pdf/2005.03191.pdf

    Args:
        num_features (int): Number of input features
        num_classes (int): Number of output classes
        kernel_size (int): Kernel size of convolution layers (default 3).
        num_blocks (int): Number of context block (default 6).
        num_layers (int): Number of depthwise convolution layers for each 
                context block (except first and last block) (default 5).
        conv_out_channels (List[int]): Number of output channels produced by context blocks, 
                len(conv_out_channels) = num_blocks (default [*[256] * 2, *[512] * 3, 640]).
        subsampling_layers (List[int]): Indexs of subsampling layers (default [1, 3]).
        alpha (float): The factor to scale the output channel of the network (default 1.5).
        dropout (float): Dropout (default 0.1).
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        kernel_size: int = 3,
        num_blocks: int = 6,
        num_layers: int = 5,
        conv_out_channels: List[int] = [*[256] * 2, *[512] * 3, 640],
        subsampling_layers: List[int] = [1, 3],
        alpha: float = 1.5,
        dropout: int = 0.1,
    ):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = 2 * len(subsampling_layers)

        conv_channels = [num_features] +  \
                [int(channels * alpha) for channels in conv_out_channels]

        strides = [1] * num_blocks
        for layer in subsampling_layers:
            strides[layer] = 2
            strides[layer] = 2

        residuals = [False, *[True] * (num_blocks - 2), False ] 

        blocks_num_layers = [1, *[num_layers] * (num_blocks - 2), 1 ] 

        self.block_list = [
            ContextNetBlock(
                conv_channels[i],
                conv_channels[i+1],
                kernel_size=kernel_size,
                stride=strides[i],
                num_layers=blocks_num_layers[i],
                dropout=dropout,
                residual=residuals[i]
            ) for i in range(num_blocks)]

        self.blocks = nn.Sequential(*self.block_list)

        self.output_layer = nn.Linear(conv_channels[-1], num_classes)
    
    def forward(self, x, supervision = None):
        """
        Args:
            x (torch.Tensor): Input tensor (batch, channels, time).
            supervision: Supervison in lhotse format, get from batch['supervisions'].
                        It's not used here, just to keep consistent with transformer.

        Returns:
            torch.Tensor: Output tensor (batch, channels, time).
        """
        x = x.transpose(1, -1)
        x = self.blocks(x)
        x = self.output_layer(x)
        x = nn.functional.log_softmax(x, dim=-1).transpose(1, -1)
        return x, None, None


class ContextNetBlock(torch.nn.Module):
    """A block in ContextNet.

    Args:
        in_channels (int): Number of output channels of this model.
        out_channels (int): Number of input channels of this model.
        kernel_size (int) : Kernel size of convolution layers (default 3).
        stride (int): Stride of this context block (default 1).
        num_layers (int): Number of depthwise convolution layers for this context block (default 5).
        dropout (float): Dropout (default 0.1).
        residual (bool): Whether to apply residual connection at this context block (default None).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        num_layers: int = 1,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()

        self.convs_list = [
            ConvModule(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride if i == num_layers - 1 else 1,
                padding=kernel_size // 2 - stride + 1 if i == num_layers - 1 else kernel_size // 2
            ) for i in range(num_layers)]

        self.convs = nn.Sequential(*self.convs_list)

        self.SE = SEModule(channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

        self.drop = nn.Dropout(dropout)

        if residual:
            self.residual = ConvModule(in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2 - stride + 1,
                stride=stride,
                activation=None)
        else:
            self.residual = None
        
        self.activation = Swish()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (batch, time, channels).
        """
        out = self.convs(x)
        out = self.SE(out)
        if self.residual:
            out = out + self.residual(x)
        out = self.activation(out)
        out = self.drop(out)
        return out


class SEModule(torch.nn.Module):
    """Squeeze-and-Excitation module.

    Args:
        channels (int): Input and output channels.
        kernel_size (int) : Kernel size of convolution layers (default 3).
        padding (int): Zero-padding added to both sides of the input.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        padding: int
    ):
        super().__init__()

        self.conv = ConvModule(channels, channels, kernel_size=kernel_size, padding=padding, stride=1)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.bottleneck = nn.Sequential(
            torch.nn.Linear(channels, channels // 8),
            Swish(),
            torch.nn.Linear(channels // 8, channels),
            Swish(),
        )

        self.final_act = torch.nn.Sigmoid()

    def forward(self, x):
        """Squeeze and excitation

        Args:
            x (torch.Tensor): Input tensor (batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (batch, time, channels).
        """
        B, T, C = x.shape

        x = self.conv(x).transpose(1, -1) # (B, C, T)
        avg = self.avg_pool(x).transpose(1,-1) # (B, 1, C)
        avg = self.bottleneck(avg)
        avg = self.final_act(avg)
        context = avg.repeat(1, T, 1) # (B, T, C)
        out = x.transpose(1, -1) * context
        return out


class ConvModule(torch.nn.Module):
    """
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution (default 1).
        dilation (int): Spacing between kernel elements (default 1).
        padding (int): Zero-padding added to both sides of the input.
        padding_mode (str): 'zeros', 'reflect', 'replicate' or 'circular' (default 'zeros').
        bias (bool): If True, adds a learnable bias to the output (default: True).
        activation (object): activation function used in this convolution module. (default: Swish)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        padding_mode : str = 'zeros',
        bias: bool = True,
        activation = Swish
    ):
        super().__init__()

        self.conv = SeparableConv1D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode=padding_mode,
            bias=bias,
        )

        self.norm = torch.nn.BatchNorm1d(out_channels)

        if activation:
            self.activation = activation()
        else:
            self.activation = None

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (batch, new_time, channels).
        """
        x = self.conv(x).transpose(1, -1) # (B, C, T)
        x = self.norm(x)
        if self.activation:   
            x = self.activation(x)
        x = x.transpose(1, -1) # (B, T, C)
        return x


class SeparableConv1D(nn.Module):
    """Depthwise separable 1D convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution (default 1).
        dilation (int): Spacing between kernel elements (default 1).
        padding (int): Zero-padding added to both sides of the input.
        padding_mode (str): 'zeros', 'reflect', 'replicate' or 'circular' (default 'zeros').
        bias (bool): If True, adds a learnable bias to the output (default: True).

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        padding_mode : str = 'zeros',
        bias: bool = True,
    ):
        super().__init__()

        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode=padding_mode,
            groups=in_channels,
            bias=bias,
        )

        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (batch, time, channels).
        """
        x = x.transpose(1, -1) # (B, C, T)
        x = self.pointwise(self.depthwise(x)).transpose(1, -1) # (B, T, C)
        return x