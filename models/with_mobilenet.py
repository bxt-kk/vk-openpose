from typing import Tuple, Callable
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.conv import conv, conv_dw, conv_dw_no_bn

# <<<
DEFAULT_NORM_LAYER = nn.BatchNorm2d # LayerNorm2d
DEFAULT_ACTIVATION = nn.Hardswish
DEFAULT_SIGMOID    = nn.Hardsigmoid


class ConvNormActive(nn.Sequential):

    def __init__(
            self,
            in_planes:   int,
            out_planes:  int,
            kernel_size: int | Tuple[int, int]=3,
            stride:      int | Tuple[int, int]=1,
            dilation:    int | Tuple[int, int]=1,
            groups:      int=1,
            norm_layer:  Callable[..., nn.Module] | None=DEFAULT_NORM_LAYER,
            activation:  Callable[..., nn.Module] | None=DEFAULT_ACTIVATION,
        ):

        if isinstance(kernel_size, int):
            kernel_size = kernel_size, kernel_size
        if isinstance(dilation, int):
            dilation = dilation, dilation
        padding = tuple(
            (ks + 2 * (dl - 1) - 1) // 2 for ks, dl in zip(kernel_size, dilation))
        layers = [nn.Conv2d(
            in_planes, out_planes, kernel_size, stride, padding, dilation, groups=groups),
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_planes))
        if activation is not None:
            layers.append(activation())
        super().__init__(*layers)


class ChannelAttention(nn.Module):

    def __init__(
            self,
            in_planes:     int,
            shrink_factor: int=4,
        ):

        super().__init__()

        shrink_dim = in_planes // shrink_factor
        self.dense = nn.Sequential(
            nn.Conv2d(in_planes, shrink_dim, 1),
            DEFAULT_ACTIVATION(inplace=False),
            nn.Conv2d(shrink_dim, in_planes, 1),
            DEFAULT_SIGMOID(inplace=False),
        )

    def forward(self, x:Tensor) -> Tensor:
        f = F.adaptive_avg_pool2d(x, 1)
        return self.dense(f) * x


class SpatialAttention(nn.Module):

    def __init__(
            self,
            in_planes:   int,
            kernel_size: int=7,
        ):

        super().__init__()

        self.fc = ConvNormActive(in_planes, 1, 1, norm_layer=None)
        self.dense = ConvNormActive(
            2, 1, kernel_size, norm_layer=None, activation=DEFAULT_SIGMOID)

    def forward(self, x:Tensor) -> Tensor:
        f = torch.cat([
            x.mean(dim=1, keepdim=True),
            self.fc(x),
        ], dim=1)
        return self.dense(f) * x


class CBANet(nn.Module):

    def __init__(
            self,
            in_planes:     int,
            out_planes:    int,
            kernel_size:   int=7,
            shrink_factor: int=4,
            norm_layer:    Callable[..., nn.Module]=DEFAULT_NORM_LAYER,
        ):

        super().__init__()

        self.channel_attention = ChannelAttention(in_planes, shrink_factor)
        self.spatial_attention = SpatialAttention(in_planes, kernel_size)
        self.project = ConvNormActive(
            in_planes, out_planes, 1, norm_layer=norm_layer, activation=None)

        self.alpha = nn.Parameter(torch.zeros(1, ))

    def forward(self, x:Tensor) -> Tensor:
        x0 = x
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        x = self.project(x)
        return x0 * (1 - self.alpha) + x * self.alpha
# >>>


class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        self.model = nn.Sequential(
            conv(     3,  32, stride=2, bias=False),
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5
        )
        self.cpm = Cpm(512, num_channels)

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))

        self.cpm_out_channels = num_channels
        self.ext_modules = {}

    def insert_cbanet(self) -> CBANet:
        module = CBANet(self.cpm_out_channels, self.cpm_out_channels)
        self.ext_modules['cbanet'] = module
        return module

    def forward(self, x):
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        cbanet = self.ext_modules.get('cbanet')
        if cbanet:
            backbone_features = cbanet(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))

        return stages_output
