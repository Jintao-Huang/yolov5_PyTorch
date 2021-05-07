# Author: Jintao Huang
# Date: 2021-4-26
import torch.nn as nn
import torch


class ConvBnSiLU(nn.Module):
    # Standard convolution
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, activation=True):
        super(ConvBnSiLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, 1e-3, 0.03)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super(Bottleneck, self).__init__()
        neck = int(out_channels * expansion)
        self.conv1 = ConvBnSiLU(in_channels, neck, 1, 1, 0, True)
        self.conv2 = ConvBnSiLU(neck, out_channels, 3, 1, 1, True)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        x0 = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = (x0 + x) if self.shortcut else x
        return x


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, in_channels, out_channels, repeat=1, shortcut=True, expansion=0.5):
        super(C3, self).__init__()
        neck = int(out_channels * expansion)
        self.conv1 = ConvBnSiLU(in_channels, neck, 1, 1, 0, True)
        self.conv2 = ConvBnSiLU(in_channels, neck, 1, 1, 0, True)
        self.bottleneck_n = nn.Sequential(*[Bottleneck(neck, neck, shortcut, 1.0) for _ in range(repeat)])
        self.conv3 = ConvBnSiLU(2 * neck, out_channels, 1, 1, 0, True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bottleneck_n(x1)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3(x)
        return x


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, in_channels, out_channels, kernel_size_list=(5, 9, 13)):
        super(SPP, self).__init__()
        neck = in_channels // 2
        self.conv1 = ConvBnSiLU(in_channels, neck, 1, 1, 0)
        self.conv2 = ConvBnSiLU(neck * (len(kernel_size_list) + 1), out_channels, 1, 1, 0)
        self.max_pool_list = nn.ModuleList(
            [nn.MaxPool2d(kernel_size, 1, kernel_size // 2) for kernel_size in kernel_size_list])

    def forward(self, x):
        x = self.conv1(x)
        x0 = x
        x = [max_pool(x) for max_pool in self.max_pool_list]
        x = torch.cat([x0] + x, 1)
        x = self.conv2(x)
        return x


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=True):
        super(Focus, self).__init__()
        self.conv = ConvBnSiLU(in_channels * 4, out_channels, kernel_size, stride, padding, activation)

    def forward(self, x):
        x = torch.cat([x[:, :, ::2, ::2], x[:, :, 1::2, ::2], x[:, :, ::2, 1::2], x[:, :, 1::2, 1::2]], 1)
        x = self.conv(x)
        return x


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, self.dim)
