import torch
import torch.nn as nn
from .edl import NormalInvGammaConv2d
import math
from collections import namedtuple

import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum, nn

# Two U-Net implementations are provided, the latter is used currently.
class DoubleConv(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.activ = nn.SiLU() # or relu
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, kernel_size=3, stride=1, padding=1),
            self.activ,
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=1, padding=1),
            self.activ
        )
    
    def forward(self, x):
        return self.nn(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, feature_sizes=[64,128,256,512], evidential=False):
        super().__init__()
        self.down_ops = nn.ModuleList()
        self.pool_ops = nn.ModuleList()
        prev_channels = in_channels
        for feature in feature_sizes:
            self.down_ops.append(DoubleConv(prev_channels, feature))
            self.pool_ops.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = feature

        self.bottleneck = DoubleConv(feature_sizes[-1], feature_sizes[-1]*2)

        self.up_ops = nn.ModuleList()
        for feature in reversed(feature_sizes):
            self.up_ops.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.up_ops.append(
                DoubleConv(feature*2, feature)
            )

        self.final_conv = nn.Conv2d(feature_sizes[0], feature_sizes[0] // 2, kernel_size=1)
        if evidential:
            self.using_evidential = True
            self.evidential = NormalInvGammaConv2d(feature_sizes[0] // 2, out_channels)


    def forward(self, x):
        skip_connections = []
        for down, pool in zip(self.down_ops, self.pool_ops):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for i, up in enumerate(self.up_ops):
            if i % 2 == 0:
                x = up(x) # ConvTranspose2d
                skip_connection = skip_connections.pop()
                x = torch.cat([x, skip_connection], dim=1)
            else:
                x = up(x) # DoubleConv

        x = self.final_conv(x)
        if self.using_evidential:
            mu, v, alpha, beta = self.evidential(x)
            return mu, v, alpha, beta
        return x
    


class EvidentialUNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.bottleneck = self.conv_block(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self.conv_block(64, 32)
        self.evidential = NormalInvGammaConv2d(32, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.evidential(d1)