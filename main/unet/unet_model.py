#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
from .unet_parts import *
import pdb

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 8)
        self.down1 = down(8, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 8)
        self.up4 = up(16, 16)
        self.outc = outconv(16, n_classes)

    def forward(self, x):
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = F.max_pool2d(x5, 40)
        x7 = x6.view(-1, 512)
        x8 = F.relu(self.fc(x7))
        return x8

        x = self.conv(x)
        x = F.max_pool2d(x, 16)

        x = self.up1(x, x)
        x = self.up2(x, x)
        x = self.up3(x, x)
        x = self.up4(x, x)
        x = self.outc(x)
        return x
        """
        # x.shape == torch.Size([4, 3, 640, 640])

        x1 = self.inc(x)
        # x1.shape == torch.Size([4, 64, 640, 640])

        x2 = self.down1(x1)
        # x2.shape == torch.Size([4, 128, 320, 320])

        x3 = self.down2(x2)
        # x3.shape == torch.Size([4, 256, 160, 160])

        x4 = self.down3(x3)
        # x4.shape == torch.Size([4, 512, 80, 80])

        x5 = self.down4(x4)
        # x5.shape == torch.Size([4, 512, 40, 40])

        x = self.up1(x5, x4)
        # x.shape == torch.Size([4, 256, 80, 80])

        x = self.up2(x, x3)
        # x.shape == torch.Size([4, 128, 160, 160])

        x = self.up3(x, x2)
        # x.shape == torch.Size([4, 64, 320, 320])

        x = self.up4(x, x1)
        # x.shape == torch.Size([4, 64, 640, 640])

        x = self.outc(x)
        # x.shape == torch.Size([4, 1, 640, 640])

        return x
