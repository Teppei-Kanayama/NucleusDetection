#!/usr/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear and False:
            #self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.up = nn.Upsample(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

# UNet本体
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        """
        self.inc = InConv(n_channels, 8)
        self.down1 = Down(8, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 8)
        self.up4 = Up(16, 16)
        self.outc = OutConv(16, n_classes)
        """
        self.inc = InConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.up4 = Up(32, 16)
        self.outc = OutConv(16, n_classes)


    def forward(self, x):
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
