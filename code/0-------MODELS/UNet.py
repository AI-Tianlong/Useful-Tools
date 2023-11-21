""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):   #做两次一模一样的卷积
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):   #中间的通道  ch1-> mid ch ->ch2
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""       #下采样，做完maxpool 然后做两次卷积，尺寸减半，通道翻倍，

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""                #上采样，如果是双线性插值，用普通的卷积来减少通道数，如果没有双线性，则用转置卷积

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)  #对x1进行 上采样，转置卷积/普通的卷积，让他通道数翻倍
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]    #将X1 填充到和X2尺寸一致
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)  #输入的3通道变为64  Xen1
        self.down1 = Down(64, 128) #64 -> 128                   Xen2
        self.down2 = Down(128, 256) #128 -> 256                 Xen3
        self.down3 = Down(256, 512)# 256 -> 512                 Xen4
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)                  #Xen5,Xde5
        self.up1 = Up(1024, 512 // factor, bilinear)  #上采样    Xde4
        self.up2 = Up(512, 256 // factor, bilinear)  #上采样     Xde3
        self.up3 = Up(256, 128 // factor, bilinear)  #上采样     Xde2
        self.up4 = Up(128, 64, bilinear)                    #   Xde1
        self.outc = OutConv(64, n_classes)                  # 最后的输入，是类别数的通道

    def forward(self, x):
        x1 = self.inc(x)                                  #输入变为Xen1
        x2 = self.down1(x1)                               #Xen1 -> Xen2
        x3 = self.down2(x2)                               #Xen2 -> Xen3
        x4 = self.down3(x3)                               #Xen3 -> Xen4
        x5 = self.down4(x4)                               #Xen4 -> Xen5
        x = self.up1(x5, x4)                              #Xen5 -> Xen4
        x = self.up2(x, x3)                               #把Xen4 -> Xen3
        x = self.up3(x, x2)                               #Xen3 -> Xen2
        x = self.up4(x, x1)                               #Xen2 -> Xen1
        logits = self.outc(x)                             #Xen1 输出来
        return logits
