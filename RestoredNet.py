import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        #若x1、x2出现偏差，加入这样的 padding 操作来确保两个特征图能够完美对齐。
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class NestedUNet(nn.Module):
    def __init__(self, n_inchannels=3, n_outchannels=3):
        super(NestedUNet, self).__init__()
        self.n_channels = n_inchannels
        self.n_outchannels = n_outchannels

        self.inc = DoubleConv(n_inchannels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.up5 = Up(64, 32)
        # 多路径跳跃连接
        self.upcat1 = nn.Conv2d(1024, 512, kernel_size=1)
        self.upcat2 = nn.Conv2d(512, 256, kernel_size=1)
        self.upcat3 = nn.Conv2d(256, 128, kernel_size=1)
        self.upcat4 = nn.Conv2d(128, 64, kernel_size=1)
        self.outc = nn.Conv2d(32, n_outchannels, kernel_size=1)
        #补偿
        self.GP_conv0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.GP_pool1 = nn.MaxPool2d((2, 2))
        self.GP_conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.GP_pool2 = nn.MaxPool2d((4, 4))
        self.GP_conv2 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.GP_pool3 = nn.MaxPool2d((8, 8))
        self.resi1 =nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.resi2=nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.resi3=nn.Conv2d(512, 256, kernel_size=3, padding=1)

    def forward(self, x):
        GP0 = self.GP_conv0(x)#[16, 64, 224, 224]
        GP0 = self.GP_pool1(GP0)  # [16, 64, 112, 112]

        GP1 = self.GP_pool2(x)#[16, 3, 56, 56]
        GP1 = self.GP_conv1(GP1)#[16, 128, 56,56]

        GP2 = self.GP_pool3(x) #[16, 3, 28, 28]
        GP2 = self.GP_conv2(GP2)#[16, 256, 28, 28]

        x1 = self.inc(x)#16,32,224,224
        x2 = self.down1(x1)#16,64,112,112
        res1 = torch.cat([x2,GP0], dim=1)#16,128,112,112
        res1=self.resi1(res1)#16,64,112,112
        x3 = self.down2(res1)#16,128,56,56
        res2 = torch.cat([x3,GP1], dim=1)  # 16,256,56,56
        res2=self.resi2(res2)#16,128,56,56
        x4 = self.down3(res2)#16,256,28,28
        res3 = torch.cat([x4,GP2], dim=1)  # 16,512,28,28
        res3=self.resi3(res3)#16,256,28,28
        x5 = self.down4(res3)#16,512,14,14
        x6 = self.down5(x5)#16,1024,7,7

        up1 = self.up1(x6, x5)  # (16,512,14,14)
        upcat1 = torch.cat([up1, x5], dim=1)# (16,1024,14,14)
        d5 = self.upcat1(upcat1)  # (16,512,14,14)

        up2 = self.up2(d5, x4)# (16,256,28,28)
        upcat2 = torch.cat([up2, x4], dim=1)  # (16,512,28,28)
        d4 = self.upcat2(upcat2)  # (16,256,28,28)

        up3 = self.up3(d4, x3)#(16,128,56,56)
        upcat3 = torch.cat([up3, x3], dim=1)  # (16,256,56,56)
        d3 = self.upcat3(upcat3)  # (16,128,56,56)

        up4 = self.up4(d3, x2)#(16,64,112,112)
        upcat4 = torch.cat([up4, x2], dim=1)
        d2 = self.upcat4(upcat4)#16,64,112,112

        up5=self.up5(d2,x1)#16,32,224,224
        x = self.outc(up5)#16,3,224,224

        return x

class RestoredNet3(nn.Module):
    def __init__(self):
        super(RestoredNet3, self).__init__()
        self.NestedUNet = NestedUNet(n_inchannels=3, n_outchannels=3).cuda()
    def forward(self, input):

        input = input.view(-1, input.size(-3), input.size(-2), input.size(-1))
        restored = self.NestedUNet(input)
        return restored
