import torch
from torch import nn
import torch.nn.functional as F
import settings
from itertools import combinations,product
import math

class Residual_Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(Residual_Block, self).__init__()
        self.channel_num = settings.channel
        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        # self.convert = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 3, stride, 1),
        #     nn.LeakyReLU(0.2)
        # )
        self.res = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x):
        convert = x
        out = convert + self.res(convert)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        idx = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return idx*self.sigmoid(x)

class eca_layer_avg(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3):
        super(eca_layer_avg, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return y.expand_as(x)


class eca_layer_max(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3):
        super(eca_layer_max, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.max_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return y.expand_as(x)
class Pyramid_module(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self):
        super(Pyramid_module, self).__init__()
        self.channel = settings.channel
        self.pool1 = nn.MaxPool2d(1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(4, 4)
        self.pool8 = nn.MaxPool2d(8, 8)
        self.conv11 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv21 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv41 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))

        self.conv12 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv22 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv42 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2),)

        self.conv13 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2),
                                    nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv23 = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2),
                                    nn.Conv2d(self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))

        self.spa1 = SpatialAttention(kernel_size=3)
        self.spa2 = SpatialAttention(kernel_size=3)
        self.spa4 = SpatialAttention(kernel_size=3)
        self.eca1 = eca_layer_avg()
        self.eca2 = eca_layer_avg()
        self.eca4 = eca_layer_avg()

    def forward(self, x):
        pool1 = self.pool1(x)
        b1, c1, h1, w1 = pool1.size()
        pool2 = self.pool2(x)
        b2, c2, h2, w2 = pool2.size()
        pool4 = self.pool4(x)

        conv11 = self.conv11(pool1)
        conv21 = self.conv21(pool2)
        conv41 = self.conv41(pool4)

        spa1 = self.spa1(conv11)
        spa2 = self.spa2(conv21)
        spa4 = self.spa4(conv41)

        eca1 = conv11 * self.eca1(spa1)
        eca2 = conv21 * self.eca2(spa2)
        eca4 = conv41 * self.eca4(spa4)

        conv12 = self.conv12(eca1)
        conv22 = self.conv22(eca2)
        conv42 = self.conv42(eca4)

        conv23 = self.conv23(conv22 + F.upsample(conv42, [h2, w2]))
        conv13 = self.conv13(conv12 + F.upsample(conv23, [h1, w1]) + F.upsample(conv42, [h1, w1]))

        out = conv13 + x
        return out

class DenseConnection(nn.Module):
    def __init__(self, unit_num):
        super(DenseConnection, self).__init__()
        self.unit_num = unit_num
        self.channel = settings.channel
        self.units = nn.ModuleList()
        self.conv1x1 = nn.ModuleList()
        for i in range(self.unit_num):
            self.units.append(Pyramid_module())
            self.conv1x1.append(nn.Sequential(nn.Conv2d((i+2)*self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2)))
    
    def forward(self, x):
        cat = []
        cat.append(x)
        out = x
        for i in range(self.unit_num):
            tmp = self.units[i](out)
            cat.append(tmp)
            out = self.conv1x1[i](torch.cat(cat,dim=1))
        return out


class ODE_DerainNet(nn.Module):
    def __init__(self):
        super(ODE_DerainNet, self).__init__()
        self.channel = settings.channel
        # self.unit_num = settings.unit_num
        self.enterBlock1 = nn.Sequential(nn.Conv2d(3, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.enterBlock2 = nn.Sequential(nn.Conv2d(3, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.up_net = DenseConnection(16)
        self.net1 = DenseConnection(4)
        self.net2 = DenseConnection(12)
        self.net3 = DenseConnection(32)
        self.exitBlock1 = nn.Sequential(nn.Conv2d(self.channel, 3, 3, 1, 1))
        self.exitBlock2 = nn.Sequential(nn.Conv2d(self.channel, 3, 3, 1, 1))
        self.cat1 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
        self.cat2 = nn.Sequential(nn.Conv2d(2 * self.channel, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))

    def forward(self, x):  
        feature1 = self.enterBlock1(x)
        feature2 = self.enterBlock2(x)

        net1 = self.net1(feature2)
        net2 = self.net2(net1)

        up_net = self.up_net(self.cat1(torch.cat([net1,feature1],dim=1)))

        net3 = self.net3(self.cat2(torch.cat([net2,up_net],dim=1)))

        rain1 = self.exitBlock1(up_net)
        derain1 = x - rain1

        rain2 = self.exitBlock2(net3)
        derain2 = x - rain2

        return derain1, derain2
