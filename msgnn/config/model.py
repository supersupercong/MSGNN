import torch
from torch import nn
import torch.nn.functional as F
import settings
from itertools import combinations, product
import math
import numpy as np
from torch import einsum
# from einops import rearrange
from graph.submodules import *


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# def calc_rel_pos(n):
#     pos = torch.meshgrid(torch.arange(n), torch.arange(n))
#     pos = rearrange(torch.stack(pos), 'n i j -> (i j) n')  # [n*n, 2] pos[n] = (i, j)
#     rel_pos = pos[None, :] - pos[:, None]  # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
#     rel_pos += n - 1  # shift value range from [-n+1, n-1] to [0, 2n-2]
#     return rel_pos


# lambda layer

# class LambdaLayer(nn.Module):
#     def __init__(self, dim=32, *, dim_k=12, n=48, r=None, heads=4, dim_out=None, dim_u=1):
#         super().__init__()
#         dim_out = default(dim_out, dim)
#         self.u = dim_u  # intra-depth dimension
#         self.heads = heads
#
#         # dim = 32,  # channels going in
#         # dim_out = 32,  # channels out
#         # n = 64,  # size of the receptive window - max(height, width)
#         # dim_k = 16,  # key dimension
#         # heads = 4,  # number of heads, for multi-query
#         # dim_u = 1  # 'intra-depth' dimension
#
#         assert (dim_out % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
#         dim_v = dim_out // heads
#
#         self.to_q = nn.Conv2d(dim, dim_k * heads, 1, bias=False)
#         self.to_k = nn.Conv2d(dim, dim_k * dim_u, 1, bias=False)
#         self.to_v = nn.Conv2d(dim, dim_v * dim_u, 1, bias=False)
#
#         self.norm_q = nn.BatchNorm2d(dim_k * heads)
#         self.norm_v = nn.BatchNorm2d(dim_v * dim_u)
#
#         self.local_contexts = exists(r)
#         if exists(r):
#             assert (r % 2) == 1, 'Receptive kernel size should be odd'
#             self.pos_conv = nn.Conv3d(dim_u, dim_k, (1, r, r), padding=(0, r // 2, r // 2))
#         else:
#             assert exists(n), 'You must specify the window size (n=h=w)'
#             rel_lengths = 2 * n - 1
#             self.rel_pos_emb = nn.Parameter(torch.randn(rel_lengths, rel_lengths, dim_k, dim_u))
#             self.rel_pos = calc_rel_pos(n)
#
#     def forward(self, x):
#         b, c, hh, ww, u, h = *x.shape, self.u, self.heads
#
#         q = self.to_q(x)
#         k = self.to_k(x)
#         v = self.to_v(x)
#
#         q = self.norm_q(q)
#         v = self.norm_v(v)
#
#         q = rearrange(q, 'b (h k) hh ww -> b h k (hh ww)', h=h)
#         k = rearrange(k, 'b (u k) hh ww -> b u k (hh ww)', u=u)
#         v = rearrange(v, 'b (u v) hh ww -> b u v (hh ww)', u=u)
#
#         k = k.softmax(dim=-1)
#
#         λc = einsum('b u k m, b u v m -> b k v', k, v)
#         Yc = einsum('b h k n, b k v -> b h v n', q, λc)
#
#         if self.local_contexts:
#             v = rearrange(v, 'b u v (hh ww) -> b u v hh ww', hh=hh, ww=ww)
#             λp = self.pos_conv(v)
#             Yp = einsum('b h k n, b k v n -> b h v n', q, λp.flatten(3))
#         else:
#             n, m = self.rel_pos.unbind(dim=-1)
#             rel_pos_emb = self.rel_pos_emb[n, m]
#             λp = einsum('n m k u, b u v m -> b n k v', rel_pos_emb, v)
#             Yp = einsum('b h k n, b n k v -> b h v n', q, λp)
#
#         Y = Yc + Yp
#         out = rearrange(Y, 'b h v (hh ww) -> b (h v) hh ww', hh=hh, ww=ww)
#         return out
#
#
# class Space_attention_v2(torch.nn.Module):
#     def __init__(self, input_size, output_size, kernel_size, stride, padding, scale):
#         super(Space_attention_v2, self).__init__()
#
#         self.input_size = input_size
#         self.output_size = output_size
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.scale = scale
#         # downscale = scale + 4
#
#         self.K = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
#         self.Q = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
#         self.V = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
#         self.pool = nn.MaxPool2d(kernel_size=self.scale + 2, stride=self.scale, padding=1)
#         # self.bn = nn.BatchNorm2d(output_size)
#         if kernel_size == 1:
#             self.local_weight = torch.nn.Conv2d(output_size, input_size, kernel_size, stride, padding, bias=True)
#         else:
#             self.local_weight = torch.nn.ConvTranspose2d(output_size, input_size, kernel_size, stride, padding,
#                                                          bias=True)
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         K = self.K(x)
#         Q = self.Q(x)
#         # Q = F.interpolate(Q, scale_factor=1 / self.scale, mode='bicubic')
#         if self.stride > 1:
#             Q = self.pool(Q)
#         else:
#             Q = Q
#         V = self.V(x)
#         # V = F.interpolate(V, scale_factor=1 / self.scale, mode='bicubic')
#         if self.stride > 1:
#             V = self.pool(V)
#         else:
#             V = V
#         V_reshape = V.view(batch_size, self.output_size, -1)
#         V_reshape = V_reshape.permute(0, 2, 1)
#         # if self.type == 'softmax':
#         Q_reshape = Q.view(batch_size, self.output_size, -1)
#
#         K_reshape = K.view(batch_size, self.output_size, -1)
#         K_reshape = K_reshape.permute(0, 2, 1)
#
#         QV = torch.matmul(Q_reshape, V_reshape)
#         attention = F.softmax(QV, dim=-1)
#
#         vector = torch.matmul(K_reshape, attention)
#         vector_reshape = vector.permute(0, 2, 1).contiguous()
#         O = vector_reshape.view(batch_size, self.output_size, x.size(2) // self.stride, x.size(3) // self.stride)
#         W = self.local_weight(O)
#         output = x + W
#         # output = self.bn(output)
#         return output


class GCT(nn.Module):

    def __init__(self, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()
        self.channel = settings.channel
        self.alpha = nn.Parameter(torch.ones(1, self.channel, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, self.channel, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, self.channel, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


class eca_layer_avg(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=5):
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

        return x * y.expand_as(x)


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

        return x * y.expand_as(x)


class Residual_Block(nn.Module):
    def __init__(self):
        super(Residual_Block, self).__init__()
        self.channel_num = settings.channel
        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        if settings.eca is True:
            self.avg_eca = GCT()
            # self.max_eca = eca_layer_max()
            # self.conv11 = nn.Conv2d(2 * self.channel_num, self.channel_num, 1, 1)
        if settings.lamda is True:
            self.lamda = LambdaLayer(dim=self.channel_num, dim_out=self.channel_num, r=23, dim_k=16, heads=4, dim_u=4)
        self.res = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
        )

    def forward(self, x):
        identity = x
        if settings.lamda is True:
            x = self.lamda(x)
        if settings.eca is True:
            x = self.avg_eca(x)
            # eca_max = self.max_eca(x)
            # x = self.conv11(torch.cat([eca_avg, eca_max], dim=1))
            # x = eca_avg #+ eca_max
        out = identity + self.res(x)
        return out


class DenseConnection(nn.Module):
    def __init__(self, unit_num):
        super(DenseConnection, self).__init__()
        self.unit_num = unit_num
        self.channel = settings.channel
        self.units = nn.ModuleList()
        self.conv1x1 = nn.ModuleList()
        self.conv_fusion = nn.Sequential(nn.Conv2d(self.unit_num * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2))
        for i in range(self.unit_num):
            self.units.append(Residual_Block())
            self.conv1x1.append(nn.Sequential(nn.Conv2d((i + 2) * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2)))

    def forward(self, x, input_fusion=None):
        # print(x)
        cat = []
        cat.append(x)
        out = x
        output_fusion = []
        for i in range(self.unit_num):
            if input_fusion is None:
                tmp = self.units[i](out)
            else:
                tmp = self.units[i](out + input_fusion)
            cat.append(tmp)
            output_fusion.append(tmp)
            out = self.conv1x1[i](torch.cat(cat, dim=1))
        fusion = self.conv_fusion(torch.cat(output_fusion, dim=1))
        return tmp, fusion


class Resbase(nn.Module):
    def __init__(self):
        super(Resbase, self).__init__()
        self.channel_num = settings.channel
        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.res = nn.Sequential(
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.channel_num, self.channel_num, 3, 1, 1),
        )

    def forward(self, x):
        out = x + self.res(x)
        return out


class Dense(nn.Module):
    def __init__(self, unit_num):
        super(Dense, self).__init__()
        self.unit_num = unit_num
        self.channel = settings.channel
        self.units = nn.ModuleList()
        self.conv1x1 = nn.ModuleList()
        for i in range(self.unit_num):
            self.units.append(Resbase())
            self.conv1x1.append(nn.Sequential(nn.Conv2d((i + 2) * self.channel, self.channel, 1, 1), nn.LeakyReLU(0.2)))

    def forward(self, x):
        cat = []
        cat.append(x)
        out = x
        for i in range(self.unit_num):
            tmp = self.units[i](out)
            cat.append(tmp)
            out = self.conv1x1[i](torch.cat(cat, dim=1))
        return out


class ODE_DerainNet(nn.Module):
    def __init__(self):
        super(ODE_DerainNet, self).__init__()
        self.out_planes = settings.channel
        self.num_res = settings.unit_num
        self.num_group_res = settings.num_group_res
        self.WITH_DIFF = settings.WITH_DIFF
        self.res_group = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.training = settings.training
        self.with_window = settings.WITH_WINDOW
        self.WITH_DIFF = settings.WITH_DIFF
        self.with_score = settings.WITH_SCORE
        self.n_neighbors = settings.n_neighbors
        self.patch_size = settings.graph_patch_size
        self.gcn_stride = settings.gcn_stride
        self.window = settings.window

        self.convert_rainy = nn.Conv2d(3, self.out_planes, 3, 1, 1)
        if settings.graph is True:
            self.convert_rainy1 = nn.Sequential(nn.Conv2d(3, self.out_planes, 3, 1, 1), nn.LeakyReLU(0.2),
                                                Dense(4))
            self.convert_rainy2 = nn.Sequential(nn.Conv2d(3, self.out_planes, 3, 1, 1), nn.LeakyReLU(0.2),
                                                Dense(4))
            self.convert_rainy4 = nn.Sequential(nn.Conv2d(3, self.out_planes, 3, 1, 1), nn.LeakyReLU(0.2),
                                                Dense(4))
            self.convert_guide = nn.Sequential(nn.Conv2d(3, self.out_planes, 3, 1, 1), nn.LeakyReLU(0.2),
                                               Dense(4))
        for i in range(self.num_group_res):
            self.res_group.append(DenseConnection(self.num_res))
            if settings.graph is True:
                if settings.guide is True:
                    self.fuse.append(nn.Sequential(
                        nn.Conv2d(5 * self.out_planes, self.out_planes, 1, 1), nn.LeakyReLU(0.2),
                        nn.Conv2d(self.out_planes, self.out_planes, 1, 1)
                    ))
                else:
                    self.fuse.append(nn.Sequential(
                        nn.Conv2d(4 * self.out_planes, self.out_planes, 5, 1, 2), nn.LeakyReLU(0.2),
                        nn.Conv2d(self.out_planes, self.out_planes, 5, 1, 2)
                    ))
        if settings.graph is True:
            self.graph_1 = Graph(scale=1, k=self.n_neighbors, patchsize=self.patch_size, stride=self.gcn_stride,
                                 window_size=self.window, in_channels=self.out_planes, embed_ch=self.out_planes,
                                 embed_out=self.out_planes, WITH_DIFF=self.WITH_DIFF, WITH_SCORE=self.with_score,
                                 embedcnn=None)
            self.gcn_1 = GCNBlock(self.out_planes, scale=1, k=self.n_neighbors, patchsize=self.patch_size,
                                  stride=self.gcn_stride, diff_n=self.out_planes)

            self.graph_2 = Graph(scale=2, k=self.n_neighbors, patchsize=self.patch_size, stride=self.gcn_stride,
                                 window_size=self.window, in_channels=self.out_planes, embed_ch=self.out_planes,
                                 embed_out=self.out_planes, WITH_DIFF=self.WITH_DIFF, WITH_SCORE=self.with_score,
                                 embedcnn=None)
            self.gcn_2 = GCNBlock(self.out_planes, scale=2, k=self.n_neighbors, patchsize=self.patch_size,
                                  stride=self.gcn_stride, diff_n=self.out_planes)

            self.graph_4 = Graph(scale=4, k=self.n_neighbors, patchsize=self.patch_size, stride=self.gcn_stride,
                                 window_size=self.window, in_channels=self.out_planes, embed_ch=self.out_planes,
                                 embed_out=self.out_planes, WITH_DIFF=self.WITH_DIFF, WITH_SCORE=self.with_score,
                                 embedcnn=None)
            self.gcn_4 = GCNBlock(self.out_planes, scale=4, k=self.n_neighbors, patchsize=self.patch_size,
                                  stride=self.gcn_stride, diff_n=self.out_planes)
        if settings.graph is True:
            if settings.guide is True:
                self.graph_guide = Graph(scale=1, k=self.n_neighbors, patchsize=self.patch_size, stride=self.gcn_stride,
                                         window_size=self.window, in_channels=self.out_planes, embed_ch=self.out_planes,
                                         embed_out=self.out_planes, embedcnn=None)
                self.gcn_guide = GCNBlock(self.out_planes, scale=1, k=self.n_neighbors, patchsize=self.patch_size,
                                          stride=self.gcn_stride, diff_n=self.out_planes)

        self.out = nn.Sequential(
            nn.Conv2d(self.out_planes, self.out_planes, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(self.out_planes, 3, 1, 1)
        )

    def forward(self, rainy, rainy_2, rainy_4, guide):
        # print(rainy)
        if settings.graph is True:
            # print('rainy',rainy.size())
            score_k_1, idx_k_1, diff_patch_1 = self.graph_1(rainy, rainy)
            score_k_2, idx_k_2, diff_patch_2 = self.graph_2(rainy_2, rainy)
            score_k_4, idx_k_4, diff_patch_4 = self.graph_4(rainy_4, rainy)
            if settings.guide is True:
                score_k_guide, idx_k_guide, diff_patch_guide = self.graph_guide(guide, rainy)
            # print(diff_patch_1)
            # print(diff_patch_2)
            # print(diff_patch_4.size())

        temp = self.convert_rainy(rainy)
        # print('temp', temp.size())
        if settings.graph is True:
            rainy1 = self.convert_rainy1(rainy)
            rainy2 = self.convert_rainy2(rainy_2)
            rainy4 = self.convert_rainy4(rainy_4)
            if settings.guide is True:
                guide = self.convert_guide(guide)

            if self.WITH_DIFF:
                diff_patch_1 = diff_patch_1.detach()
                diff_patch_2 = diff_patch_2.detach()
                diff_patch_4 = diff_patch_4.detach()
                if settings.guide is True:
                    diff_patch_guide = diff_patch_guide.detach()
        for i in range(self.num_group_res):
            # print('self.num_group_res', self.num_group_res)
            # print('self.num_res', self.num_res)
            if i == 0:
                body1, output_fusion = self.res_group[i](temp, input_fusion=None)
                # print(body1)
                # print(output_fusion)

            else:
                # print('temp', temp.size())
                # print('output_fusion', output_fusion.size())
                body1, output_fusion = self.res_group[i](temp, input_fusion=output_fusion)
                # print('body1', body1.size())
            if settings.graph is True:
                x1 = self.gcn_1(rainy1, body1, idx_k_1, diff_patch_1)
                x2 = self.gcn_2(rainy2, body1, idx_k_2, diff_patch_2)
                x4 = self.gcn_4(rainy4, body1, idx_k_4, diff_patch_4)
                if settings.guide is True:
                    x_guide = self.gcn_guide(guide, body1, idx_k_guide, diff_patch_guide)
                    fuse = self.fuse[i](torch.cat([body1, x1, x2, x4, x_guide], dim=1))
                else:
                    fuse = self.fuse[i](torch.cat([body1, x1, x2, x4], dim=1))
                temp = fuse
            else:
                temp = body1
        rain = self.out(body1)

        estimated_clean = rainy - rain

        return estimated_clean
# class ODE_DerainNet(nn.Module):
#     def __init__(self):
#         super(ODE_DerainNet, self).__init__()
#         self.channel = settings.channel
#         self.unit_num = settings.unit_num
#         self.enterBlock = nn.Sequential(nn.Conv2d(3, self.channel, 3, 1, 1), nn.LeakyReLU(0.2))
#         self.derain_net = DenseConnection(Residual_Block, self.unit_num)
#         self.exitBlock = nn.Sequential(nn.Conv2d(self.channel, 3, 3, 1, 1), nn.LeakyReLU(0.2))
#
#
#     def forward(self, x):
#         image_feature = self.enterBlock(x)
#         rain_feature = self.derain_net(image_feature)
#         rain = self.exitBlock(rain_feature)
#         derain = x - rain
#         return derain
