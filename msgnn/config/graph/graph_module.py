#!/usr/bin/python
# -*- coding: utf-8 -*-
#
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
# import utils.network_utils as net_utils
from .graph_agg import *

# from config import cfg

act = nn.LeakyReLU(0.2)

####################################################################
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .lib import network_utils as net_utils
from .lib import ops

WITH_WINDOW = True
#  WINDOW_SIZE:                 = 30
WITH_ADAIN_NROM = True
WITH_DIFF = True
WITH_SCORE = False


def compute_distances(xe, ye, I, train=True):
    r"""
    Computes pairwise distances for all pairs of query items and
    potential neighbors.

    :param xe: BxNxE tensor of database (son) item embeddings
    :param ye: BxMxE tensor of query (father) item embeddings
    :param I: BxMxO index tensor that selects O potential neighbors in a window for each item in ye
    :param train: whether to use tensor comprehensions for inference (forward only)

    :return D: a BxMxO tensor of distances
    """

    # xe -> b n e
    # ye -> b m e
    # I  -> b m o
    b, n, e = xe.shape
    m = ye.shape[1]

    if train or not WITH_WINDOW:
        # D_full -> b m n
        D = ops.euclidean_distance(ye, xe.permute(0, 2, 1))
        if WITH_WINDOW:
            # D -> b m o
            D = D.gather(dim=2, index=I) + 1e-5
    else:
        o = I.shape[2]
        # xe_ind -> b m o e
        If = I.view(b, m * o, 1).expand(b, m * o, e)
        # D -> b m o
        ye = ye.unsqueeze(3)
        D = -2 * ops.indexed_matmul_1_efficient(xe, ye.squeeze(3), I).unsqueeze(3)

        xe_sqs = (xe ** 2).sum(dim=-1, keepdim=True)
        xe_sqs_ind = xe_sqs.gather(dim=1, index=If[:, :, 0:1]).view(b, m, o, 1)
        D += xe_sqs_ind
        D += (ye ** 2).sum(dim=-2, keepdim=True) + 1e-5

        D = D.squeeze(3)

    return D


def hard_knn(D, k, I):
    r"""
    input D: b m n
    output Idx: b m k
    """
    score, idx = torch.topk(D, k, dim=2, largest=False, sorted=True)
    if WITH_WINDOW:
        idx = I.gather(dim=2, index=idx)

    return score, idx


############################################################################
# out_shape = (H-1)//stride + 1 # for dilation=1
def conv(in_channels, out_channels, kernel_size=3, act=True, stride=1, groups=1, bias=True):
    m = []
    m.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                       padding=(kernel_size - 1) // 2, groups=groups, bias=bias))
    if act: m.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*m)


# out_shape = H*stride + kernel - 2*padding - stride + out_padding # for dilation=1
def upconv(in_channels, out_channels, stride=2, act=True, groups=1, bias=True):
    m = []
    kernel_size = 2 + stride
    m.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                padding=1, groups=groups, bias=bias))
    if act: m.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*m)


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3, res_scale=1, bias=True):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size - 1) // 2, bias=bias))
            if i == 0:
                m.append(nn.LeakyReLU(0.2, inplace=True))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range=1., rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1., 1., 1.), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


def upsampler(in_channels, kernel_size=3, scale=2, act=False, bias=True):
    m = []
    if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
        for _ in range(int(math.log(scale, 2))):
            m.append(nn.Conv2d(in_channels, 4 * in_channels, kernel_size, padding=1, bias=bias))
            m.append(nn.PixelShuffle(2))
            if act: m.append(nn.LeakyReLU(0.2, inplace=True))

    elif scale == 3:
        m.append(nn.Conv2d(in_channels, 9 * in_channels, kernel_size, padding=1, bias=bias))
        m.append(nn.PixelShuffle(3))
        if act: m.append(nn.LeakyReLU(0.2, inplace=True))
    else:
        raise NotImplementedError

    return nn.Sequential(*m)


class PixelShuffle_Down(nn.Module):
    def __init__(self, scale=2):
        super(PixelShuffle_Down, self).__init__()
        self.scale = scale

    def forward(self, x):
        # assert h%scale==0 and w%scale==0
        b, c, h, w = x.size()
        x = x[:, :, :int(h - h % self.scale), :int(w - w % self.scale)]
        out_c = c * (self.scale ** 2)
        out_h = h // self.scale
        out_w = w // self.scale
        out = x.contiguous().view(b, c, out_h, self.scale, out_w, self.scale)
        return out.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_c, out_h, out_w)


class GraphConstruct(nn.Module):
    r"""
    Graph Construction
    """

    def __init__(self, scale, indexer, k, patchsize, stride, padding=None, with_diff=True, with_score=False,
                 training=True):
        r"""
        :param scale: downsampling factor
        :param indexer: function for creating index tensor
        :param k: number of neighbors
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        """
        super(GraphConstruct, self).__init__()
        self.scale = scale
        self.patchsize = patchsize
        self.stride = stride
        self.indexer = indexer
        self.k = k
        self.padding = padding
        self.with_diff = with_diff
        self.with_score = with_score
        self.training = training

    def graph_k(self, xe, ye, I):
        # xe -> b n e
        # ye -> b m e
        # I  -> b m o
        n = xe.shape[1]
        b, m, e = ye.shape
        k = self.k

        # Euclidean Distance
        D = compute_distances(xe, ye, I, train=self.training)

        # hard knn
        # return: b m k
        score_k, idx_k = hard_knn(D, k, I)

        # xe -> b m e n
        # idx-> b m e k
        xe = xe.permute(0, 2, 1).contiguous()
        xe_e = xe.view(b, 1, e, n).expand(b, m, e, n)
        idx_k_e = idx_k.view(b, m, 1, k).expand(b, m, e, k)

        if self.with_diff:
            ye_e = ye.view(b, m, e, 1).expand(b, m, e, k)
            diff_patch = ye_e - torch.gather(xe_e, dim=3, index=idx_k_e)
        else:
            diff_patch = None

        if self.with_score:
            score_k = (-score_k / 10.).exp()
        else:
            score_k = None

        # score_k: b m k
        # idx_k: b m k
        # diff_patch: b m e k
        return score_k, idx_k, diff_patch

    def forward(self, xe, ye):
        r"""
        :param xe: embedding of son features
        :param ye: embedding of father features

        :return score_k: similarity scores of top k nearest neighbors
        :return idx_k: indexs of top k nearest neighbors
        :return diff_patch: difference vectors between query and k nearest neighbors
        """
        # Convert everything to patches
        # print('xe',xe.size())
        # print('ye',ye.size())
        H, W = ye.shape[2:]
        xe_patch = ops.im2patch(xe, self.patchsize, self.stride, self.padding)
        ye_patch, padding = ops.im2patch(ye, self.patchsize, self.stride, self.padding, returnpadding=True)
        # print('xe_patch', xe_patch.size())
        # print('ye_patch', ye_patch.size())

        I = self.indexer(xe_patch, ye_patch)
        # print('I',I.size())

        if not self.training:
            index_neighbours_cache.clear()

        # bacth, channel, patchsize1, patchsize2, h, w
        _, _, _, _, n1, n2 = xe_patch.shape
        b, ce, e1, e2, m1, m2 = ye_patch.shape

        k = self.k
        n = n1 * n2;
        m = m1 * m2;
        e = ce * e1 * e2
        xe_patch = xe_patch.permute(0, 4, 5, 1, 2, 3).contiguous().view(b, n, e)
        ye_patch = ye_patch.permute(0, 4, 5, 1, 2, 3).contiguous().view(b, m, e)

        # Get nearest neighbor volumes
        score_k, idx_k, diff_patch = self.graph_k(xe_patch, ye_patch, I)
        # print('idx_k---', idx_k.size())
        # print('diff_patch---', diff_patch.size())

        if self.with_diff:
            # diff_patch -> b,m,e,k      b m1*m2 ce e1*e2 k
            diff_patch = abs(diff_patch.view(b, m, ce, e1 * e2, k))
            diff_patch = torch.sum(diff_patch, dim=3, keepdim=True)
            diff_patch = diff_patch.expand(b, m, ce, e1 * self.scale * e2 * self.scale, k)

            # diff_patch: b m ce e1*s*e2*s k; e1==p1, e2==p2;
            # diff_patch -> b k ce e1*s*e2*s m
            diff_patch = diff_patch.permute(0, 4, 2, 3, 1).contiguous()
            # diff_patch -> b k*ce e1*s e2*s m1 m2
            diff_patch = diff_patch.view(b, k * ce, e1 * self.scale, e2 * self.scale, m1, m2)
            padding_sr = [p * self.scale for p in padding]
            # z_sr -> b k*c_y H*s W*s
            diff_patch = ops.patch2im(diff_patch, self.patchsize * self.scale, self.stride * self.scale, padding_sr)
            diff_patch = diff_patch.contiguous().view(b, k * ce, H * self.scale, W * self.scale)

        if self.with_score:
            # score_k: b,m,k --> b,k,e1*s,e2*s,m1,m2
            score_k = score_k.permute(0, 2, 1).contiguous().view(b, k, 1, 1, m1, m2)
            score_k = score_k.view(b, k, 1, 1, m1, m2).expand(b, k, e1 * self.scale, e2 * self.scale, m1, m2)
            padding_sr = [p * self.scale for p in padding]
            score_k = ops.patch2im(score_k, self.patchsize * self.scale, self.stride * self.scale, padding_sr)
            score_k = score_k.contiguous().view(b, k, H * self.scale, W * self.scale)

        # print('scale', self.scale)
        # score_k: b k H*s W*s
        # idx_k: b m k
        # diff_patch: b k*ce H*s W*s
        # print('idx_k------', idx_k.size())
        # print('diff_patch------', diff_patch.size())
        # idx_k torch.Size([1, 400, 5])
        # diff_patch torch.Size([1, 400, 576, 5])
        return score_k, idx_k, diff_patch


class GraphAggregation(nn.Module):
    r"""
    Graph Aggregation
    """

    def __init__(self, scale, k, patchsize, stride, padding=None):
        r"""
        :param k: number of neighbors
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        """
        super(GraphAggregation, self).__init__()
        self.scale = scale
        self.patchsize = patchsize
        self.stride = stride
        self.k = k
        self.padding = padding

    def aggregation(self, yd, idx_k):
        r"""
        :param yd: database items, shape BxNxF
        :param idx_k: indexs of top k nearest neighbors

        :return: gathered features
        """
        # yd  -> b n f
        # I  -> b m o
        m = idx_k.shape[1]
        b, n, f = yd.shape
        k = self.k

        # yd -> b m f n
        # idx-> b m f k
        yd = yd.permute(0, 2, 1).contiguous()
        yd_e = yd.view(b, 1, f, n).expand(b, m, f, n)
        idx_k_e = idx_k.view(b, m, 1, k).expand(b, m, f, k)
        z = torch.gather(yd_e, dim=3, index=idx_k_e)

        # b m1*m2 c*p1*p2 k
        return z

    def forward(self, y, yd, idx_k):
        r"""
        :param y: query lr features
        :param yd: pixelshuffle_down features of y
        :param idx_k: indexs of top k nearest neighbors

        :return: aggregated hr features
        """
        # Convert everything to patches
        y_patch, padding = ops.im2patch(y, self.patchsize, self.stride, self.padding, returnpadding=True)
        yd_patch = ops.im2patch(yd, self.patchsize, self.stride, self.padding)

        # bacth, channel, patchsize1, patchsize2, h, w
        _, _, H, W = y.shape
        _, _, _, _, m1, m2 = y_patch.shape
        b, c, p1, p2, n1, n2 = yd_patch.shape

        m = m1 * m2;
        n = n1 * n2;
        f = c * p1 * p2
        k = self.k

        y_patch = y_patch.permute(0, 4, 5, 1, 2, 3).contiguous().view(b, m, f // self.scale ** 2)
        yd_patch = yd_patch.permute(0, 4, 5, 1, 2, 3).contiguous().view(b, n, f)

        # Get nearest neighbor volumes
        # z_patch -> b m1*m2 c*p1*p2 k
        z_patch = self.aggregation(yd_patch, idx_k)

        # Adaptive_instance_normalization
        if WITH_ADAIN_NROM:
            reduce_scale = self.scale ** 2
            y_patch_norm = y_patch.view(b, m, c // reduce_scale, p1 * p2)
            z_patch_norm = z_patch.view(b, m, c // reduce_scale, reduce_scale * p1 * p2, k)
            z_patch = net_utils.adaptive_instance_normalization(y_patch_norm, z_patch_norm).view(*z_patch.size())

        # z_patch -> b k*c p1 p2 m1 m2
        z_patch = z_patch.permute(0, 3, 2, 1).contiguous()

        z_patch_sr = z_patch.view(b, k, c // self.scale ** 2, self.scale, self.scale, p1, p2, m1, m2).permute(0, 1, 2,
                                                                                                              5, 3, 6,
                                                                                                              4, 7,
                                                                                                              8).contiguous()
        z_patch_sr = z_patch_sr.view(b, k * (c // self.scale ** 2), p1 * self.scale, p2 * self.scale, m1, m2)
        padding_sr = [p * self.scale for p in padding]
        # z_sr -> b k*c_y H*s W*s
        z_sr = ops.patch2im(z_patch_sr, self.patchsize * self.scale, self.stride * self.scale, padding_sr)
        z_sr = z_sr.contiguous().view(b, k * (c // self.scale ** 2), H * self.scale, W * self.scale)

        return z_sr


index_neighbours_cache = {}


def index_neighbours(xe_patch, ye_patch, window_size, scale):
    r"""
    This function generates the indexing tensors that define neighborhoods for each query patch in (father) features
    It selects a neighborhood of window_size x window_size patches around each patch in xe (son) features
    Index tensors get cached in order to speed up execution time
    """
    # if cfg.NETWORK.WITH_WINDOW == False:
    #     return None
    # dev = xe_patch.get_device()
    # key = "{}_{}_{}_{}_{}_{}".format(n1,n2,m1,m2,s,dev)
    # if not key in index_neighbours_cache:
    #     I = torch.tensor(range(n), device=dev, dtype=torch.int64).view(1,1,n)
    #     I = I.repeat(b, m, 1)
    #     index_neighbours_cache[key] = I

    # I = index_neighbours_cache[key]
    # return Variable(I, requires_grad=False)

    b, _, _, _, n1, n2 = xe_patch.shape
    s = window_size
    # print('s', s)
    # print('n1',n1)
    # print('n2',n2)
    #
    # if s>=n1 and s>=n2:
    #     cfg.NETWORK.WITH_WINDOW = False
    #     return None

    s = min(min(s, n1), n2)
    o = s ** 2
    b, _, _, _, m1, m2 = ye_patch.shape

    dev = xe_patch.get_device()
    key = "{}_{}_{}_{}_{}_{}".format(n1, n2, m1, m2, s, dev)
    if not key in index_neighbours_cache:
        I = torch.empty(1, m1 * m2, o, device=dev, dtype=torch.int64)

        ih = torch.tensor(range(s), device=dev, dtype=torch.int64).view(1, 1, s, 1)
        iw = torch.tensor(range(s), device=dev, dtype=torch.int64).view(1, 1, 1, s) * n2

        i = torch.tensor(range(m1), device=dev, dtype=torch.int64).view(m1, 1, 1, 1)
        j = torch.tensor(range(m2), device=dev, dtype=torch.int64).view(1, m2, 1, 1)

        i_s = (torch.tensor(range(m1), device=dev, dtype=torch.int64).view(m1, 1, 1, 1) // 2.0).long()
        j_s = (torch.tensor(range(m2), device=dev, dtype=torch.int64).view(1, m2, 1, 1) // 2.0).long()

        ch = (i_s - s // scale).clamp(0, n1 - s)
        cw = (j_s - s // scale).clamp(0, n2 - s)

        cidx = ch * n2 + cw
        mI = cidx + ih + iw
        mI = mI.view(m1 * m2, -1)
        I[0, :, :] = mI

        index_neighbours_cache[key] = I

    I = index_neighbours_cache[key]
    I = I.repeat(b, 1, 1)

    return Variable(I, requires_grad=False)


# ----------GCNBlock---------- #
class Graph(nn.Module):
    r"""
    Graph Construction
    """

    def __init__(self, scale, k=5, patchsize=3, stride=1, window_size=7, in_channels=16, embed_ch=32, embed_out=32,
                 WITH_DIFF=True, WITH_SCORE=False, embedcnn=None, training=True):
        r"""
        :param k: number of neighbors to sample
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        :param window_size: size of matching window around each patch,
            i.e. the window_size x window_size patches around a query patch
            are used for matching
        :param in_channels: number of input channels
        :param embedcnn_opt: options for the embedding cnn
        """
        super(Graph, self).__init__()
        self.scale = scale
        self.k = k
        self.vgg = embedcnn is not None
        self.with_diff = WITH_DIFF
        self.with_score = WITH_SCORE

        if embedcnn is None:
            embed_ch = 64
            embed_out = 32
            self.embedcnn = nn.Sequential(
                conv(3, embed_ch, kernel_size=3),
                conv(embed_ch, embed_ch, kernel_size=3),
                conv(embed_ch, embed_out, kernel_size=3)
            )
        else:
            # self.embedcnn = embedcnn
            self.embedcnn = nn.Sequential(
                conv(3, embed_ch, kernel_size=3),
                conv(embed_ch, embed_ch, kernel_size=3),
                conv(embed_ch, embed_out, kernel_size=3))

        indexer = lambda xe_patch, ye_patch: index_neighbours(xe_patch, ye_patch, window_size, scale)

        self.graph_construct = GraphConstruct(scale=scale, indexer=indexer, k=k, patchsize=patchsize,
                                              stride=stride, with_diff=self.with_diff, with_score=self.with_score,
                                              training=training)

    def forward(self, x, y):
        # x: son features, y: father features

        xe = self.embedcnn(x)
        ye = self.embedcnn(y)
        # xe torch.Size([2, 64, 20, 20])
        # ye torch.Size([2, 64, 40, 40])

        score_k, idx_k, diff_patch = self.graph_construct(xe, ye)
        # idx_k torch.Size([1, 400, 5])
        # diff_patch torch.Size([1, 320, 80, 80])
        return score_k, idx_k, diff_patch


class GCNBlock(nn.Module):
    r"""
    Graph Aggregation
    """

    def __init__(self, nplanes_in, scale, k=5, patchsize=3, stride=1, diff_n=32):
        r"""
        :param nplanes_in: number of input features
        :param scale: downsampling factor
        :param k: number of neighbors to sample
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        :param diff_n: number of diff vector channels
        """
        super(GCNBlock, self).__init__()
        self.nplanes_in = nplanes_in
        self.scale = scale
        self.k = k
        self.diff_n = diff_n

        self.pixelshuffle_down = PixelShuffle_Down(scale)

        self.graph_aggregate = GraphAggregation(scale=scale, k=k, patchsize=patchsize, stride=stride)

        self.knn_downsample = nn.Sequential(
            conv(nplanes_in, nplanes_in, kernel_size=5, stride=scale),
            conv(nplanes_in, nplanes_in, kernel_size=3),
            conv(nplanes_in, nplanes_in, kernel_size=3, act=False)
        )

        self.diff_downsample = nn.AvgPool2d(kernel_size=scale, stride=scale)

        self.weightnet_lr = nn.Sequential(
            conv(self.diff_n, self.diff_n, kernel_size=1, act=False),
            ResBlock(self.diff_n, kernel_size=1, res_scale=1),
            conv(self.diff_n, 1, kernel_size=1, act=False)
        )
        self.weightnet_hr = nn.Sequential(
            conv(self.diff_n, self.diff_n, kernel_size=1, act=False),
            ResBlock(self.diff_n, kernel_size=1, res_scale=1),
            conv(self.diff_n, 1, kernel_size=1, act=False)
        )

    def weight_edge(self, knn_hr, diff_patch):
        b, c, h_hr, w_hr = knn_hr.shape
        b, ce, _, _ = diff_patch.shape
        h_lr, w_lr = h_hr // self.scale, w_hr // self.scale
        # print('scale', self.scale)
        # print('k', self.k)

        knn_hr = knn_hr.view(b, self.k, c // self.k, h_hr, w_hr)
        diff_patch = diff_patch.view(b, self.k, ce // self.k, h_hr, w_hr)

        knn_lr, weight_lr, weight_hr = [], [], []
        for i in range(self.k):
            knn_lr.append(self.knn_downsample(knn_hr[:, i]).view(b, 1, c // self.k, h_lr, w_lr))
            diff_patch_lr = self.diff_downsample(diff_patch[:, i])
            weight_lr.append(self.weightnet_lr(diff_patch_lr))
            weight_hr.append(self.weightnet_hr(diff_patch[:, i]))

        weight_lr = torch.cat(weight_lr, dim=1)
        weight_lr = weight_lr.view(b, self.k, 1, h_lr, w_lr)
        weight_lr = F.softmax(weight_lr, dim=1)

        weight_hr = torch.cat(weight_hr, dim=1)
        weight_hr = weight_hr.view(b, self.k, 1, h_hr, w_hr)
        weight_hr = F.softmax(weight_hr, dim=1)

        knn_lr = torch.cat(knn_lr, dim=1)
        knn_lr = torch.sum(knn_lr * weight_lr, dim=1, keepdim=False)
        knn_hr = torch.sum(knn_hr * weight_hr, dim=1, keepdim=False)

        return knn_lr, knn_hr

    def forward(self, y, idx_k, diff_patch):
        # print('self.diff_n', self.diff_n)

        # graph_aggregate
        # if guide is True:
        #     yd = self.pixelshuffle_down(x)
        # else:
        yd = self.pixelshuffle_down(y)
        # print('yd', yd.size())

        # b k*c h*s w*s
        # print('y',y.size())
        # print('yd', yd.size())
        # print('idx_k',idx_k.size())
        # print('diff_patch', diff_patch.size())
        # y torch.Size([1, 256, 40, 40])
        # x torch.Size([1, 256, 20, 20])
        # idx_k torch.Size([1, 400, 5])
        # diff_patch torch.Size([1, 320, 80, 80])

        knn_hr = self.graph_aggregate(y, yd, idx_k)
        # knn_hr torch.Size([1, 1280, 80, 80])

        # for diff socre
        knn_lr, knn_hr = self.weight_edge(knn_hr, diff_patch)
        # print('knn_lr----', knn_lr.size())
        # print('knn_hr----', knn_hr.size())

        return knn_lr, knn_hr