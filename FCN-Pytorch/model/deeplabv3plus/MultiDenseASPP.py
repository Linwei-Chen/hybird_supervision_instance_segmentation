import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.deeplabv3plus.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from model.deeplabv3plus.backbone import build_backbone
from model.deeplabv3plus.ASPP import ASPP
from model.deeplabv3plus.BalancedDenseASPP import _DenseASPPHead, _PrimeDenseASPPHead, _DenseASPPWithoutGlobalHead
import torch.utils.checkpoint as cp


class DeepLabv3PlusMultiDenseASPP(nn.Module):
    def __init__(self, cfg, prime=False, ca=False):
        """
        :param cfg: configuration
        :param prime: use prime denseblock or not
        :param ca: use channel attention or not
        """
        super().__init__()
        self.backbone = None
        self.backbone_layers = None
        cfg.MODEL_BACKBONE = 'xception3strides'
        input_channels = (256, 728, 2048)
        out_channels = (64, 128, 256)
        DenseASPP = _PrimeDenseASPPHead if prime else _DenseASPPHead
        self.denseaspp1 = DenseASPP(in_channels=input_channels[0],
                                    dense_squeeze_channels=64,
                                    dense_out_channels=16,
                                    out_channels=out_channels[0],
                                    rate=1,
                                    norm_layer=SynchronizedBatchNorm2d,
                                    norm_kwargs={'momentum': cfg.TRAIN_BN_MOM}, ca=ca)
        self.denseaspp2 = DenseASPP(in_channels=input_channels[1],
                                    dense_squeeze_channels=128,
                                    dense_out_channels=32,
                                    out_channels=out_channels[1],
                                    rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                                    norm_layer=SynchronizedBatchNorm2d,
                                    norm_kwargs={'momentum': cfg.TRAIN_BN_MOM}, ca=ca)
        self.denseaspp3 = DenseASPP(in_channels=input_channels[2],
                                    dense_squeeze_channels=256,
                                    dense_out_channels=64,
                                    out_channels=out_channels[2],
                                    rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                                    norm_layer=SynchronizedBatchNorm2d,
                                    norm_kwargs={'momentum': cfg.TRAIN_BN_MOM}, ca=ca)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

        self.upsample_all = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE // 8)
        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE // 4)

        indim = 256
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1,
                      padding=cfg.MODEL_SHORTCUT_KERNEL // 2, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
        )
        # self.channel_attention = ca
        #
        # if ca:
        #     self.ca = GAU(channels_high=cfg.MODEL_ASPP_OUTDIM, channels_low=cfg.MODEL_SHORTCUT_DIM,
        #                   norm_layer=SynchronizedBatchNorm2d, norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})

        self.cat_conv = nn.Sequential(
            nn.Conv2d(sum(out_channels), cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,
                      bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()

    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        # for l in layers:
        #     print(l.shape)

        dense_aspp3 = self.denseaspp3(layers[-1])
        dense_aspp3 = self.dropout3(dense_aspp3)
        dense_aspp3 = self.upsample3(dense_aspp3)

        dense_aspp2 = self.denseaspp2(layers[-2])
        dense_aspp2 = self.dropout2(dense_aspp2)
        dense_aspp2 = self.upsample2(dense_aspp2)

        dense_aspp1 = self.denseaspp1(layers[-3])
        dense_aspp1 = self.dropout1(dense_aspp1)

        # print(dense_aspp1.shape, dense_aspp2.shape, dense_aspp3.shape)

        feature_cat = torch.cat([dense_aspp3, dense_aspp2, dense_aspp1], dim=1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = self.upsample_all(result)
        return result


class DeepLabv3PlusMultiDenseASPPv2(nn.Module):
    def __init__(self, cfg, prime=False, ca=False, me=True, dense_out=None):
        """
        :param cfg: configuration
        :param prime: use prime denseblock or not
        :param ca: use channel attention or not
        """
        super().__init__()
        self.backbone = None
        self.backbone_layers = None
        cfg.MODEL_BACKBONE = 'xception3strides'
        input_channels = (256, 728, 2048)
        if dense_out is None:
            dense_out = (64, 128, 256)

        DenseASPP = _PrimeDenseASPPHead if prime else _DenseASPPHead
        DenseASPPWithoutGlobal = _DenseASPPWithoutGlobalHead
        self.denseaspp1 = DenseASPPWithoutGlobal(in_channels=input_channels[0],
                                                 dense_squeeze_channels=128,
                                                 dense_out_channels=32,
                                                 out_channels=dense_out[0],
                                                 rate=1,
                                                 norm_layer=SynchronizedBatchNorm2d,
                                                 norm_kwargs={'momentum': cfg.TRAIN_BN_MOM}, ca=ca, memory_efficent=me)
        self.denseaspp2 = DenseASPPWithoutGlobal(in_channels=input_channels[1],
                                                 dense_squeeze_channels=384,
                                                 dense_out_channels=128,
                                                 out_channels=dense_out[1],
                                                 rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                                                 norm_layer=SynchronizedBatchNorm2d,
                                                 norm_kwargs={'momentum': cfg.TRAIN_BN_MOM}, ca=ca, memory_efficent=me)
        self.denseaspp3 = DenseASPP(in_channels=input_channels[2],
                                    dense_squeeze_channels=1024,
                                    dense_out_channels=256,
                                    out_channels=dense_out[2],
                                    rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                                    norm_layer=SynchronizedBatchNorm2d,
                                    norm_kwargs={'momentum': cfg.TRAIN_BN_MOM}, ca=ca, memory_efficent=me)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

        self.upsample_all = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE // 8)
        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE // 4)

        indim = 256
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1,
                      padding=cfg.MODEL_SHORTCUT_KERNEL // 2, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(sum(dense_out), cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,
                      bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()

    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        # for l in layers:
        #     print(l.shape)

        dense_aspp3 = self.denseaspp3(layers[-1])
        dense_aspp3 = self.dropout3(dense_aspp3)
        dense_aspp3 = self.upsample3(dense_aspp3)

        dense_aspp2 = self.denseaspp2(layers[-2])
        dense_aspp2 = self.dropout2(dense_aspp2)
        dense_aspp2 = self.upsample2(dense_aspp2)

        dense_aspp1 = self.denseaspp1(layers[-3])
        dense_aspp1 = self.dropout1(dense_aspp1)

        # print(dense_aspp1.shape, dense_aspp2.shape, dense_aspp3.shape)

        feature_cat = torch.cat([dense_aspp3, dense_aspp2, dense_aspp1], dim=1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = self.upsample_all(result)
        return result


def _me_function_factory(conv, cat=False):
    def me_function(x):
        out = conv(x)
        return out

    def me_cat_function(*input):
        x = torch.cat(input, dim=1)
        out = conv(x)
        return out

    return me_cat_function if cat else me_function


class DeepLabv3PlusMultiDenseASPPv3(nn.Module):
    """
    coder & decoder
    """

    def __init__(self, cfg, v=3, ga=False, me=True):
        """
        :param cfg: configuration
        :param prime: use prime denseblock or not
        :param ca: use channel attention or not
        """
        super().__init__()
        self.backbone = None
        self.backbone_layers = None
        cfg.MODEL_BACKBONE = 'xception3strides'
        input_channels = (256, 728, 2048)


        inter_channels1 = {3:(128, 256, 1024),3.1:(128, 384, 1024)}.get(v)
        inter_channels2 = {3:(32, 64, 256),3.1:(32, 128, 256)}.get(v)
        dense_out = {3:(64, 128, 256),3.1:(64, 128, 256)}.get(v)


        DenseASPP = _DenseASPPBlockv2
        DenseASPPWithoutGlobal = _DenseASPPBlockWithoutGlobalv2
        self.denseaspp1 = DenseASPPWithoutGlobal(in_channels=input_channels[0],
                                                 inter_channels1=inter_channels1[0],
                                                 inter_channels2=inter_channels2[0],
                                                 out_channels=dense_out[0],
                                                 rate=1,
                                                 norm_layer=SynchronizedBatchNorm2d,
                                                 norm_kwargs={'momentum': cfg.TRAIN_BN_MOM}, global_attention=ga, memory_efficent=me)
        self.denseaspp2 = DenseASPPWithoutGlobal(in_channels=input_channels[1],
                                                 inter_channels1=inter_channels1[1],
                                                 inter_channels2=inter_channels2[1],
                                                 out_channels=dense_out[1],
                                                 rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                                                 norm_layer=SynchronizedBatchNorm2d,
                                                 norm_kwargs={'momentum': cfg.TRAIN_BN_MOM}, global_attention=ga, memory_efficent=me)
        self.denseaspp3 = DenseASPP(in_channels=input_channels[2],
                                    inter_channels1=inter_channels1[2],
                                    inter_channels2=inter_channels2[2],
                                    out_channels=dense_out[2],
                                    rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                                    norm_layer=SynchronizedBatchNorm2d,
                                    norm_kwargs={'momentum': cfg.TRAIN_BN_MOM}, global_attention=ga, memory_efficent=me)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

        self.upsample_all = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE // 8)
        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE // 4)

        indim = 256
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1,
                      padding=cfg.MODEL_SHORTCUT_KERNEL // 2, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(sum(dense_out), cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,
                      bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()
        self.memory_efficient = me

    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        # for l in layers:
        #     print(l.shape)

        dense_aspp3 = self.denseaspp3(layers[-1])
        dense_aspp3 = self.dropout3(dense_aspp3)
        dense_aspp3 = self.upsample3(dense_aspp3)

        dense_aspp2 = self.denseaspp2(layers[-2])
        dense_aspp2 = self.dropout2(dense_aspp2)
        dense_aspp2 = self.upsample2(dense_aspp2)

        dense_aspp1 = self.denseaspp1(layers[-3])
        dense_aspp1 = self.dropout1(dense_aspp1)

        if self.memory_efficient:
            result = cp.checkpoint(_me_function_factory(self.cat_conv, cat=True),
                                   dense_aspp3, dense_aspp2, dense_aspp1)
        else:
            feature_cat = torch.cat([dense_aspp3, dense_aspp2, dense_aspp1], 1)
            result = self.cat_conv(feature_cat)

        # feature_cat = torch.cat([dense_aspp3, dense_aspp2, dense_aspp1], dim=1)
        # result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = self.upsample_all(result)
        return result


class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features



def _conv_and_cat(conv):
    def me_function(x):
        out = conv(x)
        return torch.cat([out, x], dim=1)

    return me_function

class _DenseASPPBlockv2(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2, out_channels, rate,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, global_attention=False, memory_efficent=False):
        super().__init__()
        self.aspp_6 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 6 * rate,
                                     0.1, norm_layer, norm_kwargs)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 12 * rate,
                                      0.1, norm_layer, norm_kwargs)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 18 * rate,
                                      0.1, norm_layer, norm_kwargs)

        self.global_attention = global_attention
        if global_attention:
            self.ca = GAU(channels_high=in_channels, channels_low=3 * inter_channels2,
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels2 * 3, inter_channels2, kernel_size=1),
            norm_layer(inter_channels2, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(inplace=True)
        )
        self.global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels + inter_channels2 * 4, inter_channels2, kernel_size=1),
            norm_layer(inter_channels2, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(inplace=True)
        )
        self.last = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels2 * 5, out_channels, 1)
        )
        self.memory_efficient = memory_efficent

    def forward(self, x):
        [_, c, row, col] = x.size()
        if self.memory_efficient:
            x = cp.checkpoint(_conv_and_cat(self.aspp_6), x)
            x = cp.checkpoint(_conv_and_cat(self.aspp_12), x)
            x = cp.checkpoint(_conv_and_cat(self.aspp_18), x)
        else:
            aspp6 = self.aspp_6(x)
            x = torch.cat([aspp6, x], dim=1)

            aspp12 = self.aspp_12(x)
            x = torch.cat([aspp12, x], dim=1)

            aspp18 = self.aspp_18(x)
            x = torch.cat([aspp18, x], dim=1)

        if self.global_attention:
            x = self.ca(fms_high=x[:, -c:, :, :], fms_low=x[:, :-c, :, :])

        conv1x1 = self.conv1x1(x)
        x = torch.cat([conv1x1, x], dim=1)

        global_pooling = self.global_pooling(x)
        global_pooling = F.interpolate(global_pooling, (row, col), None, 'bilinear', True)
        x = torch.cat([global_pooling, x], dim=1)
        x = self.last(x[:, :-c, :, :])
        return x


class _DenseASPPBlockWithoutGlobalv2(nn.Module):
    """
    4-branches:
        conv1x1
        conv3x3 dilation=6
        conv3x3 dilation=12
        conv3x3 dilation=18
    """
    def __init__(self, in_channels, inter_channels1, inter_channels2, out_channels, rate,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, global_attention=False, memory_efficent=False):
        super().__init__()
        self.aspp_6 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 6 * rate,
                                     0.1, norm_layer, norm_kwargs)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 12 * rate,
                                      0.1, norm_layer, norm_kwargs)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 18 * rate,
                                      0.1, norm_layer, norm_kwargs)

        self.global_attention = global_attention
        if global_attention:
            self.ca = GAU(channels_high=in_channels, channels_low=3 * inter_channels2,
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels2 * 3, inter_channels2, kernel_size=1),
            norm_layer(inter_channels2, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(inplace=True)
        )
        self.last = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels2 * 4, out_channels, 1)
        )
        self.memory_efficient = memory_efficent

    def forward(self, x):
        [_, c, row, col] = x.size()

        if self.memory_efficient:
            x = cp.checkpoint(_conv_and_cat(self.aspp_6), x)
            x = cp.checkpoint(_conv_and_cat(self.aspp_12), x)
            x = cp.checkpoint(_conv_and_cat(self.aspp_18), x)
            if self.global_attention:
                x = self.ca(fms_high=x[:, -c:, :, :], fms_low=x[:, :-c, :, :])
            x = cp.checkpoint(_conv_and_cat(self.conv1x1), x)
        else:
            aspp6 = self.aspp_6(x)
            x = torch.cat([aspp6, x], dim=1)

            aspp12 = self.aspp_12(x)
            x = torch.cat([aspp12, x], dim=1)

            aspp18 = self.aspp_18(x)
            x = torch.cat([aspp18, x], dim=1)

            if self.global_attention:
                x = self.ca(fms_high=x[:, -c:, :, :], fms_low=x[:, :-c, :, :])

            conv1x1 = self.conv1x1(x)
            x = torch.cat([conv1x1, x], dim=1)
        x = self.last(x[:, :-c, :, :])
        return x


class GAU(nn.Module):
    """
    https://github.com/JaveyWang/Pyramid-Attention-Networks-pytorch/blob/master/networks.py
    """

    def __init__(self, channels_high, channels_low, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(GAU, self).__init__()
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = norm_layer(channels_low, **({} if norm_kwargs is None else norm_kwargs))

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = norm_layer(channels_low, **({} if norm_kwargs is None else norm_kwargs))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :param fm_mask:
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        out = torch.cat([fms_att, fms_high], dim=1)
        return out


if __name__ == '__main__':
    from model.deeplabv3plus import Configuration

    cfg = Configuration()
    model = DeepLabv3PlusMultiDenseASPPv3(cfg)
    print(model)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(y.shape)
    # print(model)
    pass
