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
import torch.utils.checkpoint as cp


class DeepLabv3PlusBalancedDenseASPP(nn.Module):
    """
    dense-block: 3, 5, 11, 17 + global pooling
    prime-dense-block: 2, 3, 5, 7, 11, 13, 17, 19 + global pooling

    """
    def __init__(self, cfg, prime=False, ca=False, me=False, dense_out=None):
        """
        :param cfg: configuration
        :param prime: use prime denseblock or not
        :param ca: use channel attention or not
        """
        super().__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = 2048
        DenseASPP = _PrimeDenseASPPHead if prime else _DenseASPPHead
        if dense_out is None:
            dense_out = cfg.MODEL_ASPP_OUTDIM
        self.denseaspp = DenseASPP(in_channels=input_channel,
                                   dense_squeeze_channels=cfg.MODEL_ASPP_OUTDIM,
                                   dense_out_channels=cfg.MODEL_ASPP_OUTDIM // 4,
                                   out_channels=dense_out,
                                   rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                                   norm_layer=SynchronizedBatchNorm2d,
                                   norm_kwargs={'momentum': cfg.TRAIN_BN_MOM}, ca=ca, memory_efficent=me)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE // 4)

        indim = 256
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1,
                      padding=cfg.MODEL_SHORTCUT_KERNEL // 2, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
        )
        self.global_attention = ca
        if ca:
            self.ca = GAU(channels_high=dense_out, channels_low=cfg.MODEL_SHORTCUT_DIM,
                          norm_layer=SynchronizedBatchNorm2d, norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})

        self.cat_conv = nn.Sequential(
            nn.Conv2d(dense_out + cfg.MODEL_SHORTCUT_DIM, dense_out, 3, 1, padding=1, bias=True),
            SynchronizedBatchNorm2d(dense_out, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(dense_out, dense_out, 3, 1, padding=1, bias=True),
            SynchronizedBatchNorm2d(dense_out, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(dense_out, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
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
        feature_aspp = self.denseaspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        if self.global_attention:
            feature_cat = self.ca(fms_high=feature_aspp, fms_low=feature_shallow)
        else:
            feature_cat = torch.cat([feature_shallow, feature_aspp], 1)

        result = self.cat_conv(feature_cat)

        result = self.cls_conv(result)

        result = self.upsample4(result)
        return result


class _PrimeDenseASPPHead(nn.Module):
    def __init__(self, in_channels, dense_squeeze_channels, dense_out_channels, out_channels,
                 rate, norm_layer=SynchronizedBatchNorm2d, norm_kwargs=None, ca=False, **kwargs):
        super().__init__()
        self.dense_aspp_block = _PrimeDenseASPPBlock(in_channels, dense_squeeze_channels, dense_out_channels,
                                                     rate, norm_layer, norm_kwargs, global_attention=ca)
        self.block = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dense_out_channels * 10, out_channels, 1)
        )

    def forward(self, x):
        x = self.dense_aspp_block(x)
        # print(x.shape)
        return self.block(x)


class _DenseASPPHead(nn.Module):
    def __init__(self, in_channels, dense_squeeze_channels, dense_out_channels, out_channels,
                 rate, norm_layer=SynchronizedBatchNorm2d, norm_kwargs=None, ca=False, memory_efficent=False, **kwargs):
        super(_DenseASPPHead, self).__init__()
        self.dense_aspp_block = _DenseASPPBlock(in_channels, dense_squeeze_channels, dense_out_channels,
                                                rate, norm_layer, norm_kwargs,
                                                global_attention=ca, memory_efficent=memory_efficent)
        self.block = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dense_out_channels * 6, out_channels, 1)
        )

    def forward(self, x):
        x = self.dense_aspp_block(x)
        # print(x.shape)
        return self.block(x)


class _DenseASPPWithoutGlobalHead(nn.Module):
    def __init__(self, in_channels, dense_squeeze_channels, dense_out_channels, out_channels,
                 rate, norm_layer=SynchronizedBatchNorm2d, norm_kwargs=None, ca=False, memory_efficent=False, **kwargs):
        super().__init__()
        self.dense_aspp_block = _DenseASPPBlockWithoutGlobal(in_channels, dense_squeeze_channels, dense_out_channels,
                                                             rate, norm_layer, norm_kwargs,
                                                             global_attention=ca, memory_efficent=memory_efficent)
        self.block = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dense_out_channels * 6, out_channels, 1)
        )

    def forward(self, x):
        x = self.dense_aspp_block(x)
        # print(x.shape)
        return self.block(x)


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


def _me_function_factory(conv):
    def me_function(x):
        out = conv(x)
        return torch.cat([out, x], dim=1)

    return me_function


class _DenseASPPBlock(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2, rate,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, global_attention=False, memory_efficent=False):
        super(_DenseASPPBlock, self).__init__()
        self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 3 * rate, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 5 * rate,
                                     0.1, norm_layer, norm_kwargs)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 11 * rate,
                                      0.1, norm_layer, norm_kwargs)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 17 * rate,
                                      0.1, norm_layer, norm_kwargs)

        self.global_attention = global_attention
        if global_attention:
            self.ca = GAU(channels_high=in_channels, channels_low=4 * inter_channels2,
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels2 * 4, inter_channels2, kernel_size=1),
            norm_layer(inter_channels2, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(inplace=True)
        )
        self.global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels + inter_channels2 * 5, inter_channels2, kernel_size=1),
            norm_layer(inter_channels2, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(inplace=True)
        )
        self.memory_efficient = memory_efficent

    def forward(self, x):
        [_, c, row, col] = x.size()
        if self.memory_efficient:
            x = cp.checkpoint(_me_function_factory(self.aspp_3), x)
            x = cp.checkpoint(_me_function_factory(self.aspp_6), x)
            x = cp.checkpoint(_me_function_factory(self.aspp_12), x)
            x = cp.checkpoint(_me_function_factory(self.aspp_18), x)
        else:
            aspp3 = self.aspp_3(x)
            x = torch.cat([aspp3, x], dim=1)

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

        return x[:, :-c, :, :]


class _DenseASPPBlockWithoutGlobal(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2, rate,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, global_attention=False, memory_efficent=False):
        super().__init__()
        self.aspp_1 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 1 * rate, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_3 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 3 * rate,
                                     0.1, norm_layer, norm_kwargs)
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 5 * rate,
                                     0.1, norm_layer, norm_kwargs)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 11 * rate,
                                      0.1, norm_layer, norm_kwargs)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 17 * rate,
                                      0.1, norm_layer, norm_kwargs)

        self.global_attention = global_attention
        if global_attention:
            self.ca = GAU(channels_high=in_channels, channels_low=4 * inter_channels2,
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels2 * 5, inter_channels2, kernel_size=1),
            norm_layer(inter_channels2, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(inplace=True)
        )
        self.memory_efficient = memory_efficent

    def forward(self, x):
        [_, c, row, col] = x.size()

        if self.memory_efficient:
            x = cp.checkpoint(_me_function_factory(self.aspp_1), x)
            x = cp.checkpoint(_me_function_factory(self.aspp_3), x)
            x = cp.checkpoint(_me_function_factory(self.aspp_6), x)
            x = cp.checkpoint(_me_function_factory(self.aspp_12), x)
            x = cp.checkpoint(_me_function_factory(self.aspp_18), x)
            if self.global_attention:
                x = self.ca(fms_high=x[:, -c:, :, :], fms_low=x[:, :-c, :, :])
            x = cp.checkpoint(_me_function_factory(self.conv1x1), x)
        else:
            aspp1 = self.aspp_3(x)
            x = torch.cat([aspp1, x], dim=1)

            aspp3 = self.aspp_3(x)
            x = torch.cat([aspp3, x], dim=1)

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

        return x[:, :-c, :, :]


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


class _PrimeDenseASPPBlock(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2, rate,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, global_attention=False):
        super().__init__()
        self.aspp_2 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 2 * rate, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_3 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 3 * rate,
                                     0.1, norm_layer, norm_kwargs)
        self.aspp_5 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 5 * rate,
                                     0.1, norm_layer, norm_kwargs)
        self.aspp_7 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 7 * rate,
                                     0.1, norm_layer, norm_kwargs)
        self.aspp_11 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 11 * rate,
                                      0.1, norm_layer, norm_kwargs)
        self.aspp_13 = _DenseASPPConv(in_channels + inter_channels2 * 5, inter_channels1, inter_channels2, 13 * rate,
                                      0.1, norm_layer, norm_kwargs)
        self.aspp_17 = _DenseASPPConv(in_channels + inter_channels2 * 6, inter_channels1, inter_channels2, 17 * rate,
                                      0.1, norm_layer, norm_kwargs)
        self.aspp_19 = _DenseASPPConv(in_channels + inter_channels2 * 7, inter_channels1, inter_channels2, 19 * rate,
                                      0.1, norm_layer, norm_kwargs)

        self.global_attention = global_attention
        if global_attention:
            self.ca = GAU(channels_high=in_channels, channels_low=8 * inter_channels2,
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels2 * 8, inter_channels2, kernel_size=1),
            norm_layer(inter_channels2, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(inplace=True)
        )
        self.global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels + inter_channels2 * 9, inter_channels2, kernel_size=1),
            norm_layer(inter_channels2, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        [_, c, row, col] = x.size()

        aspp2 = self.aspp_2(x)
        x = torch.cat([aspp2, x], dim=1)

        aspp3 = self.aspp_3(x)
        x = torch.cat([aspp3, x], dim=1)

        aspp5 = self.aspp_5(x)
        x = torch.cat([aspp5, x], dim=1)

        aspp7 = self.aspp_7(x)
        x = torch.cat([aspp7, x], dim=1)

        aspp11 = self.aspp_11(x)
        x = torch.cat([aspp11, x], dim=1)

        aspp13 = self.aspp_13(x)
        x = torch.cat([aspp13, x], dim=1)

        aspp17 = self.aspp_17(x)
        x = torch.cat([aspp17, x], dim=1)

        aspp19 = self.aspp_19(x)
        x = torch.cat([aspp19, x], dim=1)

        if self.global_attention:
            x = self.ca(fms_high=x[:, -c:, :, :], fms_low=x[:, :-c, :, :])

        conv1x1 = self.conv1x1(x)
        x = torch.cat([conv1x1, x], dim=1)

        global_pooling = self.global_pooling(x)
        global_pooling = F.interpolate(global_pooling, (row, col), None, 'bilinear', True)
        x = torch.cat([global_pooling, x], dim=1)

        return x[:, :-c, :, :]


if __name__ == '__main__':
    from model.deeplabv3plus import Configuration

    cfg = Configuration()
    cfg.MODEL_ASPP_OUTDIM = 1024
    model = DeepLabv3PlusBalancedDenseASPP(cfg, False, False, me=True, dense_out=256)
    print(model)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(y.shape)
    # print(model)
    pass
