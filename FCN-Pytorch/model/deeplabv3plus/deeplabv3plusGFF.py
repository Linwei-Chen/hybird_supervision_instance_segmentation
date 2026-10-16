# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.deeplabv3plus.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from model.deeplabv3plus.backbone import build_backbone
from model.deeplabv3plus.ASPP import ASPP


class deeplabv3plusGFF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = 2048
        self.aspp = ASPP(dim_in=input_channel,
                         dim_out=cfg.MODEL_ASPP_OUTDIM,
                         rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                         bn_mom=cfg.TRAIN_BN_MOM)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE // 4)

        indim = 256
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1,
                      padding=cfg.MODEL_SHORTCUT_KERNEL // 2, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(cfg.MODEL_ASPP_OUTDIM + cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,
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

        self.gate_shallow = GatedConv2dWithActivation(indim, indim, kernel_size=1,
                                                      norm_layer=SynchronizedBatchNorm2d,
                                                      norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})
        self.gate_aspp = GatedConv2dWithActivation(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, kernel_size=1,
                                                   norm_layer=SynchronizedBatchNorm2d,
                                                   norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})

    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow, feature_shallow_mask = self.gate_shallow(layers[0])
        feature_aspp, feature_aspp_mask = self.gate_aspp(feature_aspp)

        feature_shallow_after_gff = feature_shallow + feature_aspp * (1 - feature_shallow_mask)
        feature_aspp_after_gff = feature_aspp + feature_shallow * (1 - feature_aspp_mask)

        # print(feature_shallow.shape)
        shortcut = self.shortcut_conv(feature_shallow_after_gff)
        feature_cat = torch.cat([feature_aspp_after_gff, shortcut], 1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = self.upsample4(result)
        return result


class GFFASPP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = 256 * 3
        self.aspp = ASPP(dim_in=input_channel,
                         dim_out=cfg.MODEL_ASPP_OUTDIM,
                         rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                         bn_mom=cfg.TRAIN_BN_MOM)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()

        in_channel = [256, 728, 2048]
        self.gate_1 = GatedConv2dWithActivation(in_channel[0], 256, kernel_size=1,
                                                norm_layer=SynchronizedBatchNorm2d,
                                                norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})
        self.gate_2 = GatedConv2dWithActivation(in_channel[1], 256, kernel_size=1,
                                                norm_layer=SynchronizedBatchNorm2d,
                                                norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})
        self.gate_3 = GatedConv2dWithActivation(in_channel[2], 256, kernel_size=1,
                                                norm_layer=SynchronizedBatchNorm2d,
                                                norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})

    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()

        size_1 = layers[0].shape[2:]
        feature_1, feature_1_mask = self.gate_1(layers[0])
        feature_2, feature_2_mask = self.gate_2(F.interpolate(layers[1], size_1, mode='bilinear', align_corners=True))
        feature_3, feature_3_mask = self.gate_3(F.interpolate(layers[2], size_1, mode='bilinear', align_corners=True))

        feature_1_after_gff = feature_1 + (1 - feature_1_mask) * (feature_2 + feature_3)
        feature_2_after_gff = feature_2 + (1 - feature_2_mask) * (feature_1 + feature_3)
        feature_3_after_gff = feature_3 + (1 - feature_3_mask) * (feature_2 + feature_1)

        feature_cat = torch.cat([feature_1_after_gff, feature_2_after_gff, feature_3_after_gff], 1)

        feature_aspp = self.aspp(feature_cat)
        feature_aspp = self.dropout1(feature_aspp)
        result = self.cls_conv(feature_aspp)
        result = self.upsample4(result)
        return result


# https://github.com/avalonstrel/GatedConvolution_pytorch/blob/master/models/networks.py
class GatedConv2dWithActivation(nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))

    :return: gated feature map, mask
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 norm_layer=None, norm_kwargs=None,
                 activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        self.batch_norm = norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))
        self.activation = activation
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride, padding, dilation, groups, bias)
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups, bias)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input).sigmoid()
        if self.activation is not None:
            x = self.activation(x) * mask
        else:
            x = x * mask
        x = self.batch_norm(x)
        return x, mask


if __name__ == '__main__':
    from model.deeplabv3plus import Configuration

    cfg = Configuration()
    model = GFFASPP(cfg)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(y.shape)
    # print(model)
    pass
