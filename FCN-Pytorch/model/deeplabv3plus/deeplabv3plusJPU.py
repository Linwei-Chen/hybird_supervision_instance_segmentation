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


class deeplabv3plusJPU(nn.Module):
    def __init__(self, cfg, jpu='same width'):
        cfg.MODEL_BACKBONE = 'xception3strides'
        super().__init__()
        self.backbone = None
        self.backbone_layers = None

        self.jpu = JPU(in_channels=(256, 728, 2048), width=256,
                       norm_layer=SynchronizedBatchNorm2d, norm_kwargs={'momentum': cfg.TRAIN_BN_MOM},
                       up_kwargs={'mode': 'bilinear'})
        input_channel = 768 if jpu == 'same width' else sum([128, 364, 1024])

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

    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        x = self.jpu(layers)
        feature_aspp = self.aspp(x)
        feature_aspp = self.dropout1(feature_aspp)
        result = self.cls_conv(feature_aspp)
        result = self.upsample4(result)
        return result


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False,
                 norm_layer=SynchronizedBatchNorm2d, norm_kwargs=None):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes,
                               bias=bias)
        self.bn = norm_layer(inplanes, **({} if norm_kwargs is None else norm_kwargs))
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


# norm_layer=norm_layer,norm_kwargs=norm_kwargs

class JPU(nn.Module):
    """
    do not have separate dilated conv
    """

    def __init__(self, in_channels=(256, 728, 2048), width=512,
                 norm_layer=SynchronizedBatchNorm2d, norm_kwargs=None, up_kwargs=None):
        super(JPU, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            # nn.PixelShuffle(4),
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width, **({} if norm_kwargs is None else norm_kwargs)),
            nn.PReLU())
        self.conv4 = nn.Sequential(
            # nn.PixelShuffle(2),
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width, **({} if norm_kwargs is None else norm_kwargs)),
            nn.PReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(width, **({} if norm_kwargs is None else norm_kwargs)),
            nn.PReLU())

    def forward(self, inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs, align_corners=True)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs, align_corners=True)
        feat = torch.cat(feats, dim=1)
        # print(feat.shape)
        return feat


class JPUOrignal(nn.Module):
    def __init__(self, in_channels, width=512, norm_layer=SynchronizedBatchNorm2d, norm_kwargs=None, up_kwargs=None):
        super().__init__()
        self.up_kwargs = up_kwargs
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(width, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True))

        self.dilation1 = nn.Sequential(
            SeparableConv2d(3 * width, width, 3, padding=1, dilation=1, bias=False,
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs),
            norm_layer(width, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(3 * width, width, 3, padding=2, dilation=2, bias=False,
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs),
            norm_layer(width, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True))
        self.dilation3 = nn.Sequential(
            SeparableConv2d(3 * width, width, 3, padding=4, dilation=4, bias=False,
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs),
            norm_layer(width, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True))
        self.dilation4 = nn.Sequential(
            SeparableConv2d(3 * width, width, 3, padding=8, dilation=8, bias=False,
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs),
            norm_layer(width, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True))

    def forward(self, inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        size = feats[-1].size()[2:]
        feats[-2] = F.interpolate(feats[-2], size, **self.up_kwargs, align_corners=True)
        feats[-3] = F.interpolate(feats[-3], size, **self.up_kwargs, align_corners=True)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat),
                          self.dilation3(feat), self.dilation4(feat)],dim=1)

        return feat


class deeplabv3plusJPUv2(nn.Module):
    def __init__(self, cfg):
        cfg.MODEL_BACKBONE = 'xception3strides'
        super().__init__()
        self.backbone = None
        self.backbone_layers = None

        self.jpu = JPUOrignal(in_channels=(256, 728, 2048), width=256,
                              norm_layer=SynchronizedBatchNorm2d, norm_kwargs={'momentum': cfg.TRAIN_BN_MOM},
                              up_kwargs={'mode': 'bilinear'})
        input_channel = 1024

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

    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        x = self.jpu(layers)
        feature_aspp = self.aspp(x)
        feature_aspp = self.dropout1(feature_aspp)
        result = self.cls_conv(feature_aspp)
        result = self.upsample4(result)
        return result


if __name__ == '__main__':
    from model.deeplabv3plus import Configuration

    cfg = Configuration()
    model = deeplabv3plusJPUv2(cfg)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(y.shape)
    # print(model)
    pass
