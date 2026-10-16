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


class deeplabv3plusCA(nn.Module):
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
        self.shortcut_attention = GlobalAttention(low_dim=cfg.MODEL_SHORTCUT_DIM, high_dim=cfg.MODEL_ASPP_OUTDIM)
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

    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_attentioned = self.shortcut_attention(low=feature_shallow, high=feature_aspp)
        # print(feature_attentioned.shape)
        feature_cat = torch.cat([feature_aspp, feature_attentioned], 1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = self.upsample4(result)
        return result


class GlobalAttention(nn.Module):
    def __init__(self, low_dim, high_dim):
        super().__init__()
        self.con1x1_1 = nn.Conv2d(in_channels=(low_dim + high_dim), out_channels=low_dim,
                                  kernel_size=1, padding=0, bias=False)
        self.con1x1_2 = nn.Conv2d(in_channels=low_dim, out_channels=low_dim,
                                  kernel_size=1, padding=0, bias=False)

    def forward(self, low: torch.Tensor, high: torch.Tensor):
        low_1x1 = F.adaptive_avg_pool2d(low, 1)
        high_1x1 = F.adaptive_avg_pool2d(high, 1)
        x = self.con1x1_1(torch.cat([low_1x1, high_1x1], dim=1))
        attention = self.con1x1_2(F.relu(x)).sigmoid()

        return attention * low


if __name__ == '__main__':
    from model.deeplabv3plus import Configuration

    cfg = Configuration()
    model = deeplabv3plusCA(cfg)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(y.shape)
    # print(model)
    pass
