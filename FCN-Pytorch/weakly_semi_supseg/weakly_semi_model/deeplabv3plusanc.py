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
from criterion import label_to_one_hot


class CategoryAwareSpacialAttention(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        """
        branch 1~4 is Atrous Conv with dilation rate  [0 (conv-1x1), 6, 12, 18] * rate
        branch 5 is global pooling
        :param dim_in: in_channel
        :param dim_out: out_channel
        :param rate: dilation rate coefficient
        :param bn_mom:
        """
        super().__init__()
        dim_min = max(dim_out // 16, 32)
        self.dim_min = dim_min
        self.dim_out = dim_out

        self.edge = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=rate, dilation=rate, bias=True),
        )

        self.pixel = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, bias=True),
        )

        self.category_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        self.category_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
        )
        self.mlp = nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, bias=True)

    def forward(self, x):
        [b, c, h, w] = x.size()
        edge = self.edge(x)
        pixel = self.pixel(x)
        ca = self.category_avg(x)
        cm = self.category_max(x)
        category_att = (self.mlp(ca) + self.mlp(cm)).sigmoid()
        # print(category_att)
        spatial_att = (edge + pixel).sigmoid()
        return category_att + spatial_att


class deeplabv3plus(nn.Module):
    def __init__(self, cfg):
        super(deeplabv3plus, self).__init__()
        self.cfg = cfg
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

        multi = True
        if multi:
            self.bbox_attention1 = CategoryAwareSpacialAttention(cfg.MODEL_NUM_CLASSES ,
                                                                input_channel,
                                                                rate=1,
                                                                bn_mom=cfg.TRAIN_BN_MOM)

            self.bbox_attention2 = CategoryAwareSpacialAttention(cfg.MODEL_NUM_CLASSES,
                                                                indim,
                                                                rate=1,
                                                                bn_mom=cfg.TRAIN_BN_MOM)
        else:
            self.bbox_attention1 = nn.Sequential(
                nn.Conv2d(cfg.MODEL_NUM_CLASSES, input_channel, 3, 1, padding=1, bias=True),
                nn.Sigmoid()
            )
            self.bbox_attention2 = nn.Sequential(
                nn.Conv2d(cfg.MODEL_NUM_CLASSES, indim, 3, 1, padding=1, bias=True),
                nn.Sigmoid()
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

    def forward(self, x, bbox_target_one_hot):
        # bbox_target_one_hot = label_to_one_hot(bbox_target, n_class=self.cfg.MODEL_NUM_CLASSES)
        if not isinstance(bbox_target_one_hot, torch.FloatTensor):
            bbox_target_one_hot = bbox_target_one_hot.float()

        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()

        bbox_size1 = F.interpolate(bbox_target_one_hot, size=layers[-1].shape[2:], mode='nearest')
        bbox_size2 = F.interpolate(bbox_target_one_hot, size=layers[0].shape[2:], mode='nearest')

        bbox_attention1 = self.bbox_attention1(bbox_size1)
        bbox_attention2 = self.bbox_attention2(bbox_size2)
        layers[-1] = bbox_attention1 * layers[-1]
        layers[0] = bbox_attention2 * layers[0]

        # for l in layers:
        #     print(l.shape)
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)
        # print(feature_aspp.shape)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = self.upsample4(result)

        if self.cfg.REFINE and not self.training:
            bbox_target_one_hot[:, 0] = 1.
            result = bbox_target_one_hot * result
            return result
        else:
            return result


class deeplabv3pluslabel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
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
        self.bbox_attention1 = nn.Sequential(
            nn.Conv2d(cfg.MODEL_NUM_CLASSES, input_channel, 1, 1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.bbox_attention2 = nn.Sequential(
            nn.Conv2d(cfg.MODEL_NUM_CLASSES, indim, 1, 1, padding=0, bias=False),
            nn.Sigmoid()
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

    def forward(self, x, bbox_target_one_hot):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        labels = bbox_target_one_hot.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0].float()
        # print(labels)
        bbox_attention1 = self.bbox_attention1(labels)
        bbox_attention2 = self.bbox_attention2(labels)
        layers[-1] = bbox_attention1 * layers[-1]
        layers[0] = bbox_attention2 * layers[0]

        # for l in layers:
        #     print(l.shape)
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)
        # print(feature_aspp.shape)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = self.upsample4(result)
        if self.cfg.REFINE and not self.training:
            result = labels * result
            return result
        else:
            return result


if __name__ == '__main__':
    from weakly_semi_supseg.weakly_semi_model import Configuration

    cfg = Configuration()
    cfg.REFINE = True
    model = deeplabv3plus(cfg)
    print(model.__class__.__name__)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    bbox = torch.randint(low=0, high=2, size=(1, 21, 32, 32))
    y = model(x, bbox)
    print(y.shape)
    # print(model)
    pass
