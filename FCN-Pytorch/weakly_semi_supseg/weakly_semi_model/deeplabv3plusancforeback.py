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

    def forward(self, x, bbox_target):
        bbox_target_fore_back = torch.zeros(bbox_target.shape)
        bbox_target_fore_back[bbox_target > 0] = 1
        bbox_target_fore_back = bbox_target_fore_back.long()
        bbox_target_one_hot = label_to_one_hot(bbox_target_fore_back, n_class=self.cfg.MODEL_NUM_CLASSES)
        # print(bbox_target[bbox_target > 0])
        # print(bbox_target_fore_back[bbox_target_fore_back > 0])

        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()

        bbox_size1 = F.interpolate(bbox_target_one_hot, size=layers[-1].shape[2:], mode='nearest').to(x.device)
        bbox_size2 = F.interpolate(bbox_target_one_hot, size=layers[0].shape[2:], mode='nearest').to(x.device)

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
        if self.training:
            return result
        else:
            bbox_one_hot = label_to_one_hot(bbox_target, n_class=self.cfg.MODEL_NUM_CLASSES)
            bbox_refine_result = torch.zeros(bbox_one_hot.shape)
            # print(result[:, 0, :, :].shape, bbox_refine_result[:, 0, :, :].shape)
            # use the model-predicted background
            bbox_refine_result[:, 0, :, :] = result[:, 0, :, :]
            bbox_refine_result[:, 1:, :, :] = result[:, 1, :, :].unsqueeze(dim=1) * bbox_one_hot[:, 1:, :, :]
            # refine the background
            bbox_refine_result[:, 0, :, :][bbox_one_hot[:, 1, :, :] > 0] = 1
            return bbox_refine_result


if __name__ == '__main__':
    from model.deeplabv3plus import Configuration

    cfg = Configuration()
    model = deeplabv3plus(cfg)
    print(model.__class__.__name__)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    bbox = torch.randint(low=0, high=21, size=(2, 1, 32, 32))
    y = model(x, bbox)
    print(y.shape)
    # print(model)
    pass
