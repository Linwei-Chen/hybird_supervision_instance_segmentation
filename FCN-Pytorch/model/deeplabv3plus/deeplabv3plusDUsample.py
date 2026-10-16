# ----------------------------------------
# Written by Linwei Chen
# ----------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.deeplabv3plus.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from model.deeplabv3plus.backbone import build_backbone
from model.deeplabv3plus.ASPP import ASPP


class deeplabv3plusDUpsample(nn.Module):
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

        self.DUsampling = DUpsampling(inplanes=cfg.MODEL_ASPP_OUTDIM, scale=4,
                                      num_class=cfg.MODEL_NUM_CLASSES, pad=0)
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

    def forward(self, x, target_one_hot=None, optimizer=None):
        # def forward(self, x):

        if target_one_hot is not  None and optimizer is not None:
            # for param in self.DUsampling.parameters():
            #     param.requires_grad = True

            self.train_DUsample(target_one_hot)
            optimizer.step()
            optimizer.zero_grad()
        else:
            # for param in self.DUsampling.parameters():
            #     param.requires_grad = False

            x_bottom = self.backbone(x)
            layers = self.backbone.get_layers()
            feature_aspp = self.aspp(layers[-1])
            feature_aspp = self.dropout1(feature_aspp)
            feature_aspp = self.upsample_sub(feature_aspp)

            feature_shallow = self.shortcut_conv(layers[0])
            feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
            result = self.cat_conv(feature_cat)
            result = self.DUsampling(result)
            return result

    def train_DUsample(self, target_one_hot, scale=4):
        N, C, H, W = target_one_hot.shape
        target_one_hot = target_one_hot.permute(0, 2, 3, 1)
        # N, H, W/sacle, C*scale
        target_one_hot = target_one_hot.contiguous().view((N, H, int(W / scale), C * scale))
        # N, W/sacle, H, C*scale
        target_one_hot = target_one_hot.permute(0, 2, 1, 3)

        target_one_hot = target_one_hot.contiguous().view((N, int(W / scale), int(H / scale), C * scale * scale))

        target_one_hot = target_one_hot.permute(0, 3, 2, 1)

        reconstructed = self.DUsampling.conv_w(self.DUsampling.conv_p(target_one_hot))
        reconstruct_loss = torch.mean(torch.pow(target_one_hot - reconstructed, 2))
        print(f'DUpsample loss:{reconstruct_loss.item()}')
        reconstruct_loss.backward()
        # should optimizer.step()


class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale, num_class=21, pad=0):
        super(DUpsampling, self).__init__()
        ## W matrix
        self.conv_w = nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding=pad, bias=False)
        ## P matrix
        self.conv_p = nn.Conv2d(num_class * scale * scale, inplanes, kernel_size=1, padding=pad, bias=False)

        self.scale = scale

    def forward(self, x):
        x = self.conv_w(x)
        N, C, H, W = x.size()

        # N, W, H, C
        x_permuted = x.permute(0, 3, 2, 1)

        # N, W, H*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, W, H * self.scale, int(C / (self.scale))))

        # N, H*scale, W, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, H*scale, W*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view(
            (N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        # N, C/(scale**2), H*scale, W*scale
        x = x_permuted.permute(0, 3, 1, 2)

        return x


if __name__ == '__main__':
    from model.deeplabv3plus import Configuration

    cfg = Configuration()
    model = deeplabv3plusDUpsample(cfg)
    model = nn.DataParallel(model)
    # model.
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(y.shape)
    # print(model)
    pass
