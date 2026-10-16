###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample, normalize, interpolate
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from model.deeplabv3plus.sync_batchnorm import SynchronizedBatchNorm2d
from model.deeplabv3plus.backbone import build_backbone

__all__ = ['DANet']


class DANet(nn.Module):
    r"""Fully Convolutional Networks for Semantic Segmentation
    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
    Reference:
        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015
    """

    def __init__(self, cfg, only_fuse=True):
        super().__init__()
        self.head = DANetHead(2048, cfg.MODEL_NUM_CLASSES, SynchronizedBatchNorm2d,
                              norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})
        self.backbone = None
        self.backbone_layers = None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(cfg.MODEL_BACKBONE, pretrained=True, os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()
        self.only_fuse = only_fuse

    def forward(self, x):
        imsize = x.size()[2:]
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        x = self.head(layers[-1])
        # print(x.shape)
        x = list(x)
        x[0] = interpolate(x[0], imsize, mode='bilinear', align_corners=True)
        x[1] = interpolate(x[1], imsize, mode='bilinear', align_corners=True)
        x[2] = interpolate(x[2], imsize, mode='bilinear', align_corners=True)

        outputs = [x[0]]
        outputs.append(x[1])
        outputs.append(x[2])

        # print(len(x))
        # x = interpolate(x, imsize, mode='bilinear', align_corners=True)
        # return x
        if self.only_fuse or not self.training:
            return outputs[0]
        else:
            return outputs


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels, **norm_kwargs),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels, **norm_kwargs),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels, **norm_kwargs),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels, **norm_kwargs),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

    def forward(self, x):
        # position attention
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        # channel attention
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        # fuse
        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)
        # return sasc_output


class PAM_Module(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


import torch
import torch.nn as nn
from model.deeplabv3plus.sync_batchnorm import SynchronizedBatchNorm2d
from model.deeplabv3plus.backbone import build_backbone
from model.deeplabv3plus.ASPP import ASPP


class DANetHeadv2(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs):
        super().__init__()
        if out_channels is None:
            inter_channels = in_channels // 4
        else:
            inter_channels = out_channels
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels, **norm_kwargs),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels, **norm_kwargs),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels, **norm_kwargs),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels, **norm_kwargs),
                                    nn.ReLU())

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        # sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        # sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        return (feat_sum, sa_conv, sc_conv)


class deeplabv3plusDA(nn.Module):
    """
    replace the ASPP with DA module
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = 2048
        cfg.MODEL_ASPP_OUTDIM = 512
        self.head = DANetHeadv2(input_channel, None, SynchronizedBatchNorm2d,
                                norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE // 4)

        indim = 256
        cfg.MODEL_SHORTCUT_DIM = 96
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

    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        # for l in layers:
        #     print(l.shape)
        feature = self.head(layers[-1])[0]
        feature = self.dropout1(feature)
        feature = self.upsample_sub(feature)
        # print(feature_aspp.shape)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature, feature_shallow], 1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = self.upsample4(result)
        return result


class deeplabv3plusASPPDA(nn.Module):
    r"""
    deeplabv3 +dual attention + aspp
    """

    def __init__(self, cfg, only_fuse=True):
        super().__init__()
        input_channel = 2048
        self.aspp = ASPP(dim_in=input_channel,
                         dim_out=cfg.MODEL_ASPP_OUTDIM,
                         rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                         bn_mom=cfg.TRAIN_BN_MOM)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE // 4)

        self.dahead = DANetHeadv2(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, SynchronizedBatchNorm2d,
                                  norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})
        self.backbone = None
        self.backbone_layers = None
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(256, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1,
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
        self.cls_conv_1 = nn.Sequential(nn.Dropout2d(0.1, False),
                                        nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0))
        self.cls_conv_2 = nn.Sequential(nn.Dropout2d(0.1, False),
                                        nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0))
        self.cls_conv_3 = nn.Sequential(nn.Dropout2d(0.1, False),
                                        nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(cfg.MODEL_BACKBONE, pretrained=True, os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()
        self.only_fuse = only_fuse

    def forward(self, x):
        imsize = x.size()[2:]
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)
        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)

        result = self.cat_conv(feature_cat)

        x = self.dahead(result)
        x = list(x)

        # for i in x: print(i.shape)

        x[0] = self.cls_conv_1(x[0])
        x[1] = self.cls_conv_2(x[1])
        x[2] = self.cls_conv_3(x[2])

        x[0] = interpolate(x[0], imsize, mode='bilinear', align_corners=True)
        x[1] = interpolate(x[1], imsize, mode='bilinear', align_corners=True)
        x[2] = interpolate(x[2], imsize, mode='bilinear', align_corners=True)

        outputs = [x[0]]
        outputs.append(x[1])
        outputs.append(x[2])
        # x = interpolate(x, imsize, mode='bilinear', align_corners=True)
        # return x
        if self.only_fuse or not self.training:
            return outputs[0]
        else:
            return outputs


class deeplabv3plusDAASPP(nn.Module):
    r"""
    deeplabv3 +dual attention + aspp
    """

    def __init__(self, cfg, only_fuse=True):
        super().__init__()
        input_channel = 2048
        self.dahead = DANetHeadv2(input_channel, 512, SynchronizedBatchNorm2d,
                                  norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})
        self.aspp = ASPP(dim_in=512,
                         dim_out=cfg.MODEL_ASPP_OUTDIM,
                         rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                         bn_mom=cfg.TRAIN_BN_MOM)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE // 4)

        self.backbone = None
        self.backbone_layers = None
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(256, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1,
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
        self.cls_conv_1 = nn.Sequential(nn.Dropout2d(0.1, False),
                                        nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0))
        self.cls_conv_2 = nn.Sequential(nn.Dropout2d(0.1, False),
                                        nn.Conv2d(512, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0))
        self.cls_conv_3 = nn.Sequential(nn.Dropout2d(0.1, False),
                                        nn.Conv2d(512, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(cfg.MODEL_BACKBONE, pretrained=True, os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()
        self.only_fuse = only_fuse

    def forward(self, x):
        imsize = x.size()[2:]
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        da = self.dahead(layers[-1])
        da = list(da)

        feature_aspp = self.aspp(da[0])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)
        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv_1(result)
        result = self.upsample4(result)

        # for i in x: print(i.shape)

        da[1] = self.cls_conv_2(da[1])
        da[2] = self.cls_conv_3(da[2])

        da[1] = interpolate(da[1], imsize, mode='bilinear', align_corners=True)
        da[2] = interpolate(da[2], imsize, mode='bilinear', align_corners=True)

        outputs = [result]
        outputs.append(da[1])
        outputs.append(da[2])

        if self.only_fuse or not self.training:
            return outputs[0]
        else:
            return outputs


class deeplabv3plusPA(nn.Module):
    """
    backbone + position attention + aspp +decoder
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = 2048
        danet_out = 512
        self.PAM = nn.Sequential(
            nn.Conv2d(input_channel, danet_out, 3, padding=1, bias=False),
            SynchronizedBatchNorm2d(danet_out, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(True),
            PAM_Module(in_dim=danet_out),
            nn.Conv2d(danet_out, danet_out, 3, padding=1, bias=False),
            SynchronizedBatchNorm2d(danet_out, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(True))
        self.aspp = ASPP(dim_in=danet_out,
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

    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        feature = self.PAM(layers[-1])
        feature_aspp = self.aspp(feature)
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = self.upsample4(result)
        return result


from model.hrnet import get_hrnet_basebone


class DANetwithHRNet(nn.Module):
    """
    hrnet + DA module
    """

    def __init__(self, cfg, pretrain=True, only_fuse=True):
        super().__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = 720
        self.dahead = DANetHeadv2(input_channel, input_channel, SynchronizedBatchNorm2d,
                                  norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})

        self.only_fuse = only_fuse
        # self.use_aspp = use_aspp
        # if use_aspp:
        #     self.aspp = ASPP(dim_in=input_channel,
        #                      dim_out=cfg.MODEL_ASPP_OUTDIM,
        #                      rate=16 // cfg.MODEL_OUTPUT_STRIDE,
        #                      bn_mom=cfg.TRAIN_BN_MOM)
        self.dropout1 = nn.Dropout(0.5)

        self.cls_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=input_channel,
                kernel_size=1,
                stride=1,
                padding=0),
            SynchronizedBatchNorm2d(input_channel, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=cfg.MODEL_NUM_CLASSES,
                kernel_size=1,
                stride=1,
                padding=0)
        )
        self.cls_conv_2 = nn.Sequential(nn.Dropout2d(0.1, False),
                                        nn.Conv2d(input_channel, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0))
        self.cls_conv_3 = nn.Sequential(nn.Dropout2d(0.1, False),
                                        nn.Conv2d(input_channel, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0))

        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = get_hrnet_basebone(cfg.MODEL_NUM_CLASSES, pretrain=pretrain)

    def forward(self, x):
        imsize = x.shape[2:]
        x = self.backbone(x)
        features = list(self.dahead(x))
        # if self.use_aspp:
        #     feature_aspp = self.aspp(features[0])
        features[0] = self.dropout1(features[0])
        features[0] = self.cls_conv(features[0])
        features[0] = self.upsample4(features[0])

        if not self.only_fuse:
            features[1] = self.cls_conv_2(features[1])
            features[2] = self.cls_conv_3(features[2])

            features[1] = interpolate(features[1], imsize, mode='bilinear', align_corners=True)
            features[2] = interpolate(features[2], imsize, mode='bilinear', align_corners=True)
        if self.only_fuse or not self.training:
            return features[0]
        else:
            return features


if __name__ == '__main__':
    from model.deeplabv3plus import Configuration
    from logger import ModelSaver

    cfg = Configuration()
    # model = deeplabv3plusDA(cfg)
    cfg.MODEL_OUTPUT_STRIDE = 8
    """
    model = deeplabv3plusDAASPP(cfg)
    model = nn.DataParallel(model)
    ModelSaver.load_tool(
        load_path='/Users/chenlinwei/code/20190717FCN-Pytorch/danetaspp/danetaspp_miou_0.7744_pa_0.9506.pkl',
        model=model)
        """
    model = deeplabv3plusPA(cfg)
    # model = DANetwithHRNet(cfg, only_fuse=False)
    model.eval()
    # model = DANet(cfg, only_fuse=False)
    # model.eval()

    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    for i in y:
        print(i.shape)
    pass
