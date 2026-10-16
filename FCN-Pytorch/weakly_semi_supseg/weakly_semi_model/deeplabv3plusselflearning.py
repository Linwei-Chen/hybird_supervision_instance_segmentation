# ----------------------------------------
# Written by charles
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
from weakly_semi_supseg.weakly_semi_model.refine_model import RefineLabel
from tqdm import tqdm
from train_utils import poly_adjust_lr
from torchvision import transforms


class SelfLearningBase(nn.Module):
    """
    normal: do not use the aucillary info like bbox, label
    info: use the ancillary info
    refine-learning: refine the output by label or bbox first and then softmax, as for bbox, bkgd should set as zero first
    refine: output the reuslt refined by label or bbox
    self-learning: softmax first and then refine the ouput by label or bbox

    """

    def __init__(self):
        super().__init__()
        self.mode = 'normal'

    def refine(self):
        self.mode = 'refine'
        # self.train()
        print(f'---> mode:{self.mode} | trianing:{self.training}')

    def normal(self):
        self.mode = 'normal'
        # self.train()
        print(f'---> mode:{self.mode} | trianing:{self.training}')

    def info(self):
        self.mode = 'info'
        # self.train()
        print(f'---> mode:{self.mode} | trianing:{self.training}')

    def info_eval(self):
        self.mode = 'info'
        # self.eval()
        print(f'---> mode:{self.mode} | trianing:{self.training}')

    def normal_eval(self):
        """
        ignore the bbox_targets
        :return:
        """
        self.mode = 'normal'
        # self.eval()
        print(f'---> mode:{self.mode} | trianing:{self.training}')

    def refine_eval(self):
        self.mode = 'refine'
        # self.eval()
        print(f'---> mode:{self.mode} | trianing:{self.training}')

    def self_learning(self):
        self.mode = 'self_learning'
        # self.eval()
        print(f'---> mode:{self.mode} | trianing:{self.training}')

    def refine_learning(self):
        self.mode = 'refine_learning'
        # self.eval()
        print(f'---> mode:{self.mode} | trianing:{self.training}')

    def learn_extra(self):
        """
        learning to cls or detect
        :return:
        """
        self.mode = 'learn_extra'
        # self.train()
        print(f'---> mode:{self.mode} | trianing:{self.training}')


class deeplabv3plus(SelfLearningBase):
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
        self.mode = 'normal'

    def forward(self, x, bbox_target_one_hot=None):
        _, _, ih, iw = x.shape
        x = F.interpolate(x, size=(ih - ih % 16, iw - iw % 16), mode='bilinear', align_corners=True)

        if bbox_target_one_hot is None:
            bbox_target_one_hot = torch.zeros([x.size(0), self.cfg.MODEL_NUM_CLASSES, x.size(2), x.size(3)]).to(
                x.device)
        elif not isinstance(bbox_target_one_hot, torch.FloatTensor):
            bbox_target_one_hot = bbox_target_one_hot.float()

        if self.mode == 'normal':
            bbox_target_one_hot = torch.zeros(bbox_target_one_hot.shape).to(bbox_target_one_hot.device)

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

        result = F.interpolate(result, size=(ih, iw), mode='bilinear', align_corners=True)

        if self.mode == 'refine':
            bbox_target_one_hot[:, 0] = 1.
            result = bbox_target_one_hot * result
            return result
        elif self.mode == 'self_learning':
            probs = result.softmax(dim=1)
            bbox_target_one_hot[:, 0] = 1.
            probs = bbox_target_one_hot * probs
            return probs
        elif self.mode == 'refine_learning':
            bbox_target_one_hot[:, 0] = 1.
            result = bbox_target_one_hot * result
            probs = result.softmax(dim=1)
            return probs
        elif self.mode in ('normal', 'info'):
            return result

        else:
            raise NotImplementedError


class deeplabv3pluslabel(SelfLearningBase):
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
        self.mode = 'normal'

    def forward(self, x, bbox_target_one_hot=None):
        if bbox_target_one_hot is not None:
            labels = bbox_target_one_hot.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0].float()
        else:
            b, c = x.size(0), x.size(1)
            labels = torch.zeros([b, int(self.cfg.MODEL_NUM_CLASSES), 1, 1]).to(x.device)

        if self.mode == 'normal':
            labels = torch.zeros(labels.shape).to(x.device)
        bbox_target_one_hot = None

        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()

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

        if self.mode == 'refine':
            result = labels * result
            return result
        elif self.mode == 'self_learning':
            probs = result.softmax(dim=1)
            labels[:, 0] = 1.
            probs = labels * probs
            return probs
        elif self.mode == 'refine_learning':
            labels[:, 0] = 1.
            result = labels * result
            probs = result.softmax(dim=1)
            return probs
        elif self.mode in ('normal', 'info'):
            return result
        else:
            raise NotImplementedError


class deeplabv3pluslabelrefinelabel(deeplabv3pluslabel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.refine_model = RefineLabel(cfg, out_channel=2)

    def forward(self, x, bbox_target_one_hot):
        return super().forward(x, bbox_target_one_hot)

    def get_refine_model_loss(self, pre, bbox_target_one_hot, target, criterion):
        res = self.refine_model(pre, bbox_target_one_hot).softmax(dim=1)
        _, mask = pre.max(dim=1, keepdim=True)
        ones = torch.ones(mask.shape).to(pre.device)
        zeros = torch.zeros(mask.shape).to(pre.device)
        _255 = torch.full(mask.shape, fill_value=255.).to(pre.device)
        refine_target = torch.where(mask.float() == target.float(), ones, zeros)
        refine_target = torch.where(target != 255, refine_target, _255)
        loss = criterion(input=res, target=refine_target.long())
        loss = loss.mean()
        loss.backward()
        return loss


class deeplabv3pluslabelDIS(deeplabv3pluslabel):
    def __init__(self, cfg):
        super().__init__(cfg)
        from tools.visulizer import Visualizer
        self.img_rec = nn.Sequential(
            nn.Conv2d(in_channels=cfg.MODEL_NUM_CLASSES, out_channels=cfg.MODEL_NUM_CLASSES, kernel_size=5,
                      padding=2, stride=1, bias=True),
            # SynchronizedBatchNorm2d(cfg.MODEL_NUM_CLASSES, momentum=cfg.TRAIN_BN_MOM),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=cfg.MODEL_NUM_CLASSES, out_channels=16, kernel_size=3,
                      padding=1, stride=1, bias=True),
            # SynchronizedBatchNorm2d(16, momentum=cfg.TRAIN_BN_MOM),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3,
                      padding=1, stride=1, bias=True),
        )
        self.img_rec_decoder = Decoder(num_classes=3)
        self.rec_loss = nn.DataParallel(nn.MSELoss())
        self.visual = Visualizer(keys=['img_rec', 'img'])
        self.un_norm = transforms.Normalize(mean=[1 / i for i in [0.485, 0.456, 0.406]],
                                            std=[1 / i for i in [0.229, 0.224, 0.225]])

    def get_last_feature(self, x, bbox_target_one_hot):
        labels = bbox_target_one_hot.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0].float()
        if self.mode == 'normal':
            labels = torch.zeros(labels.shape).to(bbox_target_one_hot.device)

        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()

        bbox_attention1 = self.bbox_attention1(labels)
        bbox_attention2 = self.bbox_attention2(labels)
        layers[-1] = bbox_attention1 * layers[-1]
        layers[0] = bbox_attention2 * layers[0]

        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        result = self.cat_conv(feature_cat)

        return result

    def forward(self, x, bbox_target_one_hot):
        labels = bbox_target_one_hot.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0].float()
        if self.mode == 'normal':
            labels = torch.zeros(labels.shape).to(bbox_target_one_hot.device)

        if self.training:
            result = self.get_last_feature(x, bbox_target_one_hot)
            result = self.cls_conv(result)
            result = self.upsample4(result)

        else:
            result = self.DIS_output(x, bbox_target_one_hot)

        if self.mode == 'refine':
            result = labels * result
            return result
        elif self.mode == 'self_learning':
            probs = result.softmax(dim=1)
            probs = labels * probs
            return probs
        elif self.mode == 'refine_learning':
            labels[:, 0] = 1.
            result = labels * result
            probs = result.softmax(dim=1)
            return probs
        elif self.mode in ('normal', 'info'):
            return result

        else:
            raise NotImplementedError

    def normal_forward(self, x, bbox_target_one_hot):
        return super().forward(x, bbox_target_one_hot)

    def learn_reconstruct(self, x, bbox_target_one_hot, optimizer):
        """
        optimizer.zero_grad()
        # self.eval()
        with torch.no_grad():
            result = self.normal_forward(x, bbox_target_one_hot)
        # self.train()
        img_rec = self.img_rec(result)
        loss = self.rec_loss(input=img_rec, target=x)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        return loss.item()
        """
        # optimizer.zero_grad()
        # self.eval()
        with torch.no_grad():
            last_feature = self.get_last_feature(x, bbox_target_one_hot)
        # self.train()
        img_rec = self.img_rec_decoder(last_feature)
        # self.visual.display(image=self.un_norm(img_rec.detach().cpu()[0]), key='img_rec')
        self.visual.display(image=img_rec.detach().cpu()[0], key='img_rec')
        # self.visual.display(image=self.un_norm(x.cpu()[0]), key='img')
        self.visual.display(image=x.cpu()[0], key='img')
        loss = self.rec_loss(input=img_rec, target=x)
        loss = loss.mean()
        # loss.backward()
        # optimizer.step()
        return loss

    def DIS_output(self, x, bbox_target_one_hot, lr=100, max_iter=10):

        with torch.no_grad():
            last_feature = self.get_last_feature(x, bbox_target_one_hot).detach()
        last_feature.requires_grad = True

        for _ in tqdm(range(max_iter)):
            # lr_now = lr * (1 - _ / (max_iter + 1)) ** 0.9
            lr_now = lr
            with torch.enable_grad():
                """"
                pred = self.upsample4(self.cls_conv(last_feature))
                img_rec = self.img_rec(pred)
                """
                img_rec = self.img_rec_decoder(last_feature)
                l = self.rec_loss(input=img_rec, target=x).mean()
                l.backward()
                tqdm.write(f'loss:{l.item():.6f}')
                with torch.no_grad():
                    last_feature -= lr_now * last_feature.grad
                    last_feature.grad.zero_()

        self.zero_grad()
        result = self.upsample4(self.cls_conv(last_feature))
        return result

    def reconstruct_test(self, x, bbox_target_one_hot):
        pred = self.forward(x, bbox_target_one_hot).softmax(dim=1)
        img_rec = self.img_rec(pred)
        return img_rec


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = SynchronizedBatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


def conv3x3(in_planes, out_planes, stride=1, atrous=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1 * atrous, dilation=atrous, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    bn_mom = 0.0003

    def __init__(self, inplanes, planes, stride=1, atrous=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, atrous)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = SynchronizedBatchNorm2d(planes, momentum=self.bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = SynchronizedBatchNorm2d(planes, momentum=self.bn_mom)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(256, 128))
        self.layers.append(BasicBlock(inplanes=128, planes=128))
        self.output_conv = nn.ConvTranspose2d(
            128, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


def _test_DIS():
    # testing the DIS
    from weakly_semi_supseg.weakly_semi_model import Configuration
    from train_utils import config
    from dataset.weaklysemidataset.sbd_bbox_seg_dataset import get_sbd_data_loader
    from torchvision import transforms
    from logger import ModelSaver
    un_normalize = transforms.Normalize(mean=[1 / i for i in [0.485, 0.456, 0.406]],
                                        std=[1 / i for i in [0.229, 0.224, 0.225]])
    # state_dict = torch.load('/Users/chenlinwei/code/20190717FCN-Pytorch/weakly/deeplabv3pluslabeldis.pkl', map_location='cpu')
    cfg = Configuration()
    model = deeplabv3pluslabelDIS(cfg)
    # model.load_state_dict(state_dict)
    ModelSaver.load_tool('/Users/chenlinwei/code/20190717FCN-Pytorch/weakly/deeplabv3pluslabeldis.pkl', model)
    args = config()
    sbd_train_loader = get_sbd_data_loader(args, sub_mean=args.std > 0, stdize=args.std > 1, aug=args.sbd_aug)
    for step, (imgs, targets, bbox_targets) in enumerate(sbd_train_loader, start=1):
        model.normal()
        # model.info()
        model.train()
        img_rec = model.reconstruct_test(imgs, bbox_targets)

        # all_pixel = float(imgs.size(-2) * imgs.size(-1))

        model.train()
        model.eval()
        pred_train = model.normal_forward(imgs, bbox_targets)
        right_pixel = (pred_train[0].argmax(dim=0) == targets[0][0]).sum()
        print(f'normal : {right_pixel}')

        model.eval()
        pred_eval = model(imgs, bbox_targets)
        right_pixel_refine = (pred_eval[0].argmax(dim=0) == targets[0][0]).sum()
        print(f'refine : {right_pixel_refine}')

        transforms.ToPILImage()(pred_train[0].argmax(dim=0).float() * 20).show()
        transforms.ToPILImage()(un_normalize(imgs[0])).show()
        transforms.ToPILImage()(un_normalize(img_rec[0])).show()

        transforms.ToPILImage()(pred_eval[0].argmax(dim=0).float() * 20).show()
        pass


def _test_deeplabv3pluslabelPositionRefine():
    from weakly_semi_supseg.weakly_semi_model import Configuration
    from criterion import CrossEntropyLoss2D
    criterion = CrossEntropyLoss2D(n_calss=2)
    cfg = Configuration()
    cfg.REFINE = True
    model = deeplabv3pluslabelrefinelabel(cfg)
    x = torch.randn(2, 3, 32, 32)
    bbox = torch.randint(low=0, high=2, size=(2, 21, 32, 32)).float()
    pre = torch.randint(low=0, high=2, size=(2, 21, 32, 32)).float()
    target = torch.randint(low=0, high=2, size=(2, 1, 32, 32)).float()
    loss = model.get_refine_model_loss(pre, bbox, target, criterion)
    print(loss)


def _test():
    from weakly_semi_supseg.weakly_semi_model import Configuration

    cfg = Configuration()
    cfg.REFINE = True
    model = deeplabv3pluslabelPositionRefine(cfg)
    print(model.__class__.__name__)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    bbox = torch.randint(low=0, high=2, size=(2, 21, 32, 32))
    model.learn_extra()
    y = model(x, bbox)
    print(y.shape)

    # print(model)
    pass


class deeplabv3plus10kbbox(SelfLearningBase):
    """
    auxiliary info is label
    """

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
        self.mode = 'normal'

    def forward(self, x, bbox_target_one_hot):
        labels = bbox_target_one_hot.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0].float()

        if self.mode == 'normal':
            labels = torch.zeros(labels.shape).to(bbox_target_one_hot.device)

        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()

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

        if self.mode == 'refine':
            bbox_target_one_hot[:, 0] = 1.
            result = bbox_target_one_hot * result
            return result
        elif self.mode == 'self_learning':
            probs = result.softmax(dim=1)
            bbox_target_one_hot[:, 0] = 1.
            probs = bbox_target_one_hot * probs
            return probs
        elif self.mode == 'refine_learning':
            bbox_target_one_hot[:, 0] = 1.
            result = bbox_target_one_hot * result
            probs = result.softmax(dim=1)
            return probs
        elif self.mode in ('normal', 'info'):
            return result

        else:
            raise NotImplementedError


class deeplabv3plusnolabel(SelfLearningBase):
    def __init__(self, cfg, focal_bce=False):
        from torch import nn
        super().__init__()
        self.cfg = cfg
        self.focal_bce = focal_bce
        self.bce_loss = nn.BCELoss(reduction='none')
        self.backbone = None
        self.backbone_layers = None
        input_channel = 2048
        self.cls_anc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channel, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
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
        self.mode = 'normal'

    def forward(self, x, bbox_target_one_hot):
        labels = bbox_target_one_hot.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0].float()

        if self.mode in ('normal', 'refine', 'self_learning', 'learn_extra'):
            labels = torch.zeros(labels.shape).to(bbox_target_one_hot.device)

        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()

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

        if self.mode == 'refine':
            pt = self.cls_anc(layers[-1])
            result = pt * result
            return result
        elif self.mode == 'self_learning':
            probs = result.softmax(dim=1)
            pt = self.cls_anc(layers[-1])
            probs = pt * probs
            return probs
        elif self.mode == 'refine_learning':
            pt = self.cls_anc(layers[-1])
            result = pt * result
            probs = result.softmax(dim=1)
            return probs
        elif self.mode == 'learn_extra':
            pt = self.cls_anc(layers[-1])
            loss = self.bce_loss(pt, labels)
            if self.focal_bce:
                loss = torch.abs(pt.detach() - labels) ** 2 * loss
            loss = loss.unsqueeze(dim=0).mean()
            return loss
        elif self.mode in ('normal', 'info'):
            return result

        else:
            raise NotImplementedError


if __name__ == '__main__':
    # _test_deeplabv3pluslabelPositionRefine()
    _test_DIS()
    pass
