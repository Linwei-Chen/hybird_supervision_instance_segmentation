import torch
import torch.nn as nn
import torch.nn.functional as F
from model.deeplabv3plus.sync_batchnorm import SynchronizedBatchNorm2d


class RefineLabel(nn.Module):
    def __init__(self, cfg, out_channel=2):
        super().__init__()
        inter_channels = 64
        self.conv1 = nn.Sequential(nn.Conv2d(cfg.MODEL_NUM_CLASSES, inter_channels, 3, padding=1, bias=False),
                                   SynchronizedBatchNorm2d(inter_channels, momentum=cfg.TRAIN_BN_MOM),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   SynchronizedBatchNorm2d(inter_channels, momentum=cfg.TRAIN_BN_MOM),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   SynchronizedBatchNorm2d(inter_channels, momentum=cfg.TRAIN_BN_MOM),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   SynchronizedBatchNorm2d(inter_channels, momentum=cfg.TRAIN_BN_MOM),
                                   nn.ReLU())
        self.cls = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channel, 1))
        self.label_attention = nn.Sequential(
            nn.Conv2d(in_channels=cfg.MODEL_NUM_CLASSES, out_channels=inter_channels,
                      kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, bbox_target_one_hot):
        labels = bbox_target_one_hot.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0].float()
        label_attention = self.label_attention(labels)
        x = self.conv1(x)
        x = x * label_attention
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.cls(x)
        return x


class DualAttentionRefineLabel(nn.Module):
    def __init__(self, cfg, out_channel=2):
        super().__init__()
        self.head = DANetHead(in_channels=cfg.MODEL_NUM_CLASSES,
                              out_channels=out_channel,
                              norm_layer=SynchronizedBatchNorm2d,
                              norm_kwargs={'momentum': cfg.TRAIN_BN_MOM})

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.head(x)
        return x
        pass


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs):
        super(DANetHead, self).__init__()
        inter_channels = 64
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

        # self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
        # self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))

    def forward(self, x):
        # position attention
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        # sa_output = self.conv6(sa_conv)

        # channel attention
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        # sc_output = self.conv7(sc_conv)

        # fuse
        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        # output = [sasc_output]
        # output.append(sa_output)
        # output.append(sc_output)
        # return tuple(output)
        return sasc_output


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

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


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

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


class PositionAttentionRefine(nn.Module):
    def __init__(self, cfg, out_channel=2):
        super().__init__()
        inter_channels = 64
        self.conv1 = nn.Sequential(nn.Conv2d(cfg.MODEL_NUM_CLASSES, inter_channels, 3, padding=1, bias=False),
                                   SynchronizedBatchNorm2d(inter_channels, momentum=cfg.TRAIN_BN_MOM),
                                   nn.ReLU(), )
        self.PAM = PAM_Module(inter_channels)
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   SynchronizedBatchNorm2d(inter_channels, momentum=cfg.TRAIN_BN_MOM),
                                   nn.ReLU())
        self.cls = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channel, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.PAM(x)
        x = self.conv2(x)
        x = self.cls(x)
        return x


class PositionAttentionRefineLabel(nn.Module):
    def __init__(self, cfg, out_channel=2):
        super().__init__()
        inter_channels = 64
        self.conv1 = nn.Sequential(nn.Conv2d(cfg.MODEL_NUM_CLASSES, inter_channels, 3, padding=1, bias=False),
                                   SynchronizedBatchNorm2d(inter_channels, momentum=cfg.TRAIN_BN_MOM),
                                   nn.ReLU())
        # self.PAM = PAM_Module(inter_channels)
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   SynchronizedBatchNorm2d(inter_channels, momentum=cfg.TRAIN_BN_MOM),
                                   nn.ReLU())
        self.cls = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channel, 1))
        self.label_attention = nn.Sequential(
            nn.Conv2d(in_channels=cfg.MODEL_NUM_CLASSES, out_channels=inter_channels,
                      kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, bbox_target_one_hot):
        labels = bbox_target_one_hot.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0].float()
        label_attention = self.label_attention(labels)
        x = self.conv1(x)
        x = x * label_attention
        # x = self.PAM(x)
        x = self.conv2(x)
        x = self.cls(x)
        return x


if __name__ == '__main__':
    from model.deeplabv3plus import Configuration

    cfg = Configuration()
    x = torch.rand(2, 21, 32, 32)
    model = PositionAttentionRefineLabel(cfg, out_channel=2)
    y = model(x, x)
    print(y.shape)
