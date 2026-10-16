import torch
from torch import nn
from torchvision import models as Models
from os import path as osp
import os


def get_backbone(model_name: str, pretrain=True, freeze_front=False):
    r"""
    get pre-trained base-network for yolo-v1,
    children[:5] do not require grad
    :param model_name: name of model
    :return: pre-layer of pre-trained model without FC
    """
    model_dict = {
        'resnet18': Models.resnet18(pretrain),
        'resnet50': Models.resnet50(pretrain),
        'resnet101': Models.resnet101(pretrain)
    }
    '''
    list(resnet18.children()) consists of following modules
      [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
      [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
      [5] = Sequential(Bottleneck...),
      [6] = Sequential(Bottleneck...),
      [7] = Sequential(Bottleneck...),
      [8] = AvgPool2d, 
      [9] = Linear
    '''
    # when input shape is [, 3, 448, 448], output shape is:
    feature_maps_shape = {
        'resnet18': 512,
        'resnet50': 2048,
        'resnet101': 2048
    }
    features = list(model_dict.get(model_name).children())[:-2]
    if freeze_front:
        for parameters in [feature.parameters() for i, feature in enumerate(features) if i <= 4]:
            for parameter in parameters:
                parameter.requires_grad = False
    return features, feature_maps_shape.get(model_name)


class BaseNet(nn.Module):
    def __init__(self, pretrained=True, model='resnet101', freeze_front=False):
        super().__init__()
        features, self.last_layers = get_backbone(model_name=model, pretrain=pretrained, freeze_front=freeze_front)
        self.conv_1 = features[0]
        self.conv_2 = nn.Sequential(*features[1:5])
        self.conv_3 = features[5]
        self.conv_4 = features[6]
        self.conv_5 = features[7]

    def forward(self, x):
        output = {}
        x = self.conv_1(x)
        output['x1'] = x

        x = self.conv_2(x)
        output['x2'] = x

        x = self.conv_3(x)
        output['x3'] = x

        x = self.conv_4(x)
        output['x4'] = x

        x = self.conv_5(x)
        output['x5'] = x
        return output


class FCN2sResNet(nn.Module):
    def __init__(self, pretrained_net, last_layers, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(last_layers, 1024, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))  # size=(N, 512, x.H/16, x.W/16)
        score = score + x4  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8  )
        score = score + x3  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2  # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1  # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


def get_fcn_resnet_model(name: str, n_class=21):
    if name.lower() == 'fcn2sresnet101':
        model = FCN2sResNet(pretrained_net=BaseNet(),
                            last_layers=BaseNet().last_layers, n_class=n_class)
        return model
    else:
        raise Exception(f'*** model name wrong, {name} not legal, '
                        f'please choose from fcn2sresnet101')


if __name__ == '__main__':
    features, info = get_backbone('resnet101')
    for _, f in enumerate(features, start=0):
        print(f'{_}\n', f)
    pass
    base_net = BaseNet()
    x = torch.randn([4, 3, 224, 224])
    # y = base_net(x)
    # for i in y:
    #     print(y[i].shape)

    print(base_net)
    fcn = get_fcn_resnet_model('fcn2sresnet101')
    print(fcn)
