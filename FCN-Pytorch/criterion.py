# -*- coding: utf-8 -*-
"""
Created on 18-6-7 上午10:11

@author: ronghuaiyang
"""

import torch
import torch.nn as nn
from lovasz_losses import lovasz_softmax, lovasz_hinge


def get_criterion(args):
    if args.criterion.lower() == 'fce':
        print(f'*** Using criterion Focal Loss CE')
        criterion = FocalLoss(n_class=args.n_class, mode='ce')
    elif args.criterion.lower() == 'fbce':
        print(f'*** Using criterion Focal Loss BCE')
        criterion = FocalLoss(n_class=args.n_class, mode='bce')
    elif args.criterion.lower() == 'afbce':
        print(f'*** Using criterion Adaptive Focal Loss BCE')
        criterion = FocalLoss(n_class=args.n_class, mode='bce', use_alpha=True)

    elif args.criterion.lower() == 'lovasz+fce':
        print(f'*** Using criterion LovaszSoftMax + Focal CE')
        criterion = LovaszSoftMaxWithFCE(n_class=args.n_class, alpha=1)
    elif args.criterion.lower() == 'lovasz+ce':
        print(f'*** Using criterion LovaszSoftMax + CE')
        criterion = LovaszSoftMaxWithCE(n_class=args.n_class, alpha=1)
    elif args.criterion.lower() == 'lovasz+afce':
        print(f'*** Using criterion LovaszSoftMax +  Focal CE with Alpha')
        criterion = LovaszSoftMaxWithFCE(n_class=args.n_class, alpha=1, use_alpha=True)

    elif args.criterion.lower() == 'bce':
        print(f'*** Using criterion BCE')
        criterion = BCE2D(n_calss=args.n_class, ignore_index=255)
    elif args.criterion.lower() == 'ce':
        print(f'*** Using criterion CE')
        criterion = CrossEntropyLoss2D(n_calss=args.n_class, ignore_index=255)
    elif args.criterion.lower() == 'lovasz':
        print(f'*** Using criterion Lovasz')
        criterion = LovaszSoftMax()

    elif args.criterion.lower() == 'encnet+fbce':
        print(f'*** Using criterion EncNet + Focal BCE')
        criterion = EncNetLoss(loss=FocalLoss(n_class=args.n_class, mode='bce'), nclass=args.n_class)
    elif args.criterion.lower() == 'focalencnet+fbce':
        print(f'*** Using criterion Focal EncNet + Focal BCE')
        criterion = EncNetLoss(loss=FocalLoss(n_class=args.n_class, mode='bce'), nclass=args.n_class, se_focal=True)

    elif args.criterion.lower() == 'encnet+bce':
        print(f'*** Using criterion EncNet + BCE')
        criterion = EncNetLoss(loss=BCE2D(n_calss=args.n_class, ignore_index=255), nclass=args.n_class)
    elif args.criterion.lower() == 'focalencnet+lovasz+ce':
        print(f'*** Using criterion Focal EncNet + LovaszSoftMax + CE')
        criterion = EncNetLoss(loss=LovaszSoftMaxWithCE(n_class=args.n_class, alpha=1), nclass=args.n_class,
                               se_focal=True)
    elif args.criterion.lower() == 'focalencnet+bce':
        print(f'*** Using criterion Focal EncNet + BCE')
        criterion = EncNetLoss(loss=BCE2D(n_calss=args.n_class, ignore_index=255), nclass=args.n_class, se_focal=True)
    else:
        raise NotImplementedError(f'**** {args.criterion} not Implemented')
    print(f'===> criterion: {criterion.__class__.__name__}')
    return criterion


from torch.nn import functional as F


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss


class EncNetLoss(nn.Module):
    """2D Cross Entropy Loss with SE Loss"""

    def __init__(self, loss, se_loss=True, se_weight=0.2, nclass=19, aux=False,
                 aux_weight=0.4, weight=None, se_focal=False, ignore_index=255, **kwargs):
        super().__init__()
        self.loss = loss
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight, reduction='none')
        self.se_focal = se_focal

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if not self.aux and self.se_loss:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = self.loss(input=pred, target=target)
            pt = torch.sigmoid(se_pred)
            loss2 = self.bceloss(pt, se_target)

            if self.se_focal:
                loss2 = torch.abs(pt.detach() - se_target) ** 2 * loss2
            loss2 = loss2.mean()
            loss = loss1 + self.se_weight * loss2
            return loss
        else:
            raise NotImplementedError

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = torch.zeros(batch, nclass)
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()


class LovaszSoftMaxWithCE(nn.Module):
    def __init__(self, n_class, ignore_label=255, alpha=1):
        super().__init__()
        self.alpha = alpha
        self.l = LovaszSoftMax(ignore_label=ignore_label)
        self.ce = CrossEntropyLoss2D(n_class, ignore_index=ignore_label)

    def forward(self, input, target):
        """
        :param input: [bs, c, h, w],
        :param target: [bs, 1, h, w]
        :return: tensor
        """
        lovaszloss = self.alpha * self.l(input, target)
        celoss = self.ce(input, target)
        return lovaszloss + celoss


class LovaszSoftMaxWithFCE(nn.Module):
    def __init__(self, n_class, ignore_label=255, alpha=1, use_alpha=False):
        super().__init__()
        self.alpha = alpha
        self.l = LovaszSoftMax(ignore_label=ignore_label)
        self.fce = FocalLoss(n_class=n_class, mode='ce', use_alpha=use_alpha)

    def forward(self, input, target):
        """
        :param input: [bs, c, h, w],
        :param target: [bs, 1, h, w]
        :return: tensor
        """
        lovaszloss = self.alpha * self.l(input, target)
        fceloss = self.fce(input, target)
        return lovaszloss + fceloss


class LovaszSoftMax(nn.Module):
    def __init__(self, ignore_label=255):
        super().__init__()
        self.ignore_label = ignore_label

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return lovasz_softmax(input.softmax(dim=1), target.squeeze(dim=1), ignore=self.ignore_label)


def label_to_one_hot(targets: torch.Tensor, n_class: int, with_255: bool = False):
    """
    get one-hot tensor from targets, ignore the 255 label
    :param targets: long tensor[bs, 1, h, w]
    :param nlabels: int
    :return: float tensor [bs, nlabel, h, w]
    """
    # batch_size, _, h, w = targets.size()
    # res = torch.zeros([batch_size, nlabels, h, w])
    targets = targets.squeeze(dim=1)
    # print(targets.shape)
    zeros = torch.zeros(targets.shape).long().to(targets.device)

    # del 255.
    targets_ignore = targets >= n_class
    # print(targets_ignore)
    targets = torch.where(targets < n_class, targets, zeros)

    one_hot = torch.nn.functional.one_hot(targets, num_classes=n_class)
    if with_255:
        one_hot[targets_ignore] = 0
    else:
        one_hot[targets_ignore] = 255
    # print(one_hot[targets_ignore])
    one_hot = one_hot.transpose(3, 2)
    one_hot = one_hot.transpose(2, 1)
    # print(one_hot.size())
    return one_hot.float()


# FocalLoss的讲解
# https://zhuanlan.zhihu.com/p/32423092
# http://skyhigh233.com/blog/2018/04/04/focalloss/
class FocalLoss(nn.Module):

    def __init__(self,
                 mode='ce', n_class=21, mean=True,
                 gamma=2, eps=1e-7, use_alpha=False):
        super(FocalLoss, self).__init__()
        # self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.mode = mode
        self.n_class = n_class
        self.mean = mean
        self.alpha = {
            21: get_focal_loss_alpha().unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3) if mode == 'bce' else
            get_focal_loss_alpha()
        }
        self.use_alpha = use_alpha

    def forward(self, input: torch.Tensor, target: torch.Tensor):

        """

        :param input: [bs, c, h, w],
        :param target: [bs, 1, h, w]
        :return: tensor
        """

        if self.mode == 'bce':
            target_one_hot = label_to_one_hot(target, n_class=self.n_class)
            pt = input.sigmoid()
            BCE = nn.BCELoss(reduction='none')(pt, target_one_hot)
            BCE[target.expand(BCE.shape) >= self.n_class] = 0
            # print(BCE)
            loss = torch.abs(target_one_hot - pt.detach()) ** self.gamma * BCE
            if self.use_alpha:
                loss = loss * self.alpha[self.n_class].to(loss.device)

        elif self.mode == 'ce':
            if input.dim() > 2:
                input = input.transpose(1, 2).transpose(2, 3).reshape(-1, self.n_class)
                target = target.transpose(1, 2).transpose(2, 3).reshape(-1, 1)

            pt = input.softmax(dim=1)
            pt = pt.detach()
            CE = nn.CrossEntropyLoss(reduction='none', ignore_index=255)(input, target.view(-1))
            target = torch.where(target < self.n_class, target, torch.zeros(target.shape, device=target.device).long())

            pt = pt.gather(dim=1, index=target).view(-1)
            # print(f'pt:{pt}')

            # print(f'CE:{CE.shape}\nCE:{CE}')
            loss = (1 - pt) ** self.gamma * CE
            if self.use_alpha:
                alpha = self.alpha[self.n_class].to(loss.device).gather(dim=0, index=target.squeeze())
                # print(alpha,target.squeeze())
                loss = loss * alpha
        else:
            raise Exception(f'*** focal loss mode:{self.mode} wrong!')

        if self.mean:
            return loss.mean()
        else:
            return loss.sum()

    @staticmethod
    def BCE(pt, target):

        BCE = -1 * (target * pt.log() +
                    (torch.Tensor([1.0]).to(target.device) - target)
                    * (torch.Tensor([1.0]).to(pt.device) - pt).log())
        return BCE

    def test(self):
        x = torch.randn((4, 21, 224, 224)).float()
        y = torch.randint(0, 21, (4, 1, 224, 224)).long()
        loss = self.forward(x, y)
        print(loss.item())


class CrossEntropyLoss2D(nn.Module):
    def __init__(self, n_calss=21, ignore_index=255, reduction='mean', weight=None):
        super().__init__()
        self.n_class = n_calss
        self.CELoss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if input.dim() > 2:
            input = input.transpose(1, 2).transpose(2, 3).reshape(-1, self.n_class)
            target = target.transpose(1, 2).transpose(2, 3).reshape(-1, 1)

        CE = self.CELoss(input, target.view(-1))
        return CE


class BCE2D:
    def __init__(self, n_calss=21, ignore_index=255):
        self.n_class = n_calss
        self.BCELoss = nn.BCELoss(reduction='none')

    def __call__(self, input: torch.Tensor, target: torch.Tensor):
        target_one_hot = label_to_one_hot(target, n_class=self.n_class)
        pt = input.sigmoid()
        BCE = self.BCELoss(pt, target_one_hot)
        BCE[target.expand(BCE.shape) >= self.n_class] = 0
        # print(BCE)
        return BCE.mean()


class LovaszBCE(nn.Module):
    def __init__(self, per_image=True, ignore_label=255):
        super().__init__()
        self.ignore = ignore_label
        self.per_image = per_image

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Binary Lovasz hinge loss
          logits: [B, cls, H, W] Variable, logits at each pixel (between -\infty and +\infty)
          labels: [B, 1, H, W] Tensor, binary ground truth masks (0 or 1)
          per_image: compute the loss per image instead of per batch
          ignore: void class id
        """

        assert input.dim() == 4
        batch_size = input.size(0)
        n_class = input.size(1)
        input = input.tanh()
        target_one_hot = label_to_one_hot(target, n_class, with_255=True)
        loss = 0
        for b in range(batch_size):
            # when:
            # 1. mask = 1, and pre_logit > 1, loss will be zero
            # 2. mask = 0, and pre_logit < -1, loss will be zero
            loss += lovasz_hinge(logits=input[b], labels=target_one_hot[b],
                                 per_image=self.per_image, ignore=self.ignore)
        return loss / batch_size


class LovaszBCEWithBCE(nn.Module):
    def __init__(self, n_class, ignore_label=255, alpha=1):
        super().__init__()
        self.alpha = alpha
        self.l = LovaszBCE(ignore_label=ignore_label)
        self.bce = BCE2D(n_class, ignore_index=ignore_label)

    def forward(self, input, target):
        """
        :param input: [bs, c, h, w],
        :param target: [bs, 1, h, w]
        :return: tensor
        """
        lovaszloss = self.alpha * self.l(input, target)
        bceloss = self.bce(input, target)
        return lovaszloss + bceloss


def get_focal_loss_alpha():
    fld = {0: 0.9490,
           1: 0.9171,
           2: 0.4488,
           3: 0.8890,
           4: 0.7430,
           5: 0.8481,
           6: 0.9464,
           7: 0.8730,
           8: 0.9331,
           9: 0.4245,
           10: 0.9187,
           11: 0.5813,
           12: 0.8854,
           13: 0.8944,
           14: 0.8529,
           15: 0.8689,
           16: 0.5915,
           17: 0.9015,
           18: 0.5804,
           19: 0.8786,
           20: 0.7795}

    x = torch.Tensor([fld[i] for i in fld])
    x = -x.log()
    # print(x)
    x = x.softmax(dim=0)
    # print(x)
    res = [0.0377, 0.0390, 0.0796, 0.0402, 0.0481, 0.0421, 0.0378, 0.0409, 0.0383,
           0.0842, 0.0389, 0.0615, 0.0404, 0.0400, 0.0419, 0.0411, 0.0604, 0.0397,
           0.0616, 0.0407, 0.0459]
    return x * 21


def _test_focal_loss_alpha():
    x = torch.ones(2, 21, 4, 4)
    print(x)
    alpha = get_focal_loss_alpha().unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
    print(alpha.shape)
    x = x * alpha
    print(x)
    y = torch.randint(0, 21, (2, 1, 4, 4))
    aflce = FocalLoss(mode='ce', use_alpha=True)
    loss = aflce(input=x, target=y)
    print(f'loss:{loss.item()}')


def test_focal_loss_BCE():
    x = torch.Tensor([0.9, 0.1, 0])
    y = torch.Tensor([0, 1, 0])
    loss = nn.BCELoss(reduction='none')(x.sigmoid(), y)
    focal_loss = FocalLoss(mode='bce')(x, y)
    print(loss, focal_loss)
    # assert loss == focal_loss


def _test_label_to_one_hot():
    # [1, 1, 2, 3]
    t = torch.Tensor([[[[1, 0, 255],
                        [0, 1, 255]]]]).long()
    print('target:', t, 'shape:', t.shape)
    # [1, 1, 2, 3]
    y = label_to_one_hot(t, 2)
    print(y, y.shape)
    model_out = torch.tensor([[[[0., 1000., -1000.],
                                [1000., 0., 0.]],

                               [[1000., 0., 0.],
                                [0., 1000., 0.]]]])
    criterion = FocalLoss(mode='bce', n_class=2)
    loss = criterion(input=model_out, target=t)
    print(loss.item())


def _test_focal_loss_ce():
    t = torch.Tensor([[[[1, 0, 255],
                        [0, 1, 255]]]]).long()
    criterion = FocalLoss(mode='ce', n_class=2)
    model_out = torch.tensor([[[[0., 1000., -1000.],
                                [1000., 0., 0.]],

                               [[1000., 0., 0.],
                                [0., 1000., 0.]]]])
    focal_loss = criterion(input=model_out, target=t)
    print(focal_loss.item())


def _test_ce2D():
    t = torch.Tensor([[[[1, 0, 255],
                        [0, 1, 255]]]]).long()
    criterion = CrossEntropyLoss2D(n_calss=2)
    model_out = torch.tensor([[[[0., 10., -1000.],
                                [1000., 0., 0.]],

                               [[1000., 0., 0.],
                                [0., 1000., 0.]]]])
    loss = criterion(input=model_out, target=t)
    print(loss.item())


def _test_bce2D():
    t = torch.Tensor([[[[1, 0, 255],
                        [0, 1, 255]]]]).long()
    criterion = BCE2D(n_calss=2)
    model_out = torch.tensor([[[[0., 100., -1000.],
                                [1000., 0., 0.]],

                               [[1000., 0., 0.],
                                [0., 1000., 0.]]]])
    loss = criterion(input=model_out, target=t)
    print(loss.item())
    pass


def test():
    bce = FocalLoss(mode='bce')
    x = torch.randn(4, 21, 2, 2)
    y = torch.randint(0, 255, (4, 1, 2, 2))
    loss = bce(x, y)
    print(loss.item())


if __name__ == '__main__':
    # _test_label_to_one_hot()
    # _test_focal_loss_ce()
    # _test_ce2D()
    # _test_bce2D()
    _test_focal_loss_alpha()
    pass
