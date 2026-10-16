# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import time
import argparse
from logger import Logger, ModelSaver
from model import get_model
from torchvision import transforms
from dataset.sbd_dataset import get_sbd_data_loader
from criterion import FocalLoss
from train_utils import config, get_device, model_accelerate, get_optimizer, get_lr, update_lr, poly_adjust_lr
from eval import validation
from criterion import get_criterion
from dataset.sbd_dataset import SBDDataLoaderCreator
from dataset.voc_dataset import VOCDataLoaderCreator
import torch.nn.functional as F


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_global_pooling=False, use_sigmoid=False, use_bias=False):
        super(NLayerDiscriminator, self).__init__()
        kw = 3
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        if use_global_pooling:
            sequence += [nn.AdaptiveAvgPool2d(1)]
            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=1, stride=1, padding=0)]
            pass
        else:
            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
            pass
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


def _test_discriminator():
    x = torch.randn(4, 3, 32, 32)
    D_ma = NLayerDiscriminator(input_nc=3, use_global_pooling=True)
    D_mi = NLayerDiscriminator(input_nc=3, use_global_pooling=False)
    y_ma = D_ma(x)
    y_mi = D_mi(x)
    print(y_ma.shape, y_mi.shape)


class Discriminator(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.D_Ma = NLayerDiscriminator(input_nc=n_class + 3)
        self.D_Mi = NLayerDiscriminator(input_nc=n_class + 3)

    def forward(self, lr, hr):
        # return self.D_Ma(lr).sigmoid(), self.D_Mi(hr).sigmoid()
        return self.D_Ma(lr), self.D_Mi(hr)


def get_D(args):
    D = Discriminator(n_class=args.n_class)
    return D


def label_to_one_hot(targets: torch.Tensor, n_class: int):
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
    one_hot[targets_ignore] = 0
    # print(one_hot[targets_ignore])
    one_hot = one_hot.transpose(3, 2)
    one_hot = one_hot.transpose(2, 1)
    # print(one_hot.size())
    return one_hot.float()


# train function (forward, backward, update)
def train_one_epoch(args, model, optimizer, D, D_optimizer, train_loader, logger, model_saver):
    # use focal loss or not
    criterion = get_criterion(args)
    D_criterion = nn.MSELoss()
    # D_criterion = nn.BCELoss()
    model.train()
    # model.eval()
    device = get_device(args)
    for step, (imgs, targets) in enumerate(train_loader, start=1):
        iter = logger.size(key='train_loss')
        if args.scheduler == 'poly':
            poly_adjust_lr(optimizer, lr=args.lr if args.lr_update is None else args.lr_update,
                           itr=iter, max_itr=args.max_iter)
            poly_adjust_lr(D_optimizer, lr=args.lr if args.lr_update is None else args.lr_update,
                           itr=iter, max_itr=args.max_iter)
        t1 = time.perf_counter()
        optimizer.zero_grad()
        D_optimizer.zero_grad()

        # optimize D
        imgs, targets = imgs.to(device), targets.to(device)
        lr_outputs, hr_outputs = model(imgs)

        lr_imgs = F.interpolate(imgs, size=lr_outputs.shape[2:], mode='bilinear')
        lr_targets = F.interpolate(targets.float(), size=lr_outputs.shape[2:]).long()

        targets_one_hot = label_to_one_hot(targets, args.n_class)
        lr_targets_one_hot = label_to_one_hot(lr_targets, args.n_class)

        D_ma_false, D_mi_false = D(torch.cat([lr_outputs.detach(), lr_imgs], dim=1),
                                   torch.cat([hr_outputs.detach(), imgs], dim=1))
        D_ma_true, D_mi_true = D(torch.cat([lr_targets_one_hot, lr_imgs], dim=1),
                                 torch.cat([targets_one_hot, imgs], dim=1))

        D_ma_false_loss = D_criterion(input=D_ma_false,
                                      target=torch.zeros(D_ma_false.shape, device=D_ma_false.device))
        D_mi_false_loss = D_criterion(input=D_mi_false,
                                      target=torch.zeros(D_mi_false.shape, device=D_mi_false.device))

        D_ma_true_loss = D_criterion(input=D_ma_true, target=torch.ones(D_ma_true.shape, device=D_ma_true.device))
        D_mi_true_loss = D_criterion(input=D_mi_true, target=torch.ones(D_mi_true.shape, device=D_mi_true.device))

        D_loss = (D_ma_false_loss + D_mi_false_loss + D_ma_true_loss + D_mi_true_loss).mean()
        D_loss.backward()
        D_optimizer.step()

        # optimize model
        lr_loss = criterion(input=lr_outputs, target=lr_targets)
        hr_loss = criterion(input=hr_outputs, target=targets)
        D_ma_false, D_mi_false = D(torch.cat([lr_outputs, lr_imgs], dim=1),
                                   torch.cat([hr_outputs, imgs], dim=1))
        D_ma_false_loss = D_criterion(input=D_ma_false, target=torch.ones(D_ma_false.shape, device=D_ma_false.device))
        D_mi_false_loss = D_criterion(input=D_mi_false, target=torch.ones(D_mi_false.shape, device=D_mi_false.device))
        loss = D_ma_false_loss + 25 * lr_loss + D_mi_false_loss + 100 * hr_loss
        loss.backward()
        optimizer.step()
        t2 = time.perf_counter()
        print(f'{args.model_name} | step:{step} [{step}/{len(train_loader)}] [{iter}/{args.max_iter}] '
              f'| D_loss:{D_loss.item():.8f} | loss:{loss.item():.8f} | lr:{get_lr(optimizer):.12f} | time:{t2 - t1:.4f}')
        logger.log(key='train_loss', data=loss.item())

        # save the model, optimizer every args.save_steps
        if step % args.save_steps == 0:
            logger.visualize(key='train_loss', range=(-1000, -1))
            logger.save_log()
            model_saver.save(name=args.model_name, model=model)
            model_saver.save(name=args.optimizer, model=optimizer)
            model_saver.save(name='D', model=D)
            model_saver.save(name='D_'+args.optimizer, model=D_optimizer)
            # break


def train_voc():
    torch.cuda.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    # init the tools
    args = config()
    args.model_name = 'deeplabv3plusad'

    json_name = args.optimizer if args.test_dataset is None else f'{args.optimizer}_VOC{args.test_dataset}'
    logger = Logger(save_path=args.save, json_name=json_name)
    model_saver = ModelSaver(save_path=args.save,
                             name_list=[args.optimizer, args.model_name, args.scheduler_name,
                                        'D', 'D_' + args.optimizer, 'D_' + args.scheduler_name])

    # get model and accelerate the model training
    model = get_model(name=args.model_name, n_class=args.n_class)
    model = model_accelerate(args, model)
    model_saver.load(name=args.model_name, model=model)
    print(f'*** model is nn.DataParallel:{isinstance(model, nn.DataParallel)}')

    D = get_D(args)
    D = model_accelerate(args, D)
    model_saver.load(name='D', model=D)

    # get optimizer
    optimizer = get_optimizer(args, name=args.optimizer, model=model)
    model_saver.load(name=args.optimizer, model=optimizer)

    D_optimizer = get_optimizer(args, name=args.optimizer, model=D)
    model_saver.load(name='D_' + args.optimizer, model=optimizer)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.schedule, gamma=args.gamma)
    model_saver.load(name=args.scheduler_name, model=scheduler)

    D_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.schedule, gamma=args.gamma)
    model_saver.load(name='D_' + args.scheduler_name, model=scheduler)

    print('===> lr:{}'.format(optimizer.param_groups[0]['lr']))
    print('===> scheduler state dict:{}'.format(scheduler.state_dict()))
    if args.lr_update is not None:
        update_lr(optimizer, lr=args.lr_update)
        print('===> lr now:{}'.format(optimizer.param_groups[0]['lr']))
        try:
            print('===> lr now:{}'.format(optimizer.param_groups[1]['lr']))
        except Exception:
            pass

    voc2012_train_loader_creater = VOCDataLoaderCreator(args, sub_mean=args.std > 0, stdize=args.std > 1,
                                                        train=True, aug=args.aug, RAM=args.train_to_ram)
    sbd_train_loader_creater = SBDDataLoaderCreator(args, sub_mean=args.std > 0, stdize=args.std > 1,
                                                    aug=args.aug, RAM=args.train_to_ram)

    if args.test_dataset == '2012' or args.test_dataset is None:
        val_loader_creater = VOCDataLoaderCreator(args, sub_mean=args.std > 0, stdize=args.std > 1,
                                                  year='2012', train=False, RAM=bool(args.val_to_ram))
    elif args.test_dataset == '2011':
        val_loader_creater = VOCDataLoaderCreator(args, sub_mean=args.std > 0, stdize=args.std > 1,
                                                  year='2011', train=False, RAM=bool(args.val_to_ram))
    else:
        raise Exception(f'*** args.test_dataset year {args.test_dataset} wrong!')

    # Main loop
    for _ in range(args.epochs):
        # get dataset
        sbd_train_loader = sbd_train_loader_creater.get_data_loader()
        voc2012_train_loader = voc2012_train_loader_creater.get_data_loader()
        args.max_iter = (len(sbd_train_loader) + len(voc2012_train_loader)) * args.epochs
        # val(args, model, scheduler, val_loader, logger, model_saver)
        val_loader = val_loader_creater.get_data_loader()

        scheduler.step()
        D_scheduler.step()
        epoch_now = scheduler.state_dict()['last_epoch']
        print(f'*** Epoch now:{epoch_now}.')

        if epoch_now >= args.epochs:
            print('*** Training finished!')
        train_one_epoch(args, model, optimizer, D, D_optimizer, sbd_train_loader, logger, model_saver)
        train_one_epoch(args, model, optimizer, D, D_optimizer, voc2012_train_loader, logger, model_saver)
        acc, acc_cls, mean_iu, fwavacc, iu = validation(args, model, val_loader)
        if logger.get_max(key='MeanIoU') < mean_iu or logger.get_max(key='MeanPixel') < acc:
            model_saver.save(name=args.model_name + f'_miou_{mean_iu:.4f}_pa_{acc:.4f}', model=model)
            model_saver.save(name='D_' + f'_miou_{mean_iu:.4f}_pa_{acc:.4f}', model=D)

        logger.log(key='MeanIoU', data=mean_iu, show=True)
        logger.log(key='FWIoU', data=fwavacc, show=True)
        logger.log(key='IoUs', data=iu, show=True)
        logger.log(key='MeanPixel', data=acc, show=True)
        logger.log(key='ClassPixel', data=acc_cls, show=True)

        model_saver.save(name=args.model_name, model=model)
        model_saver.save(name=args.optimizer, model=optimizer)
        model_saver.save(name=args.scheduler_name, model=scheduler)
        model_saver.save(name='D', model=D)
        model_saver.save(name='D_' + args.optimizer, model=D_optimizer)
        model_saver.save(name='D_' + args.scheduler_name, model='D_' + args.scheduler)
        logger.save_log()
        logger.visualize()


def train_cityscape():
    pass


if __name__ == "__main__":
    train_voc()
