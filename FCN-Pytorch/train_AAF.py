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
from criterion import get_criterion, label_to_one_hot
from dataset.sbd_dataset import SBDDataLoaderCreator
from dataset.voc_dataset import VOCDataLoaderCreator
from AffinityLoss import adaptive_affinity_loss, AdaptiveAffinityLoss


class AdaptiveAffinityField(nn.Module):
    def __init__(self, args, kld_margin=3, gamma=2):
        super().__init__()
        self.n_class = args.n_class
        self.kld_margin = kld_margin
        self.w_not_edge3 = nn.Parameter(torch.ones(1, 1, 1, args.n_class, 1), requires_grad=True)
        self.w_not_edge5 = nn.Parameter(torch.ones(1, 1, 1, args.n_class, 1), requires_grad=True)
        self.w_not_edge7 = nn.Parameter(torch.ones(1, 1, 1, args.n_class, 1), requires_grad=True)
        self.w_edge3 = nn.Parameter(torch.ones(1, 1, 1, args.n_class, 1), requires_grad=True)
        self.w_edge5 = nn.Parameter(torch.ones(1, 1, 1, args.n_class, 1), requires_grad=True)
        self.w_edge7 = nn.Parameter(torch.ones(1, 1, 1, args.n_class, 1), requires_grad=True)
        self.w_focal_loss = nn.Parameter(torch.ones(1, args.n_class, 1, 1), requires_grad=True)
        self._weights_init()
        self.focal_loss_gamma = gamma
        self.adaptive_affinity_loss = AdaptiveAffinityLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor, ):
        one_hot_targets_chw = label_to_one_hot(target, self.n_class)
        one_hot_targets = one_hot_targets_chw.transpose(1, 2).transpose(2, 3)

        bce_focal_loss = self.ce_focal_loss(input, target)

        input = input.transpose(1, 2).transpose(2, 3)
        target = target.squeeze(dim=1)
        input = input.softmax(dim=1)
        aaf3 = self.adaptive_affinity_loss(labels=target, one_hot_lab=one_hot_targets,
                                           probs=input, size=1, num_classes=self.n_class, kld_margin=self.kld_margin,
                                           w_edge=self.w_edge3.softmax(dim=-1),
                                           w_not_edge=self.w_not_edge3.softmax(dim=-1))
        aaf3 = torch.cat(aaf3, dim=0)
        # aaf3 = torch.cat(aaf3, dim=0)
        aaf5 = self.adaptive_affinity_loss(labels=target, one_hot_lab=one_hot_targets,
                                           probs=input, size=2, num_classes=self.n_class, kld_margin=self.kld_margin,
                                           w_edge=self.w_edge5.softmax(dim=-1),
                                           w_not_edge=self.w_not_edge5.softmax(dim=-1))
        aaf5 = aaf5[0].mean() + aaf5[0].mean()
        aaf7 = self.adaptive_affinity_loss(labels=target, one_hot_lab=one_hot_targets,
                                           probs=input, size=3, num_classes=self.n_class, kld_margin=self.kld_margin,
                                           w_edge=self.w_edge7.softmax(dim=-1),
                                           w_not_edge=self.w_not_edge7.softmax(dim=-1))

        aaf7 = aaf7[0].mean() + aaf7[0].mean()
        patch_loss = (aaf3 + aaf5 + aaf7).mean()
        # patch_loss = aaf3.mean()
        return (patch_loss + bce_focal_loss).mean()

    def ce_focal_loss(self, input, target: torch.Tensor):
        if input.dim() > 2:
            input = input.transpose(1, 2).transpose(2, 3).reshape(-1, self.n_class)
            target = target.transpose(1, 2).transpose(2, 3).reshape(-1, 1)
        # print(input.shape, target.shape)
        # print(f'input:{input}\ntarget:{target}')

        pt = input.softmax(dim=1)
        pt = pt.detach()
        CE = nn.CrossEntropyLoss(reduction='none', ignore_index=255)(input, target.view(-1))
        target = torch.where(target < self.n_class, target, torch.zeros(target.shape, device=target.device).long())

        pt = pt.gather(dim=1, index=target).view(-1)
        # print(f'pt:{pt}')

        # print(f'CE:{CE.shape}\nCE:{CE}')
        loss = (1 - pt) ** self.focal_loss_gamma * CE
        return loss.mean()

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Parameter):
                nn.init.constant_(m.data, 1)


def _test_aff():
    args = config()
    aaf = AdaptiveAffinityField(args)
    print(aaf)

    args = config()
    aaf = AdaptiveAffinityField(args)
    print(aaf)


# train function (forward, backward, update)
def train_one_epoch(args, model, optimizer, aaf, aaf_optimizer, train_loader, logger, model_saver):
    # use focal loss or not
    model.train()
    device = get_device(args)
    for step, (imgs, targets) in enumerate(train_loader, start=1):
        iter = logger.size(key='train_loss')
        if args.scheduler == 'poly':
            poly_adjust_lr(optimizer, lr=args.lr if args.lr_update is None else args.lr_update,
                           itr=iter, max_itr=args.max_iter)
            poly_adjust_lr(aaf_optimizer, lr=args.lr if args.lr_update is None else args.lr_update,
                           itr=iter, max_itr=args.max_iter)
        t1 = time.perf_counter()
        optimizer.zero_grad()
        aaf_optimizer.zero_grad()
        imgs, targets = imgs.to(device), targets.to(device)
        outs = model(imgs)
        # loss = criterion(input=outs, target=targets_one_hot)

        aaf_loss = -aaf(input=outs.detach(), target=targets)
        aaf_loss.backward()
        aaf_optimizer.step()

        print(aaf.state_dict())
        loss = aaf(input=outs, target=targets)

        loss.backward()
        optimizer.step()
        t2 = time.perf_counter()
        print(f'{args.model_name} | step:{step} [{step}/{len(train_loader)}] [{iter}/{args.max_iter}] '
              f'| aaf_loss:{aaf_loss.item():.8f} | loss:{loss.item():.8f} '
              f'| lr:{get_lr(optimizer):.12f} | time:{t2 - t1:.4f}')
        logger.log(key='train_loss', data=loss.item())

        # save the model, optimizer every args.save_steps
        if step % args.save_steps == 0:
            logger.visualize(key='train_loss', range=(-1000, -1))
            logger.save_log()
            model_saver.save(name=args.model_name, model=model)
            model_saver.save(name=args.optimizer, model=optimizer)
            # break


def train_voc():
    torch.cuda.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    # init the tools
    args = config()

    json_name = args.optimizer if args.test_dataset is None else f'{args.optimizer}_VOC{args.test_dataset}'
    logger = Logger(save_path=args.save, json_name=json_name)
    model_saver = ModelSaver(save_path=args.save,
                             name_list=[args.optimizer, args.model_name, args.scheduler_name,
                                        'aaf_' + args.optimizer, 'aaf', 'aaf_' + args.scheduler_name])

    # get model and accelerate the model training
    model = get_model(name=args.model_name, n_class=args.n_class)
    model = model_accelerate(args, model)
    model_saver.load(name=args.model_name, model=model)
    print(f'*** model is nn.DataParallel:{isinstance(model, nn.DataParallel)}')

    aaf = AdaptiveAffinityField(args)
    aaf = model_accelerate(args, aaf)
    model_saver.load(name='aaf', model=aaf)

    # get optimizer
    optimizer = get_optimizer(args, name=args.optimizer, model=model)
    model_saver.load(name=args.optimizer, model=optimizer)

    aaf_optimizer = get_optimizer(args, name=args.optimizer, model=aaf)
    model_saver.load(name='aaf_' + args.optimizer, model=aaf_optimizer)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.schedule, gamma=args.gamma)
    model_saver.load(name=args.scheduler_name, model=scheduler)

    aaf_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=aaf_optimizer,
                                                         milestones=args.schedule, gamma=args.gamma)
    model_saver.load(name='aaf_' + args.scheduler_name, model=aaf_scheduler)

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
        aaf_scheduler.step()

        epoch_now = scheduler.state_dict()['last_epoch']
        print(f'*** Epoch now:{epoch_now}.')

        if epoch_now >= args.epochs:
            print('*** Training finished!')
        train_one_epoch(args, model, optimizer, aaf, aaf_optimizer, sbd_train_loader, logger, model_saver)
        train_one_epoch(args, model, optimizer, aaf, aaf_optimizer, voc2012_train_loader, logger, model_saver)
        acc, acc_cls, mean_iu, fwavacc, iu = validation(args, model, val_loader)
        if logger.get_max(key='MeanIoU') < mean_iu or logger.get_max(key='MeanPixel') < acc:
            model_saver.save(name=args.model_name + f'_miou_{mean_iu:.4f}_pa_{acc:.4f}', model=model)
            model_saver.save(name='aaf' + f'_miou_{mean_iu:.4f}_pa_{acc:.4f}', model=aaf)

        logger.log(key='MeanIoU', data=mean_iu, show=True)
        logger.log(key='FWIoU', data=fwavacc, show=True)
        logger.log(key='IoUs', data=iu, show=True)
        logger.log(key='MeanPixel', data=acc, show=True)
        logger.log(key='ClassPixel', data=acc_cls, show=True)

        model_saver.save(name=args.model_name, model=model)
        model_saver.save(name=args.optimizer, model=optimizer)
        model_saver.save(name=args.scheduler_name, model=scheduler)

        model_saver.save(name='aaf', model=aaf)
        model_saver.save(name='aaf_ ' + args.optimizer, model=aaf_optimizer)
        model_saver.save(name='aaf_' + args.scheduler_name, model=aaf_scheduler)
        logger.save_log()
        logger.visualize()


if __name__ == "__main__":
    train_voc()
    # _test_aff()
