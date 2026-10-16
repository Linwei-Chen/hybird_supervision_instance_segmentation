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
from dataset.cityscapes import get_cityscape_data_loader
import torch.nn.functional as F

"""
change from train_cityscapes.py, which only modify the train strategy：
1. first half of all iterations train the encoder as pre-training process, the target resize into 1/8
2. another half of iteration train the erfnet as normal
"""


# train function (forward, backward, update)
def train_one_epoch(args, model, optimizer, train_loader, logger, model_saver):
    # use focal loss or not
    criterion = get_criterion(args)
    if torch.cuda.device_count() > 1:
        criterion = nn.DataParallel(criterion)
    model.train()
    # model.eval()
    device = get_device(args)
    for step, (imgs, targets) in enumerate(train_loader, start=1):
        iter = logger.size(key='train_loss')
        if args.scheduler == 'poly':
            poly_adjust_lr(optimizer, lr=args.lr if args.lr_update is None else args.lr_update,
                           itr=iter, max_itr=args.max_iter)
        t1 = time.perf_counter()

        optimizer.zero_grad()
        imgs, targets = imgs.to(device), targets.to(device)
        if iter < args.max_iter // 2:
            outs = model(imgs, only_encode=True)
            h, w = outs.shape[2:]
            targets = F.interpolate(targets.float(), size=(h, w), mode='nearest').long()
            loss = criterion(input=outs, target=targets)
        else:
            outs = model(imgs, only_encode=False)
            loss = criterion(input=outs, target=targets)
        # outs = model(imgs)
        # loss = criterion(input=outs, target=targets)

        loss = loss.mean()
        loss.backward()
        optimizer.step()

        t2 = time.perf_counter()
        print(f'{args.model_name} | step:{step} [{step}/{len(train_loader)}] [{iter}/{args.max_iter}] '
              f'| loss:{loss.item():.8f} | lr:{get_lr(optimizer):.12f} | time:{t2 - t1:.4f}')
        logger.log(key='train_loss', data=loss.item())

        # save the model, optimizer every args.save_steps
        if step % args.save_steps == 0:
            logger.visualize(key='train_loss', range=(-1000, -1))
            logger.save_log()
            model_saver.save(name=args.model_name, model=model)
            model_saver.save(name=args.optimizer, model=optimizer)
            # break


def train_cityscapes():
    torch.cuda.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    # init the tools
    args = config()
    args.model_name = 'erfnet'
    args.n_class = 19
    assert args.n_class == 19
    assert args.model_name == 'erfnet'

    json_name = args.optimizer
    logger = Logger(save_path=args.save, json_name=json_name)
    model_saver = ModelSaver(save_path=args.save, name_list=[args.optimizer, args.model_name, args.scheduler_name])

    # get model and accelerate the model training
    model = get_model(name=args.model_name, n_class=args.n_class)
    model = model_accelerate(args, model)
    model_saver.load(name=args.model_name, model=model)
    print(f'*** model is nn.DataParallel:{isinstance(model, nn.DataParallel)}')

    # get optimizer
    optimizer = get_optimizer(args, name=args.optimizer, model=model)
    model_saver.load(name=args.optimizer, model=optimizer)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.schedule, gamma=args.gamma)
    model_saver.load(name=args.scheduler_name, model=scheduler)

    print('===> lr:{}'.format(optimizer.param_groups[0]['lr']))
    print('===> scheduler state dict:{}'.format(scheduler.state_dict()))
    if args.lr_update is not None:
        update_lr(optimizer, lr=args.lr_update)
        print('===> lr now:{}'.format(optimizer.param_groups[0]['lr']))
        try:
            print('===> lr now:{}'.format(optimizer.param_groups[1]['lr']))
        except Exception:
            pass

    # Main loop
    for _ in range(args.epochs):
        # get dataset
        train_loader = get_cityscape_data_loader(args, sub_mean=args.std > 0, stdize=args.std > 1,
                                                 split='train', aug=args.aug)
        args.max_iter = len(train_loader) * args.epochs
        # val(args, model, scheduler, val_loader, logger, model_saver)
        val_loader = get_cityscape_data_loader(args, sub_mean=args.std > 0, stdize=args.std > 1,
                                               split='val', aug=args.aug)

        scheduler.step()
        epoch_now = scheduler.state_dict()['last_epoch']
        print(f'*** Epoch now:{epoch_now}.')

        if epoch_now >= args.epochs:
            print('*** Training finished!')
        train_one_epoch(args, model, optimizer, train_loader, logger, model_saver)

        if epoch_now % args.val_epochs == 0 and epoch_now > args.epochs // 2:
            acc, acc_cls, mean_iu, fwavacc, iu = validation(args, model, val_loader)
            if logger.get_max(key='MeanIoU') < mean_iu or logger.get_max(key='MeanPixel') < acc:
                model_saver.save(name=args.model_name + f'_miou_{mean_iu:.4f}_pa_{acc:.4f}', model=model)

            logger.log(key='MeanIoU', data=mean_iu, show=True)
            logger.log(key='FWIoU', data=fwavacc, show=True)
            logger.log(key='IoUs', data=iu, show=True)
            logger.log(key='MeanPixel', data=acc, show=True)
            logger.log(key='ClassPixel', data=acc_cls, show=True)

        model_saver.save(name=args.model_name, model=model)
        model_saver.save(name=args.optimizer, model=optimizer)
        model_saver.save(name=args.scheduler_name, model=scheduler)
        logger.save_log()
        logger.visualize()


if __name__ == "__main__":
    train_cityscapes()
