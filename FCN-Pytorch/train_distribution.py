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
from train_utils import config, get_device, get_optimizer, get_lr, update_lr, poly_adjust_lr
from eval import validation
from criterion import get_criterion
from dataset.sbd_dataset import SBDDataLoaderCreator
from dataset.voc_dataset import VOCDataLoaderCreator
from criterion import label_to_one_hot


def model_accelerate(args, model):
    if torch.cuda.device_count() > 0:
        model = torch.nn.parallel.DistributedDataParallel(model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        device = get_device(args)
        model = model.to(device)
        print(f'*** {model.__class__} to GPUs, syncbatch OK.')
    return model


# train function (forward, backward, update)
def train_one_epoch(args, model, optimizer, train_loader, logger, model_saver):
    # use focal loss or not
    criterion = get_criterion(args)
    model_accelerate(args, model)

    model.train()
    # model.eval()
    device = get_device(args)
    for step, (imgs, targets) in enumerate(train_loader, start=1):
        iter = logger.size(key='train_loss')

        # poly scheduler
        if args.scheduler == 'poly':
            poly_adjust_lr(optimizer, lr=args.lr if args.lr_update is None else args.lr_update,
                           itr=iter, max_itr=args.max_iter)
        t1 = time.perf_counter()
        optimizer.zero_grad()
        # print(f'imgs:{imgs}\ntargets:{targets}')
        imgs, targets = imgs.to(device), targets.to(device)

        # Upsampling training
        try:
            model(None, target_one_hot=label_to_one_hot(targets, args.n_class), optimizer=optimizer)
        except Exception:
            pass

        outs = model(imgs)
        loss = 0

        # multi loss training
        if isinstance(outs, tuple) or isinstance(outs, list):
            if 'enc' in args.criterion.lower():
                loss = criterion(outs, targets)
            else:
                for i in outs:
                    # print('multi')
                    loss += criterion(input=i, target=targets)
        else:
            loss = criterion(input=outs, target=targets)

        loss = loss.mean()
        loss.backward()

        if isinstance(optimizer, nn.parallel.DistributedDataParallel):
            optimizer.module.step()
        else:
            optimizer.step()

        t2 = time.perf_counter()
        print(f'{args.model_name} | step:{step} [{step}/{len(train_loader)}] [{iter}/{args.max_iter}] '
              f'| {len(outs)}loss:{loss.item():.8f} | lr:{get_lr(optimizer):.12f} | time:{t2 - t1:.4f}')
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
    model_saver = ModelSaver(save_path=args.save, name_list=[args.optimizer, args.model_name, args.scheduler_name])

    # get model and accelerate the model training
    model = get_model(name=args.model_name, n_class=args.n_class)
    model = model_accelerate(args, model)
    model_saver.load(name=args.model_name, model=model)
    print(f'*** model is nn.parallel.DistributedDataParallel:{isinstance(model, nn.parallel.DistributedDataParallel)}')

    # get optimizer
    optimizer = get_optimizer(args, name=args.optimizer, model=model)
    optimizer = model_accelerate(args, optimizer)
    model_saver.load(name=args.optimizer, model=optimizer)

    if isinstance(optimizer, nn.parallel.DistributedDataParallel):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer.module,
                                                         milestones=args.schedule, gamma=args.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                         milestones=args.schedule, gamma=args.gamma)
    model_saver.load(name=args.scheduler_name, model=scheduler)

    try:
        print('===> lr:{}'.format(optimizer.param_groups[0]['lr']))
        print('===> scheduler state dict:{}'.format(scheduler.state_dict()))
    except Exception:
        pass

    if args.lr_update is not None:
        update_lr(optimizer, lr=args.lr_update)
        if isinstance(optimizer, nn.parallel.DistributedDataParallel):
            print('===> lr now:{}'.format(optimizer.module.param_groups[0]['lr']))
        else:
            print('===> lr now:{}'.format(optimizer.param_groups[0]['lr']))

        try:
            if isinstance(optimizer, nn.parallel.DistributedDataParallel):
                print('===> lr now:{}'.format(optimizer.module.param_groups[1]['lr']))
            else:
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
        epoch_now = scheduler.state_dict()['last_epoch']
        print(f'*** Epoch now:{epoch_now}.')

        if epoch_now >= args.epochs:
            print('*** Training finished!')

        if args.eval < 1:
            train_one_epoch(args, model, optimizer, sbd_train_loader, logger, model_saver)
            train_one_epoch(args, model, optimizer, voc2012_train_loader, logger, model_saver)
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


def train_cityscape():
    pass


if __name__ == "__main__":
    train_voc()
