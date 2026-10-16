# -*- coding: utf-8 -*-
# This training setting producing the best experiment result
# By chenlinwei

from __future__ import print_function

import sys
import os.path as osp

sys.path.append(osp.join(sys.path[0], '../'))
sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import time
import argparse
from logger import Logger, ModelSaver
from weakly_semi_supseg.weakly_semi_model import get_anc_model, get_self_learning_model
from torchvision import transforms
from criterion import FocalLoss
from train_utils import config, get_device, model_accelerate, get_optimizer, get_lr, update_lr, poly_adjust_lr
from weakly_semi_supseg.eval_anc import validation
from criterion import get_criterion
from dataset.sbd_dataset import SBDDataLoaderCreator
from dataset.weaklysemidataset.voc_bbox_seg_dataset import get_voc_data_loader
from dataset.weaklysemidataset.sbd_bbox_seg_dataset import get_sbd_data_loader
from criterion import label_to_one_hot
import gc

from weakly_semi_supseg.train_self_learning import train_one_epoch


# DIS: [ICCV17]Deep Dual Learning for Semantic Image Segmentation
def train_DIS(args, model, optimizer, train_loader, logger, model_saver, iter=0, max_iter=None):
    device = get_device(args)
    for step, (imgs, targets, bbox_targets) in enumerate(train_loader, start=1):
        if args.scheduler == 'poly' and max_iter is not None:
            poly_adjust_lr(optimizer, lr=args.lr if args.lr_update is None else args.lr_update,
                           itr=iter + step, max_itr=max_iter)

        imgs, targets, bbox_targets = imgs.to(device), targets.to(device), bbox_targets.to(device)

        optimizer.zero_grad()
        t1 = time.perf_counter()
        model.info_eval() if not isinstance(model, nn.DataParallel) else model.module.info_eval()
        if isinstance(model, nn.DataParallel):
            # def learn_reconstruct(self, x, bbox_target_one_hot, optimizer):
            loss_info = model.module.learn_reconstruct(imgs, bbox_targets, optimizer)
        else:
            loss_info = model.learn_reconstruct(imgs, bbox_targets, optimizer)
        loss_info.backward()
        optimizer.step()
        loss_info = loss_info.item()

        optimizer.zero_grad()
        model.normal_eval() if not isinstance(model, nn.DataParallel) else model.module.normal_eval()
        if isinstance(model, nn.DataParallel):
            # def learn_reconstruct(self, x, bbox_target_one_hot, optimizer):
            loss_normal = model.module.learn_reconstruct(imgs, bbox_targets, optimizer)
        else:
            loss_normal = model.learn_reconstruct(imgs, bbox_targets, optimizer)
        loss_normal.backward()
        optimizer.step()
        loss_normal = loss_normal.item()
        t2 = time.perf_counter()
        print(f'{args.model_name} | step:{step} [{step}/{len(train_loader)}] [{iter+step}/{max_iter}] '
              f'| {len(imgs)} loss_info:{loss_info:.8f} | loss_normal:{loss_normal:.8f} '
              f'| lr:{get_lr(optimizer):.12f} | time:{t2 - t1:.4f}')

    model_saver.save(name=args.model_name, model=model)
    model_saver.save(name=args.optimizer, model=optimizer)


def train_voc():
    torch.cuda.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    # init the tools
    args = config()

    json_name = args.optimizer if args.test_dataset is None else f'{args.optimizer}_VOC{args.test_dataset}'
    logger = Logger(save_path=args.save, json_name=json_name)
    model_saver = ModelSaver(save_path=args.save, strict=False,
                             name_list=[args.optimizer, args.model_name, args.scheduler_name])

    # get model and accelerate the model training
    model = get_self_learning_model(name=args.model_name, n_class=args.n_class)
    model = model_accelerate(args, model)
    model_saver.load(name=args.model_name, model=model)
    print(f'*** model is nn.DataParallel:{isinstance(model, nn.DataParallel)}')

    # get optimizer
    optimizer = get_optimizer(args, name=args.optimizer, model=model)
    # if torch.cuda.device_count() > 1:
    #     optimizer = nn.DataParallel(optimizer)

    model_saver.load(name=args.optimizer, model=optimizer)

    if isinstance(optimizer, nn.DataParallel):
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
        if isinstance(optimizer, nn.DataParallel):
            print('===> lr now:{}'.format(optimizer.module.param_groups[0]['lr']))
        else:
            print('===> lr now:{}'.format(optimizer.param_groups[0]['lr']))

        try:
            if isinstance(optimizer, nn.DataParallel):
                print('===> lr now:{}'.format(optimizer.module.param_groups[1]['lr']))
            else:
                print('===> lr now:{}'.format(optimizer.param_groups[1]['lr']))
        except Exception:
            pass
    # To solve the problem: received 0 items of ancdata
    torch.multiprocessing.set_sharing_strategy('file_system')

    max_epoch = 10
    for epoch in range(max_epoch):
        # get dataset
        voc2012_train_loader = get_voc_data_loader(args, sub_mean=args.std > 0, stdize=args.std > 1, train=True,
                                                   aug=args.voc_aug, duplicate=args.voc_duplicate)
        sbd_train_loader = get_sbd_data_loader(args, sub_mean=args.std > 0, stdize=args.std > 1, aug=args.sbd_aug)

        iter = epoch*len(voc2012_train_loader) + epoch * len(sbd_train_loader)
        max_iter = max_epoch*len(voc2012_train_loader) + max_epoch * len(sbd_train_loader)

        train_DIS(args, model, optimizer, voc2012_train_loader, logger, model_saver, iter, max_iter)
        train_DIS(args, model, optimizer, sbd_train_loader, logger, model_saver, iter, max_iter)


    # Main loop
    for _ in range(args.epochs):
        # get dataset
        voc2012_train_loader = get_voc_data_loader(args, sub_mean=args.std > 0, stdize=args.std > 1, train=True,
                                                   aug=args.voc_aug, duplicate=args.voc_duplicate)
        sbd_train_loader = get_sbd_data_loader(args, sub_mean=args.std > 0, stdize=args.std > 1, aug=args.sbd_aug)
        args.max_iter = ((len(voc2012_train_loader) if 'voc' in args.train_voc_dataset else 0) +
                         (len(sbd_train_loader) if 'sbd' in args.train_voc_dataset else 0)) * args.epochs

        # val(args, model, scheduler, val_loader, logger, model_saver)
        val_loader = get_voc_data_loader(args, sub_mean=args.std > 0, stdize=args.std > 1, train=False, aug=args.aug)

        scheduler.step()
        epoch_now = scheduler.state_dict()['last_epoch']
        print(f'*** Epoch now:{epoch_now}.')

        if epoch_now >= args.epochs:
            print('*** Training finished!')

        def val(args, model, val_loader, model_saver, logger):
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

        train_loss_1, train_loss_2 = 0, 0
        if 'voc' in args.train_voc_dataset:
            # train_loss_1 = train_one_epoch(args, model, optimizer, voc2012_train_loader, logger, model_saver,
            #                                no_mask=False)
            # train_DIS(args, model, optimizer, voc2012_train_loader, logger, model_saver)
            if epoch_now % args.val_epochs == 0:
                model.normal_eval() if not isinstance(model, nn.DataParallel) else model.module.normal_eval()
                val(args, model, val_loader, model_saver, logger)

        if 'sbd' in args.train_voc_dataset:
            train_loss_2 = train_one_epoch(args, model, optimizer, sbd_train_loader, logger, model_saver,
                                           no_mask=True)
            train_DIS(args, model, optimizer, sbd_train_loader, logger, model_saver)

            if epoch_now % args.val_epochs == 0:
                model.normal_eval() if not isinstance(model, nn.DataParallel) else model.module.normal_eval()
                val(args, model, val_loader, model_saver, logger)

        if 'sbd' not in args.train_voc_dataset and 'voc' not in args.train_voc_dataset:
            model.normal_eval() if not isinstance(model, nn.DataParallel) else model.module.normal_eval()
            val(args, model, val_loader, model_saver, logger)

        logger.log(key='TrainLoss', data=train_loss_1 + train_loss_2, show=True)
        # model_saver.save(name=args.model_name, model=model)
        # model_saver.save(name=args.optimizer, model=optimizer)
        # model_saver.save(name=args.scheduler_name, model=scheduler)
        logger.save_log()
        logger.visualize()

        del val_loader
        gc.collect()


def train_cityscape():
    pass


if __name__ == "__main__":
    train_voc()
