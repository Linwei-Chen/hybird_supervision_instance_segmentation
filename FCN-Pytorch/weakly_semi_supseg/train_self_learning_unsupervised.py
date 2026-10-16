# -*- coding: utf-8 -*-

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


# train function (forward, backward, update)
def train_one_epoch(args, model, optimizer, train_loader, logger, model_saver, no_mask=False):
    # use focal loss or not
    criterion = get_criterion(args)
    from criterion import CrossEntropyLoss2D, LovaszSoftMaxWithCE

    CEloss = CrossEntropyLoss2D(reduction='none')
    L_CEloss = LovaszSoftMaxWithCE(n_class=21, ignore_label=255, alpha=1)
    KLloss = nn.KLDivLoss()
    if torch.cuda.device_count() > 1:
        criterion = nn.DataParallel(criterion)
        CEloss = nn.DataParallel(CEloss)
        L_CEloss = nn.DataParallel(L_CEloss)
        KLloss = nn.DataParallel(KLloss)

    model.train()
    train_loss = 0
    device = get_device(args)
    for step, (imgs, targets, bbox_targets) in enumerate(train_loader, start=1):
        iter = logger.size(key='normal_loss') + logger.size(key='weakly_loss')

        # poly scheduler
        if args.scheduler == 'poly':
            poly_adjust_lr(optimizer, lr=args.lr if args.lr_update is None else args.lr_update,
                           itr=iter, max_itr=args.max_iter)
        t1 = time.perf_counter()
        # print(f'imgs:{imgs}\ntargets:{targets}')
        imgs, targets, bbox_targets = imgs.to(device), targets.to(device), bbox_targets.to(device)

        if no_mask:
            optimizer.zero_grad()
            with torch.no_grad():
                assert args.self_learning in ('refine', 'normal', 'kl')
                if args.self_learning == 'refine':
                    model.refine_learning() if not isinstance(model, nn.DataParallel) \
                        else model.module.refine_learning()
                else:
                    model.self_learning() if not isinstance(model, nn.DataParallel) else model.module.self_learning()
                outs_self_learning = model(imgs, bbox_targets).detach()
                probs, mask = outs_self_learning.max(dim=1, keepdim=True)

            model.normal() if not isinstance(model, nn.DataParallel) else model.module.normal()
            outs_normal = model(imgs, bbox_targets)
            if args.train_setting == 'pow':
                probs = probs.pow(2)
            elif args.train_setting == 'thres0.5':
                probs[probs < 0.5] = 0.
            elif args.train_setting == 'thres0.5+pow':
                probs[probs < 0.5] = 0.
                probs = probs.pow(2)
            elif args.train_setting == 'strong':
                probs = torch.ones(probs.shape).to(probs.device)
            elif args.train_setting == 'l+thres0.5':
                mask[probs < 0.5] = 255
            else:
                raise NotImplementedError
            if args.train_setting == 'l+thres0.5':
                loss0 = L_CEloss(input=outs_normal, target=mask)
            else:
                loss0 = probs.transpose(1, 2).transpose(2, 3).view(-1) * CEloss(input=outs_normal, target=mask)
            loss0 = loss0.unsqueeze(dim=0).mean()
            loss0.backward()
            optimizer.step()

            t2 = time.perf_counter()
            print(f'{args.model_name} | step:{step} [{step}/{len(train_loader)}] [{iter}/{args.max_iter}] '
                  f'| {len(outs_normal)} loss_anc:{loss0.item():.8f} | '
                  f'| lr:{get_lr(optimizer):.12f} | time:{t2 - t1:.4f}')

            logger.log(key='weakly_loss', data=loss0.item())
            train_loss += loss0.item()
            pass
        else:
            # Step 1: training with bboxes
            optimizer.zero_grad()
            model.info() if not isinstance(model, nn.DataParallel) else model.module.info()
            outs_bbox = model(imgs, bbox_targets)
            loss1 = criterion(input=outs_bbox, target=targets)
            loss1 = loss1.unsqueeze(dim=0).mean()
            loss1.backward()
            optimizer.step()

            # Step 2: normal training
            optimizer.zero_grad()
            model.normal() if not isinstance(model, nn.DataParallel) else model.module.normal()
            outs_normal = model(imgs, bbox_targets)
            loss2 = criterion(input=outs_normal, target=targets)
            loss2 = loss2.unsqueeze(dim=0).mean()
            loss2.backward()
            optimizer.step()

            # Step 3: pixel-wise training
            if loss2 > loss1 and args.self_learning in ('kl'):
                optimizer.zero_grad()
                with torch.no_grad():
                    model.info_eval() if not isinstance(model, nn.DataParallel) else model.module.info_eval()
                    outs_info = model(imgs, bbox_targets).detach()
                model.normal() if not isinstance(model, nn.DataParallel) else model.module.normal()
                outs_normal = model(imgs, bbox_targets)
                loss3 = KLloss(input=outs_normal.softmax(dim=1), target=outs_info.softmax(dim=1))
                loss3 = loss3.unsqueeze(dim=0).mean()
                loss3.backward()
                optimizer.step()
                print(f'KLloss:{loss3.item():.8f}')
                logger.log(key='KLloss', data=loss3.item())
            else:
                pass

            optimizer.zero_grad()
            model.learn_extra() if not isinstance(model, nn.DataParallel) else model.module.learn_extra()
            cls_loss = model(imgs, bbox_targets)
            cls_loss.mean().backward()
            optimizer.step()

            t2 = time.perf_counter()
            print(f'{args.model_name} | step:{step} [{step}/{len(train_loader)}] [{iter}/{args.max_iter}] '
                  f'| {len(outs_normal)} loss_bbox:{loss1.item():.8f} | loss_normal:{loss2.item():.8f} '
                  f'| lr:{get_lr(optimizer):.12f} | time:{t2 - t1:.4f}')

            logger.log(key='bbox_loss', data=loss1.item())
            logger.log(key='normal_loss', data=loss2.item())
            # logger.log(key='KLloss', data=loss3.item())

            train_loss += loss1.item()
            train_loss += loss2.item()

        # save the model, optimizer every args.save_steps
        if step % args.save_steps == 0:
            logger.visualize(key='train_loss', range=(-1000, -1))
            logger.save_log()
            model_saver.save(name=args.model_name, model=model)
            model_saver.save(name=args.optimizer, model=optimizer)
            # break

    return train_loss


def train_voc():
    torch.cuda.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    # init the tools
    args = config()

    json_name = args.optimizer if args.test_dataset is None else f'{args.optimizer}_VOC{args.test_dataset}'
    logger = Logger(save_path=args.save, json_name=json_name)
    model_saver = ModelSaver(save_path=args.save, name_list=[args.optimizer, args.model_name, args.scheduler_name])

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
            train_loss_1 = train_one_epoch(args, model, optimizer, voc2012_train_loader, logger, model_saver,
                                           no_mask=False)

            if epoch_now % args.val_epochs == 0:
                model.normal_eval() if not isinstance(model, nn.DataParallel) else model.module.normal_eval()
                val(args, model, val_loader, model_saver, logger)

        if 'sbd' in args.train_voc_dataset:
            train_loss_2 = train_one_epoch(args, model, optimizer, sbd_train_loader, logger, model_saver,
                                           no_mask=True)

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
