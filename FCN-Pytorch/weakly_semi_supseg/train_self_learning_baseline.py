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
from src.logger import Logger, ModelSaver
from src.weakly_semi_supseg.weakly_semi_model import get_anc_model, get_self_learning_model
from torchvision import transforms
from criterion import FocalLoss
from src.train_utils import config, get_device, model_accelerate, get_optimizer, get_lr, update_lr, poly_adjust_lr
from src.weakly_semi_supseg.eval_anc import validation
from src.criterion import get_criterion
from src.dataset.sbd_dataset import SBDDataLoaderCreator
from src.dataset.weaklysemidataset.voc_bbox_seg_dataset import get_voc_data_loader
from src.dataset.weaklysemidataset.sbd_bbox_seg_dataset import get_sbd_data_loader
from src.criterion import label_to_one_hot
import gc

class KLDiv(nn.Module):
    """
    KL(P||Q )=Plog(P/Q)
    """

    def __init__(self):
        super().__init__()

    def forward(self, f1, f2):
        # student, teacher
        loss = f1.softmax(dim=1) * torch.log(f1.softmax(dim=1) / f2.softmax(dim=1))
        return loss.mean()


# train function (forward, backward, update)
def train_one_epoch(args, model, optimizer, train_loader, logger, model_saver, no_mask=False):
    # use focal loss or not
    criterion = get_criterion(args)
    KLloss = nn.KLDivLoss()
    from criterion import CrossEntropyLoss2D
    CEloss = CrossEntropyLoss2D(reduction='none')
    if torch.cuda.device_count() > 1:
        criterion = nn.DataParallel(criterion)
        KLloss = nn.DataParallel(KLloss)
        CEloss = nn.DataParallel(CEloss)

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
                model.info_eval() if not isinstance(model, nn.DataParallel) else model.module.info_eval()
                outs_bbox = model(imgs, bbox_targets)
                outs_bbox = outs_bbox.detach()
                outs_bbox = outs_bbox.softmax(dim=1)

                # refine
                if isinstance(model, nn.DataParallel):
                    name = model.module.__class__.__name__
                else:
                    name = model.__class__.__name__

                if 'label' in name:
                    # print('label')
                    labels = bbox_targets.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0].float()
                    outs = outs_bbox * labels
                    probs, mask = outs.max(dim=1, keepdim=True)
                else:
                    bbox_targets[:, 0] = 1.
                    outs_bbox = outs_bbox * bbox_targets
                    probs, mask = outs_bbox.max(dim=1, keepdim=True)

            model.normal() if not isinstance(model, nn.DataParallel) else model.module.normal()
            outs_normal = model(imgs, bbox_targets)
            if args.train_setting == 'pow':
                probs = probs.pow(2)
            elif args.train_setting == 'thres0.5':
                probs[probs < 0.5] = 0.
            elif args.train_setting == 'strong':
                probs = torch.ones(probs.shape).to(probs.device)
            else:
                raise NotImplementedError

            loss0 = probs.transpose(1, 2).transpose(2, 3).view(-1) * CEloss(input=outs_normal, target=mask)
            loss0 = loss0.unsqueeze(dim=0).mean()
            loss0.backward()
            optimizer.step()

            t2 = time.perf_counter()
            print(f'{args.model_name} | step:{step} [{step}/{len(train_loader)}] [{iter}/{args.max_iter}] '
                  f'| {len(outs_normal)} loss_bbox:{loss0.item():.8f} | '
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
                                                   aug=args.aug)
        sbd_train_loader = get_sbd_data_loader(args, sub_mean=args.std > 0, stdize=args.std > 1, aug=args.aug)
        args.max_iter = ((len(voc2012_train_loader) * 5 if 'voc' in args.train_voc_dataset else 0)) * args.epochs

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
            for i in range(5):
                train_loss_1 += train_one_epoch(args, model, optimizer, voc2012_train_loader, logger, model_saver,
                                                no_mask=False)
                voc2012_train_loader = get_voc_data_loader(args, sub_mean=args.std > 0, stdize=args.std > 1, train=True,
                                                           aug=args.aug)
            if epoch_now % args.val_epochs == 0:
                model.normal_eval() if not isinstance(model, nn.DataParallel) else model.module.normal_eval()
                val(args, model, val_loader, model_saver, logger)

                model.refine_eval() if not isinstance(model, nn.DataParallel) else model.module.refine_eval()
                val(args, model, val_loader, model_saver, logger)

        logger.log(key='TrainLoss', data=train_loss_1 + train_loss_2, show=True)
        logger.save_log()
        logger.visualize()

        del val_loader
        gc.collect()


def train_cityscape():
    pass


if __name__ == "__main__":
    train_voc()
