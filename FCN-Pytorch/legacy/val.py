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
from train_utils import get_device
from tqdm import tqdm


def metrics(args, preds, targets):
    total_ious = []
    pixel_accs = []
    for pred, target in zip(preds, targets):
        for p, t in zip(pred, target):
            temp_iou = iou(args, p, t)
            temp_pa = pixel_acc(p, t)
            print(f'iou:{temp_iou} pixel_accuracy:{temp_pa}')
            total_ious.append(temp_iou)
            pixel_accs.append(temp_pa)
    ious = np.nanmean(total_ious, axis=1)  # list
    miou = np.nanmean(ious)
    pixel_accs = np.array(pixel_accs).mean()

    ious = list(ious)  # list
    miou = float(miou)
    pixel_accs = float(pixel_accs)

    # print(type(ious), type(miou), type(pixel_accs))
    print(f'**** ious:{ious}\nmiou:{miou}\n*** pixel_accs:{pixel_accs}')


def val(args, model, scheduler, val_loader, logger, model_saver):
    model.eval()
    device = get_device(args)
    total_ious = []
    pixel_accs = []
    for step, (imgs, targets) in enumerate(val_loader):
        print(f'*** processing : {step}/{len(val_loader)}')
        imgs = imgs.to(device)
        # targets_one_hot = label_to_one_hot(targets, n_class=args.n_class)
        # targets_one_hot = targets_one_hot.to(device)

        outs = model(imgs).data.cpu().numpy()

        N, _, h, w = outs.shape
        pred = outs.transpose(0, 2, 3, 1).reshape(-1, args.n_class).argmax(axis=1).reshape(N, h, w)
        target = targets.cpu().numpy().reshape(N, h, w)
        for p, t in zip(pred, target):
            temp_iou = iou(args, p, t)
            temp_pa = pixel_acc(p, t)
            print(f'iou:{temp_iou} pixel_accuracy:{temp_pa}')
            total_ious.append(temp_iou)
            pixel_accs.append(temp_pa)
        # break

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len

    ious = np.nanmean(total_ious, axis=1)  # list
    miou = np.nanmean(ious)
    pixel_accs = np.array(pixel_accs).mean()

    ious = list(ious)  # list
    miou = float(miou)
    pixel_accs = float(pixel_accs)

    # print(type(ious), type(miou), type(pixel_accs))
    print(f'**** ious:{ious}\nmiou:{miou}\n*** pixel_accs:{pixel_accs}')
    epoch_now = scheduler.state_dict()['last_epoch']
    # print("*** epoch: {}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch_now, pixel_accs, miou, ious))
    if logger.get_max(key='meanIoU') < miou or logger.get_max(key='meanPixel') < pixel_accs:
        model_saver.save(name=args.model_name + f'_miou_{miou:.4f}_pa_{pixel_accs:.4f}', model=model)

    logger.log(key='meanIoU', data=miou, show=True)
    logger.log(key='IoUs', data=ious, show=True)
    logger.log(key='meanPixel', data=pixel_accs, show=True)


# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
def iou(args, pred, target):
    ious = []
    for cls in range(0, args.n_class):
        pred_inds = pred.flatten() == cls  # true, false matrix
        target_inds = target.flatten() == cls
        intersection = (pred_inds & target_inds).sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        # print(f'intersection:{intersection}\nunion:{union}')
        if target_inds.sum() == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total = (target == target).sum()
    return correct / total


if __name__ == '__main__':
    from model import get_model
    from logger import ModelSaver
    from train_utils import config, get_device
    from dataset.voc_dataset import get_voc_data_loader

    args = config()

    device = get_device(args)
    load_path = '/Users/chenlinwei/code/20190717FCN-Pytorch/fcn8s/fcn8s_miou_0.5759_pa_0.8899.pkl'
    model = get_model(name='fcn8s', n_class=21)
    ModelSaver.load_tool(load_path=load_path, model=model)
    model = model.to(device)
    model.eval()

    voc2011 = get_voc_data_loader(args, year='2011', train=False)
    # for step, (imgs, targets) in enumerate(voc2011, start=1):
    #     model
    #     pass
    val(args, model, scheduler=None, val_loader=voc2011, logger=None, model_saver=None)
