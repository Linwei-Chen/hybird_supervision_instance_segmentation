# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim

import argparse
from logger import Logger, ModelSaver
from model import get_model
from torchvision import transforms


def config():
    parser = argparse.ArgumentParser(description='Trains FCN on SBD and VOC',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('--voc_data_path', type=str, default='/Users/chenlinwei/dataset',
                        help='Root for the voc dataset.')
    parser.add_argument('--sbd_data_path', type=str, default='/Users/chenlinwei/dataset/SBD_FULL11355/dataset',
                        help='Root for the voc dataset.')
    parser.add_argument('--cityscapes_data_path', type=str, default='/Users/chenlinwei/dataset/cityscapes',
                        help='Root for the cityscape dataset.')
    parser.add_argument('--pbdl_path', type=str, default='/Users/chenlinwei/dataset/pbdl',
                        help='Root for the voc dataset.')
    parser.add_argument('--aug', type=str, default='standard', help='DataAugmentation')

    # for weakly supervised segmentation
    parser.add_argument('--sbd_aug', type=str, default='standard--', help='DataAugmentation')
    parser.add_argument('--voc_aug', type=str, default='standard', help='DataAugmentation')
    parser.add_argument('--voc_duplicate', type=int, default=5, help='Dataset Duplicate')
    parser.add_argument('--self_learning', type=str, default='normal', help='self learning mode')
    parser.add_argument('--train_setting', type=str, default='pow', help='put the dataset to ram if 1.')

    parser.add_argument('--train_voc_dataset', type=str, default='sbd voc', help='use sbd, voc for train, and eval')

    parser.add_argument('--crop_size', type=int, default=320, help='Choose crop size of RandomCrop.')
    parser.add_argument('--stride', type=int, default=32, help='the stride of CNN model.')
    parser.add_argument('--train_to_ram', type=bool, default=False, help='put the dataset to ram if 1.')
    parser.add_argument('--val_to_ram', type=bool, default=False, help='put the dataset to ram if 1.')

    # Optimization options
    parser.add_argument('--optimizer', '-op', type=str, default='Adam', help='Optimizer to train model.')
    parser.add_argument('--scheduler', type=str, default=None, help='scheduler to train model, None means MultiSteps')
    parser.add_argument('--criterion', type=str, default='bce', help='criterion to train model.')

    parser.add_argument('--epochs', '-e', type=int, default=60, help='Number of epochs to train.')
    parser.add_argument('--val_epochs', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=2, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0007, help='The Learning Rate.')
    parser.add_argument('--lr_update', type=float, default=None, help='The Learning Rate to update.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=2e-4, help='Weight decay (L2 penalty).')
    parser.add_argument('--test_bs', type=int, default=10)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--schedule', type=int, nargs='+', default=[25, 45],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default=None, help='Folder to save checkpoints.')
    parser.add_argument('--save_steps', '-ss', type=int, default=50, help='steps to save checkpoints.')

    parser.add_argument('--eval', type=int, default=0, help='if 1, only eval, skip training.')

    parser.add_argument('--test_dataset', type=str, default=None, choices=['2011', '2012'], help='Test only flag.')

    # Architecture
    parser.add_argument('--model_name', type=str, default='deeplabv3plusxception', help='')
    parser.add_argument('--teacher_model_name', type=str, default='deeplabv3plusxception', help='')

    parser.add_argument('--n_class', type=int, default=21, help='classes number to classify.')

    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--log', type=str, default=None, help='Log folder.')
    parser.add_argument('--std', type=int, default=2, help='standardize the data distribution.')
    args = parser.parse_args()

    args.optimizer = args.optimizer.lower()  # + f'_{args.scheduler}' if args.scheduler == 'poly' else ''
    args.model_name = args.model_name.lower()
    # args.scheduler = args.scheduler.lower()
    if args.aug == 1: args.aug = 'standard'
    if args.save is None:
        args.save = f'../{args.model_name}'
    if args.log is None:
        args.log = f'../{args.model_name}'

    if args.scheduler is None:
        args.scheduler_name = f'{args.optimizer}_scheduler'
    else:
        args.scheduler_name = f'{args.optimizer}_{args.scheduler}_scheduler'
    print(args)
    return args


def get_device(args):
    return torch.device('cuda' if args.ngpu >= 1 and torch.cuda.is_available() else 'cpu')


def model_accelerate(args, model):
    from model.deeplabv3plus.sync_batchnorm.replicate import patch_replication_callback, DataParallelWithCallback
    from model.deeplabv3plus.sync_batchnorm import convert_model
    if torch.cuda.device_count() > 0:
        model = convert_model(model)
        model = torch.nn.DataParallel(model)
        patch_replication_callback(model)
        device = get_device(args)
        model = model.to(device)
        print(f'*** {model.__class__.__name__} to GPUs, syncbatch OK.')
    return model


def get_params(model, key):
    for m in model.named_modules():
        # print(m)
        if key == '1x':
            if ('backbone' in m[0] or 'pretrained_net' in m[0]) and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p
        elif key == '10x':
            if ('backbone' not in m[0] and 'pretrained_net' not in m[0]) and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p


def get_optimizer(args, name: str, model):
    assert isinstance(name, str)
    if name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr)
    elif name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    elif name.lower() == 'adam_finetune':
        try:
            optimizer = optim.Adam(params=[
                {'params': get_params(model.module, key='1x'), 'lr': 0.1 * args.lr},
                {'params': get_params(model.module, key='10x'), 'lr': args.lr}]
            )
        except:
            optimizer = optim.Adam(params=[
                {'params': get_params(model, key='1x'), 'lr': 0.1 * args.lr},
                {'params': get_params(model, key='10x'), 'lr': args.lr}]
            )
        return optimizer
    elif name.lower() == 'sgd_finetune':
        try:
            optimizer = optim.SGD(
                params=[
                    {'params': get_params(model.module, key='1x'), 'lr': 0.1 * args.lr},
                    {'params': get_params(model.module, key='10x'), 'lr': args.lr}
                ],
                momentum=args.momentum,
                weight_decay=4e-5
            )
        except:
            optimizer = optim.SGD(
                params=[
                    {'params': get_params(model, key='1x'), 'lr': 0.1 * args.lr},
                    {'params': get_params(model, key='10x'), 'lr': args.lr}
                ],
                momentum=args.momentum,
                weight_decay=4e-5
            )
        return optimizer
    else:
        raise Exception(f'*** optimizer:{name} not implemented!')
    # assert name.lower() in [op.lower() for op in op_dict.keys()]
    # return op_dict.get(name)


def get_lr(optimizer):
    if isinstance(optimizer, nn.DataParallel):
        optimizer = optimizer.module
    for param_group in optimizer.param_groups:
        return param_group['lr']


def update_lr(module, lr: float):
    if isinstance(module, nn.DataParallel):
        module = module.module
    try:
        module.param_groups[0]['lr'] = 0.1 * lr
        module.param_groups[1]['lr'] = lr
        print('====> update lr-0:{:.12f} | lr-1:{:.12f} complete.'.
              format(module.param_groups[0]['lr'], module.param_groups[1]['lr']))
    except:
        for param_group in module.param_groups:
            # print(param_group)
            param_group['lr'] = lr
        print(f'====> update lr:{lr:.12f} complete.')


def get_sheduler(args):
    pass


def from_pre_to_img(t):
    """

    :param t: tensor of [1, 21, h, w]
    :return:
    """
    from dataset.voc_dataset import COLORMAP
    cm = torch.Tensor(COLORMAP)
    t = t.cpu()
    t = t.argmax(dim=1, keepdim=True)
    t = t[0].squeeze()
    h, w = t.size()
    temp = torch.zeros((3, h, w))
    for i in range(h):
        for j in range(w):
            temp[:, i, j] = cm[t[i, j]]
    res = transforms.ToPILImage()(temp / 255.0)
    res.show()
    return res


def poly_adjust_lr(optimizer, lr, itr, max_itr, power=0.9):
    now_lr = lr * (1 - itr / (max_itr + 1)) ** power
    update_lr(optimizer, now_lr)
    return now_lr


def _test_finetune_loading():
    #  lr tset
    model = get_model('deeplabv3plusdenseaspp', 21)
    optimizer = optim.Adam([{'params': get_params(model, key='1x'), 'lr': 1},
                            {'params': get_params(model, key='10x'), 'lr': 10}])

    optimizer_2 = optim.Adam(model.parameters())
    print(optimizer)
    print(optimizer_2)
    poly_adjust_lr(optimizer, 0.1, 99, 100)
    poly_adjust_lr(optimizer_2, 0.1, 99, 100)
    print(optimizer)
    print(optimizer_2)

    #  loading test:
    print(optimizer.state_dict())
    path = '/Users/chenlinwei/code/20190717FCN-Pytorch/adam_finetune.pkl'
    d = torch.load(path, map_location='cpu')
    # print(d)
    # optimizer.state_dict().update(d)
    ModelSaver.load_tool(model=optimizer, load_path=path)
    print('optimizer:\n', optimizer.state_dict())
    print('d:\n', d)
    for k in d:
        for k2 in d[k]:
            # print(k2)
            pass


def create_voc_pretrained_model_for_cityscaps():
    path = '/Users/chenlinwei/code/20190717FCN-Pytorch/deeplabv3plusxception/deeplabv3plusxception_miou_0.7955_pa_0.9542.pkl'
    pretrianed = torch.load(path, map_location='cpu')
    model = get_model('deeplabv3plusxception', 19)
    model.load_state_dict(pretrianed, strict=False)
    torch.save(model.state_dict(), '../deeplabv3plusxception_cityscape.pkl')


def _save_best():
    model = get_model('deeplabv3plusxception', 21)
    path = '/Users/chenlinwei/code/20190717FCN-Pytorch/deeplabv3plusxception/deeplabv3plusxception_miou_0.7955_pa_0.9542.pkl'
    ModelSaver.load_tool(model=model, load_path=path)
    torch.save(model.state_dict(), '../deeplabv3plusxception_miou_0.7955_pa_0.9542_cpu.pkl')


if __name__ == '__main__':
    config()
    pass
