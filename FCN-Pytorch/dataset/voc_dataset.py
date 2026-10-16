from torchvision import datasets
from torch.utils.data import DataLoader

import torch
import numpy as np
from dataset.segmentation_transform import Compose2, Resize, Normalize, RandomHorizontalFlip, ToTensor, \
    RandomResizedCrop, RandomRotation, RandomCrop, SampleParing, CutOut
from dataset.photo_augmentations import PhotometricDistort
from dataset.segmentation_transform import MEAN, STD
from torch.utils.data import Dataset
from tqdm import tqdm
from criterion import label_to_one_hot
from dataset.AutoAugmentation import AutoAugmentation

CLASSES = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# RGB color for each class.
COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]

VOC2012_RGB_mean = [x / 255.0 for x in [115.40426167733621, 110.06103779748445, 101.88298592833736]]
VOC2011_RGB_mean = [x / 255.0 for x in [115.94563480839578, 110.63228262070014, 102.32305646996195]]
VOC2012_RGB_std = [x / 255.0 for x in [30.887643371028858, 31.592458778527075, 35.54421759971818]]
# temp
VOC2011_RGB_std = [x / 255.0 for x in [30.887643371028858, 31.592458778527075, 35.54421759971818]]


def get_transform(args, sub_mean=True, stdize=True, year='2012', train=True, del_ignore=False, aug='standard'):
    if aug == 'standard':
        train_transform = Compose2([
            RandomRotation(degrees=15, expand=True),
            RandomResizedCrop(size=args.crop_size),
            RandomHorizontalFlip(p=0.5),
            PhotometricDistort(),
            ToTensor(del_ignore=del_ignore, show=False),
            Normalize(MEAN if sub_mean else [0, 0, 0], STD if stdize else [1.0, 1.0, 1.0])
        ])
    elif aug == 'standard0.5':
        train_transform = Compose2([
            RandomRotation(degrees=15, expand=True),
            RandomResizedCrop(size=args.crop_size),
            RandomHorizontalFlip(p=0.5),
            PhotometricDistort(p=0.5),
            ToTensor(del_ignore=del_ignore),
            Normalize(MEAN if sub_mean else [0, 0, 0], STD if stdize else [1.0, 1.0, 1.0])
        ])
    elif aug == 'normal':
        train_transform = Compose2([
            RandomCrop(size=args.crop_size, pad_if_needed=True),
            ToTensor(del_ignore=del_ignore),
            Normalize(MEAN if sub_mean else [0, 0, 0], STD if stdize else [1.0, 1.0, 1.0])
        ])
    elif aug == 'standard-':
        train_transform = Compose2([
            RandomRotation(degrees=45, expand=True),
            RandomResizedCrop(size=args.crop_size, ratio=(1. / 2., 2. / 1.)),
            RandomHorizontalFlip(p=0.5),
            ToTensor(del_ignore=del_ignore),
            Normalize(MEAN if sub_mean else [0, 0, 0], STD if stdize else [1.0, 1.0, 1.0])
        ])
    elif aug == 'autoaugmentation':
        train_transform = Compose2([
            RandomResizedCrop(size=args.crop_size),
            AutoAugmentation(),
            ToTensor(del_ignore=del_ignore),
            Normalize(MEAN if sub_mean else [0, 0, 0], STD if stdize else [1.0, 1.0, 1.0])
        ])
    elif aug == 'pairingsamples':
        train_transform = Compose2([
            RandomRotation(degrees=15, expand=True),
            RandomResizedCrop(size=args.crop_size),
            RandomHorizontalFlip(p=0.5),
            ToTensor(del_ignore=del_ignore),
            SampleParing(dataset=get_voc_data_set(args, sub_mean, stdize, year, train, del_ignore, aug='standard')),
            Normalize(MEAN if sub_mean else [0, 0, 0], STD if stdize else [1.0, 1.0, 1.0])
        ])
    elif aug == 'cutout':
        train_transform = Compose2([
            RandomRotation(degrees=15, expand=True),
            RandomResizedCrop(size=args.crop_size),
            RandomHorizontalFlip(p=0.5),
            CutOut(size=args.crop_size // 3, mean=MEAN, p=1.0),
            ToTensor(del_ignore=del_ignore),
            Normalize(MEAN if sub_mean else [0, 0, 0], STD if stdize else [1.0, 1.0, 1.0])
        ])
    elif aug == 'cutout0.5':
        train_transform = Compose2([
            RandomRotation(degrees=15, expand=True),
            RandomResizedCrop(size=args.crop_size),
            RandomHorizontalFlip(p=0.5),
            CutOut(size=args.crop_size // 3, mean=MEAN, p=0.5),
            ToTensor(del_ignore=del_ignore),
            Normalize(MEAN if sub_mean else [0, 0, 0], STD if stdize else [1.0, 1.0, 1.0])
        ])
    elif aug == 'standard+cutout0.5':
        train_transform = Compose2([
            RandomRotation(degrees=15, expand=True),
            RandomResizedCrop(size=args.crop_size),
            RandomHorizontalFlip(p=0.5),
            PhotometricDistort(),
            CutOut(size=args.crop_size // 3, mean=MEAN, p=0.5),
            ToTensor(del_ignore=del_ignore),
            Normalize(MEAN if sub_mean else [0, 0, 0], STD if stdize else [1.0, 1.0, 1.0])
        ])
    else:
        raise NotImplementedError

    val_transform = Compose2([Resize(stride=args.stride, resize_both=False),
                              ToTensor(del_ignore=del_ignore),
                              Normalize(MEAN if sub_mean else [0, 0, 0],
                                        STD if stdize else [1.0, 1.0, 1.0])])
    return train_transform, val_transform


def image2label(img):
    cm2lbl = np.zeros(256 ** 3)
    for i, cm in enumerate(COLORMAP):
        cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

    data = np.array(img, dtype=np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1] * 256 + data[:, :, 2])
    return np.array(cm2lbl[idx], dtype=np.int64)


def get_voc_data_set(args, sub_mean=True, stdize=True, year='2012', train=True, del_ignore=False, aug='standard'):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

        Args:
            root (string): Root directory of the VOC Dataset.
            year (string, optional): The dataset year, supports years 2007 to 2012.
            image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
    """

    assert year == '2012' or year == '2011'
    print(f'===> VOC: submean:{sub_mean} | stdize:{stdize} | year:{year} | train:{train} | aug:{aug}')
    train_transform, val_transform = get_transform(args, sub_mean, stdize, year, train, del_ignore, aug)

    if train:
        voc_train_dataset = datasets.VOCSegmentation(root=args.voc_data_path,
                                                     year=year,
                                                     image_set='train',
                                                     download=False,
                                                     transform=None,
                                                     target_transform=None,
                                                     transforms=train_transform)
        return voc_train_dataset
    else:
        voc_val_dataset = datasets.VOCSegmentation(root=args.voc_data_path,
                                                   year=year,
                                                   image_set='val',
                                                   download=False,
                                                   transform=None,
                                                   target_transform=None,
                                                   transforms=val_transform)
        return voc_val_dataset


def get_voc_data_loader(args, sub_mean=True, stdize=True, year='2012', train=True, del_ignore=False, aug='standard'):
    if train:
        voc_train_dataset = get_voc_data_set(args, sub_mean, stdize, year, train, del_ignore, aug)
        return DataLoader(dataset=voc_train_dataset,
                          batch_size=args.batch_size,
                          shuffle=args.shuffle,
                          num_workers=args.prefetch,
                          pin_memory=True,
                          drop_last=True)
    else:
        voc_val_dataset = get_voc_data_set(args, sub_mean, stdize, year, train)
        return DataLoader(dataset=voc_val_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=args.prefetch,
                          pin_memory=True,
                          drop_last=True)


def dataset_test():
    from train import config
    from model.fcn import FCN8s, VGGNet
    import torch.nn as nn
    import torch.optim as Opim

    args = config()
    model = FCN8s(pretrained_net=VGGNet(requires_grad=True), n_class=21)
    optimizer = Opim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    # dataset = get_voc_data_loader(args, sub_mean=False, stdize=False)
    dataset = get_voc_data_loader(args, sub_mean=True, stdize=True)
    for _, (imgs, targets) in enumerate(dataset):
        optimizer.zero_grad()
        print(imgs.size(), targets.size())
        print(f'imgs:{imgs}\ntargets:{targets}')
        # targets = targets.copy_()
        # for i in targets.view(-1): print(i)
        outs = model(imgs)
        targets_one_hot = label_to_one_hot(targets, 21)
        # print(f'targets:{targets}\none-hot:{targets_one_hot}')
        targets_one_hot_argmax = targets_one_hot.argmax(dim=1, keepdim=True)
        print(f'targets_one_hot_argmax:{targets_one_hot_argmax}\ntargets:{targets}')
        print(f'check:{torch.eq(targets, targets_one_hot_argmax)}')
        # print(outs.type(), targets_one_hot.type())

        BCEWithLogitsLoss_check = nn.BCEWithLogitsLoss()(input=targets_one_hot * 1000, target=targets_one_hot)
        # BCEWithLogitsLoss_check = nn.BCELoss()(input=targets_one_hot, target=targets_one_hot)
        print(f'BCEWithLogitsLoss_check:{BCEWithLogitsLoss_check}')

        loss = criterion(outs, targets_one_hot)
        loss.backward()
        print(f'loss:{loss.item()}')
        # loss_for_check = nn.CrossEntropyLoss()(targets_one_hot.reshape(1, 21, -1) * 10000, targets.reshape(1, -1))
        # print(f'loss_for_check:{loss_for_check.item()}')
        # zero_entropy = nn.CrossEntropyLoss()(torch.Tensor([[1000, 0, 0], [10000, 0, 0], [10000, 0, 0]]).float(),
        #                                      torch.Tensor([0, 0, 0]).long())
        # print(f'zero_entropy:{zero_entropy}')
        # optimizer.step()
        pass


def cross_entropy_test():
    y = torch.tensor([10, 9]).long()
    x = torch.nn.functional.one_hot(y, 11).float()
    print(f'x:{x}\ny:{y}')
    print(f'x:{x.size()}\ny:{y.size()}')
    loss = torch.nn.CrossEntropyLoss()(input=x, target=y)
    print(f'loss:{loss.item()}')
    pass


class VOCDataLoaderCreator:
    """
    deprecate
    """

    def __init__(self, args, sub_mean=True, stdize=True, year='2012', train=True, del_ignore=False, aug='standard',
                 RAM=False):
        class VOCRAMDataSet(Dataset):
            def __init__(self):
                super().__init__()
                voc_dataset = get_voc_data_set(args, sub_mean, stdize, year, train, del_ignore, aug)
                self.data_set = []
                for i in tqdm(range(voc_dataset.__len__())):
                    self.data_set.append(voc_dataset.__getitem__(i))
                print(f'{self.__class__} Loaded!')

            def __getitem__(self, index):
                return self.data_set[index]

            def __len__(self):
                return len(self.data_set)

            def shuffle(self):
                from random import shuffle
                shuffle(self.data_set)

        self.args = args
        self.sub_mean = sub_mean
        self.stdize = stdize
        self.train = train
        self.year = year
        self.aug = aug
        self.RAM = RAM
        self.RAMDataset = VOCRAMDataSet() if RAM else None

    def get_data_loader(self):
        args = self.args
        if self.RAM:
            if self.train:
                return DataLoader(dataset=self.RAMDataset,
                                  batch_size=args.batch_size,
                                  shuffle=args.shuffle,
                                  num_workers=args.prefetch,
                                  pin_memory=True, drop_last=True)
            else:
                return DataLoader(dataset=self.RAMDataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=args.prefetch,
                                  pin_memory=True, drop_last=True)
        else:
            return get_voc_data_loader(args, self.sub_mean, self.stdize, self.year, self.train, aug=self.aug)


if __name__ == '__main__':
    # cross_entropy_test()
    from train import config

    args = config()
    dataloader_creator = VOCDataLoaderCreator(args, aug='standard', RAM=False)
    dataset_test()
    pass
