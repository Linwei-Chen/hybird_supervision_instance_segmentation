from PIL import Image
import torch.utils.data as Data
from random import shuffle
import os
import os.path as osp
import torchvision.transforms.functional as F
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from dataset.segmentation_transform import Compose2, Resize, Normalize, RandomHorizontalFlip, ToTensor, \
    RandomResizedCrop, RandomRotation, RandomCrop, SampleParing, CutOut
from dataset.photo_augmentations import PhotometricDistort
from dataset.segmentation_transform import MEAN, STD
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from collections import namedtuple
import gc
import math
import torch.nn.functional as func


class cs:
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]
    d = {
        0: 3,
        1: 9,
        2: 7,
        3: 9,
        4: 9,
        5: 9,
        6: 4,
        7: 5,
        8: 9,
        9: 3,
        10: 8,
        11: 2,
        12: 2,
        13: 1,
        14: 1,
        15: 1,
        16: 1,
        17: 2,
        18: 2,
    }


class PBDL(Data.Dataset):
    color_stand = {'_background_': (0, (0, 0, 0)),
                   'car': (1, (128, 0, 0)),
                   'human': (2, (0, 128, 0)),
                   'road': (3, (128, 128, 0)),
                   'traffic light': (4, (0, 0, 128)),
                   'traffic sign': (5, (128, 0, 128)),
                   'tree': (6, (0, 128, 128)),
                   'building': (7, (128, 128, 128)),
                   'sky': (8, (64, 0, 0)),
                   'object': (9, (192, 0, 0))}

    def __init__(self, root, split, mode='rgb', transforms=None):
        if split == 'train':
            self.items = get_train_imgs_list(root)
        elif split == 'test':
            self.items = get_test_imgs_list(root)
        else:
            raise NotImplementedError

        # assert mode == 'rgb' or mode == 'hs' or mode == 'rgb+hs'
        self.mode = mode
        self.tranforms = transforms

    def __getitem__(self, index):
        if self.mode == 'rgb':
            img, label = PBDL.load_rgb_img(self.items[index])
        elif self.mode == 'hs':
            img, label = PBDL.load_hs_img(self.items[index])
        elif self.mode == 'hs129':
            img, label = PBDL.load_hs129_img(self.items[index])
        elif self.mode == 'rgb+hs':
            img, label = PBDL.load_rgbhs_img(self.items[index])
        else:
            raise NotImplementedError

        if self.tranforms is not None:
            img, label = self.tranforms(img, label)
        return img, label

    def get_item_and_dir_name(self, index):
        dir_name = osp.split(osp.split(self.items[index]['label'])[0])[1]
        # print(f'{index}:{dir_name}')
        return self.__getitem__(index), dir_name

    def __len__(self):
        return len(self.items)

    @staticmethod
    def vflip_and_rotate(pil_img):
        pil_img = F.vflip(pil_img)
        pil_img = F.rotate(pil_img, -90, expand=True)
        return pil_img

    @staticmethod
    def load_rgbhs_img(item):
        hs_img = item['hs']
        hs_img = readHSD(hs_img, to_129=False)
        hs_img = hs_img.transpose(0, 2, 1)

        rgb_img = item['rgb']
        rgb_img = Image.open(rgb_img)
        rgb_img = PBDL.vflip_and_rotate(rgb_img)
        rgb_img = (np.array(rgb_img).transpose((2, 0, 1)) / 255.).astype(np.float32)
        img = np.concatenate((rgb_img, hs_img), axis=0)

        label = item['label']
        label = Image.open(label)
        label = PBDL.vflip_and_rotate(label)
        return img, label

    @staticmethod
    def load_hs_img(item):
        img = item['hs']
        img = readHSD(img, to_129=False)
        img = img.transpose(0, 2, 1)

        label = item['label']
        label = Image.open(label)
        label = PBDL.vflip_and_rotate(label)
        return img, label

    @staticmethod
    def load_hs129_img(item):
        img = item['hs']
        img = readHSD(img)
        img = img.transpose(0, 2, 1)

        label = item['label']
        label = Image.open(label)
        label = PBDL.vflip_and_rotate(label)
        return img, label

    @staticmethod
    def load_rgb_img(item):
        img = item['rgb']
        img = Image.open(img)
        img = PBDL.vflip_and_rotate(img)

        label = item['label']
        label = Image.open(label)
        label = PBDL.vflip_and_rotate(label)
        return img, label


def _test_get_item_and_dir_name():
    from train_utils import config
    args = config()
    dataset = get_pbdl_data_set(args)
    for i in range(len(dataset)):
        dataset.get_item_and_dir_name(i)


from tqdm import tqdm


class PBDLFast(PBDL):
    def __init__(self, root, split, mode='hs', transforms=None, pre_load=32, repeat=32):

        super().__init__(root, split, mode, transforms)
        self.pre_load = pre_load
        self.repeat = repeat
        self.storage = []
        print('===> data:', len(self.items))
        self.size = len(self.items) * self.repeat
        for _ in tqdm(range(pre_load)):
            item = self.items.pop()
            if self.mode == 'hs':
                img, label = self.load_hs_img(item)
            elif self.mode == 'hs129':
                img, label = self.load_hs129_img(item)
            elif self.mode == 'rgb+hs':
                img, label = self.load_rgbhs_img(item)
            else:
                raise NotImplementedError

            self.storage.append({'img': img, 'label': label, 'counter': 0})
            # Image.fromarray((img[0, :, :] * 128).astype(np.uint8), mode='L').show()
            # Image.fromarray((np.array(label) * 20).astype(np.uint8), mode='L').show()

    def __getitem__(self, index):
        # print(len(self.items))
        # if len(self.storage) < self.pre_load:

        choose_idx = index % len((self.storage))
        img = self.storage[choose_idx]['img']
        label = self.storage[choose_idx]['label']
        self.storage[choose_idx]['counter'] += 1

        if self.storage[choose_idx]['counter'] >= self.repeat:
            del self.storage[choose_idx]
            gc.collect()
            try:
                item = self.items.pop()
                if self.mode == 'hs':
                    img, label = self.load_hs_img(item)
                elif self.mode == 'hs129':
                    img, label = self.load_hs129_img(item)
                elif self.mode == 'rgb+hs':
                    img, label = self.load_rgbhs_img(item)
                else:
                    raise NotImplementedError
                self.storage.append({'img': img, 'label': label, 'counter': 0})
            except Exception:
                print('items empty')

            print(f'remained data:{len(self.items)}')

        if self.tranforms is not None:
            img, label = self.tranforms(img, label)
        return img, label

    def __len__(self):
        return self.size
        # return self.pre_load* self.repeat


def _test_PBDLFast():
    pass


def get_train_imgs_list(path='/Users/chenlinwei/dataset/PBDL', check=False):
    r"""

    :param test_img_dir: str of img dir
    :param use_shuffle: use random.shuffle to shuffle the list or not
    :return: list of img_dict

    """
    path = osp.join(path, 'train')
    res = []
    dir_list = os.listdir(path)
    if check:
        print(dir_list)
    for dir_path in dir_list:
        dir_path = osp.join(path, dir_path)
        for root, dirs, files in os.walk(dir_path, topdown=True):
            item = {}
            if dir_path == root:
                for file in files:
                    if '.hsd' in file:
                        item['hs'] = osp.join(dir_path, file)
                    elif 'rgb' in file:
                        item['rgb'] = osp.join(dir_path, file)
                    elif 'label_gray' in file:
                        item['label'] = osp.join(dir_path, file)
            res.append(item)

    if check:
        for i in res:
            assert len(i) == 3
            print(i['hs'])
            print(i['rgb'])
            print(i['label'])
    return res


def get_test_imgs_list(path='/Users/chenlinwei/dataset/PBDL', check=False):
    r"""

    :param test_img_dir: str of img dir
    :param use_shuffle: use random.shuffle to shuffle the list or not
    :return: list of img_dict

    """

    res = []
    dir_list = os.listdir(osp.join(path, 'test'))
    if check:
        print(dir_list)
    for dir_name in dir_list:
        for root, dirs, files in os.walk(osp.join(path, 'test', dir_name), topdown=True):
            item = {}
            if dir_name == osp.split(root)[1]:
                for file in files:
                    if '.hsd' in file:
                        item['hs'] = osp.join(path, 'test', dir_name, file)
                    elif 'rgb' in file:
                        item['rgb'] = osp.join(path, 'test', dir_name, file)
                item['label'] = osp.join(path, 'ans', dir_name, 'label_gray.png')
            res.append(item)

    if check:
        for i in res:
            assert len(i) == 3
            print(i['hs'])
            print(i['rgb'])
            print(i['label'])
    return res


class HSDRandomCropAndToTensor:
    def __init__(self, size, pad):
        self.size = size
        self.pad = pad

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img: np.array, target):
        if self.pad:
            # print(img.shape)
            img = np.pad(img, ((0, 0), (self.size // 2, self.size // 2), (self.size // 2, self.size // 2)),
                         'constant', constant_values=0)
            # print(img.shape)
            #
            # print(target.size)
            target = F.pad(target, (self.size // 2, self.size // 2))
            # print(target.size)

        c, h, w = img.shape
        i = random.randint(0, h - self.size)
        j = random.randint(0, w - self.size)
        img = img[:, i:i + self.size, j:j + self.size]
        target = F.crop(target, i, j, self.size, self.size)
        img = torch.tensor(img)
        target = torch.from_numpy(np.array(target)).long().unsqueeze(dim=0)
        # print(target.shape)
        return img, target


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}


class HSDRandomResizedCropAndToTensor(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.125, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        hight, width = img.shape[-2:]
        for attempt in range(10):
            area = width * hight
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= width and h <= hight:
                i = random.randint(0, hight - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback
        w = min(width, hight)
        i = (hight - w) // 2
        j = (width - w) // 2
        return i, j, w, w

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        target = torch.from_numpy(np.array(target)[i:i + h, j:j + w]).unsqueeze(dim=0).float()
        target = func.interpolate(target.unsqueeze(dim=0),
                                  size=self.size, mode='nearest').squeeze(0).long()
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
            img = func.interpolate(img[:, i:i + h, j:j + w].unsqueeze(dim=0),
                                   size=self.size, mode='bilinear', align_corners=True).squeeze(0)
        else:
            img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
            img = torch.from_numpy(np.array(img)).long().unsqueeze(dim=0)
        return img, target

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class HSDToTensorAndAlign:
    def __init__(self, stride, resize_both=True):
        self.stride = stride
        self.resize_both = resize_both

    def __call__(self, img, target):
        img = torch.tensor(img)
        target = torch.from_numpy(np.array(target)).unsqueeze(dim=0).float()
        # print(target[target > 0])

        _, h, w = img.shape
        img = interpolate(img.unsqueeze(dim=0), ((h // self.stride) * self.stride, (w // self.stride) * self.stride),
                          mode='bilinear', align_corners=True).squeeze(dim=0)
        if self.resize_both:
            target = interpolate(target.unsqueeze(dim=0),
                                 ((h // self.stride) * self.stride, (w // self.stride) * self.stride),
                                 mode='nearest').squeeze(dim=0).long()
        return img, target


def get_pbdl_data_set(args, split='train', mode='rgb',
                      sub_mean=True, stdize=True, aug='normal', resize_both=True, fast=False):
    assert mode in ['rgb', 'hs', 'hs129', 'rgb+hs']
    if aug == 'normal':
        train_transform = Compose2([
            # RandomCrop(size=args.crop_size, pad_if_needed=True),
            RandomRotation(degrees=15),
            RandomResizedCrop(size=args.crop_size),
            RandomHorizontalFlip(p=0.5),
            PhotometricDistort(),
            ToTensor(show=False),
            Normalize(MEAN if sub_mean else [0, 0, 0], STD if stdize else [1.0, 1.0, 1.0])
        ]) if mode == 'rgb' else Compose2([
            # HSDRandomResizedCropAndToTensor(size=args.crop_size)
            HSDRandomCropAndToTensor(size=args.crop_size, pad=False)
        ])
    else:
        raise NotImplementedError

    val_transform = Compose2([Resize(stride=args.stride, resize_both=resize_both),
                              ToTensor(),
                              Normalize(MEAN if sub_mean else [0, 0, 0],
                                        STD if stdize else [1.0, 1.0, 1.0])]) if mode == 'rgb' else \
        Compose2([HSDToTensorAndAlign(stride=16, resize_both=resize_both)])

    if split == 'train':
        if fast and mode != 'rgb':
            dataset = PBDLFast(root=args.pbdl_path, split=split, mode=mode, transforms=train_transform)
        else:
            dataset = PBDL(root=args.pbdl_path, split=split, mode=mode, transforms=train_transform)
    elif split == 'test':
        dataset = PBDL(root=args.pbdl_path, split=split, mode=mode, transforms=val_transform)
    else:
        raise NotImplementedError

    return dataset


def get_pbdl_data_loader(args, split, mode, sub_mean=True, stdize=True, aug='normal', fast=False):
    dataset = get_pbdl_data_set(args, split, mode, sub_mean, stdize, aug, fast=fast)
    if split == 'train':
        return DataLoader(dataset=dataset,
                          batch_size=args.batch_size,
                          shuffle=args.shuffle,
                          num_workers=args.prefetch,
                          pin_memory=False,
                          drop_last=True)
    elif split == 'test':
        return DataLoader(dataset=dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=args.prefetch,
                          pin_memory=False,
                          drop_last=False)
    else:
        raise NotImplementedError


def _check():
    # train check
    res = get_train_imgs_list(check=True)
    label = Image.open(res[0]['label'])
    img = Image.open(res[0]['rgb'])
    img = PBDL.vflip_and_rotate(img)
    label.show()
    img.show()
    # test check
    res = get_test_imgs_list(check=True)
    label = Image.open(res[0]['label'])
    img = Image.open(res[0]['rgb'])
    img = PBDL.vflip_and_rotate(img)
    label.show()
    img.show()
    pass


def _test_data_loader():
    from train_utils import config
    args = config()
    data_loader = get_pbdl_data_loader(args, split='train', mode='hs129', fast=False)
    for _, (imgs, labels) in enumerate(data_loader):
        print(imgs.shape, labels.shape)
        transforms.ToPILImage()(imgs[0, 10] * 10).convert('I').show()
        transforms.ToPILImage()(labels[0]).show()
        # print(imgs)
        print(labels)
        pass
    pass


def get_13c_data(scoredata, coeff, average, width, height, SR, times_255=False):
    temp = scoredata.dot(coeff)
    data = temp + average
    data = data.reshape((width[0], height[0], SR[0]), order='F') * (255. if times_255 else 1.)
    data = data.transpose(2, 1, 0)
    return data


def readHSD(file_path, to_129=True, times_255=False):
    """

    :param file_path:
    :param times_255:
    :return: data [c, h, w]
    """
    import array
    import numpy as np
    from PIL import Image
    def get_bit(b_file, type: str, bit: int):
        temp = array.array(type)
        temp.fromfile(b_file, bit)
        temp = np.array(temp)
        return temp

    f = open(file_path, 'rb')
    height = get_bit(f, 'I', 1)
    width = get_bit(f, 'I', 1)
    SR = get_bit(f, 'I', 1)
    D = get_bit(f, 'I', 1)
    startw = get_bit(f, 'I', 1)
    stepw = get_bit(f, 'f', 1)
    endw = get_bit(f, 'I', 1)

    average = get_bit(f, 'f', SR[0]).reshape(SR[0], 1)
    coeff = get_bit(f, 'f', D[0] * SR[0]).reshape((SR[0], D[0]), order='F')
    scoredata = get_bit(f, 'f', height[0] * width[0] * D[0]).reshape((D[0], height[0] * width[0]), order='F')

    average = average.transpose()
    coeff = coeff.transpose()
    scoredata = scoredata.transpose()

    if to_129:
        temp = scoredata.dot(coeff)
        data = temp + average
        data = data.reshape((width[0], height[0], SR[0]), order='F') * (255. if times_255 else 1.)
        data = data.transpose(2, 1, 0)  # c, h, w

        # data = data.transpose(0, 2, 1)
        # img_np = (data[78, :, :]).astype(np.uint8)
        # print(img_np.shape, img_np)
        # tensor = transforms.ToTensor()(img_np)
        # print(tensor.shape, tensor)
        # img = Image.fromarray(img_np, 'L')
        # img.show()
        del f, height, width, SR, D, startw, stepw, endw, average, coeff, scoredata, temp
        gc.collect()
        return data
    else:
        coeff_pinv = np.linalg.pinv(coeff)
        data = scoredata + average.dot(coeff_pinv)
        data = data.reshape((width[0], height[0], D[0]), order='F')
        data = data.transpose(2, 1, 0)
        del f, height, width, SR, D, startw, stepw, endw, average, coeff, scoredata, coeff_pinv
        gc.collect()
        return data


def _test_resdHSD():
    readHSD('/Users/chenlinwei/dataset/PBDL/train/rgb20190528_180648_73242_json/20190528_180648_73242.hsd', True)
    pass


def _test_HSDRandomCrop():
    img = readHSD('/Users/chenlinwei/dataset/PBDL/train/rgb20190528_180044_6684_json/20190528_180044_6684.hsd', True)
    target = Image.open(
        '/Users/chenlinwei/dataset/PBDL/train/rgb20190528_180044_6684_json/rgb20190528_180044_6684_cropped.png')
    img, target = HSDRandomCropAndToTensor(size=640, pad=False)(img, target)
    Image.fromarray(img[50, :, :]).show()
    target.show()


if __name__ == '__main__':
    # _test_HSDRandomCrop()
    # _test_resdHSD()
    _test_data_loader()
    pass
