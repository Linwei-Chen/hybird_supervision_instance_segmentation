import os
import sys
import tarfile
import collections
from torchvision.datasets import VisionDataset

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity
import torch

from dataset.weaklysemidataset.segmentation_transform import Compose2, Resize, Normalize, RandomHorizontalFlip, \
    ToTensor, RandomResizedCrop, RandomRotation, RandomCrop, SampleParing, CutOut
from dataset.voc_dataset import MEAN, STD
from dataset.photo_augmentations import PhotometricDistortWithBBox
import numpy as np
from criterion import label_to_one_hot


def get_transform(args, sub_mean=True, stdize=True, year='2012', train=True, del_ignore=False, aug='standard'):
    if aug == 'standard':
        train_transform = Compose2([
            RandomRotation(degrees=15, expand=True),
            RandomResizedCrop(size=args.crop_size),
            RandomHorizontalFlip(p=0.5),
            PhotometricDistortWithBBox(),
            ToTensor(del_ignore=del_ignore),
            Normalize(MEAN if sub_mean else [0, 0, 0], STD if stdize else [1.0, 1.0, 1.0])
        ])
    elif aug == 'standard0.5':
        train_transform = Compose2([
            RandomRotation(degrees=15, expand=True),
            RandomResizedCrop(size=args.crop_size),
            RandomHorizontalFlip(p=0.5),
            PhotometricDistortWithBBox(p=0.5),
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
            RandomRotation(degrees=15, expand=True),
            RandomResizedCrop(size=args.crop_size),
            RandomHorizontalFlip(p=0.5),
            ToTensor(del_ignore=del_ignore),
            Normalize(MEAN if sub_mean else [0, 0, 0], STD if stdize else [1.0, 1.0, 1.0])
        ])
    elif aug == 'standard--':
        train_transform = Compose2([
            RandomResizedCrop(size=args.crop_size),
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
            PhotometricDistortWithBBox(),
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
    print(train_transform)
    print(val_transform)
    return train_transform, val_transform


def iterable_to_str(iterable):
    return "'" + "', '".join([str(item) for item in iterable]) + "'"


def verify_str_arg(value, arg=None, valid_values=None, custom_msg=None):
    if not isinstance(value, torch._six.string_classes):
        if arg is None:
            msg = "Expected type str, but got type {type}."
        else:
            msg = "Expected type str for argument {arg}, but got type {type}."
        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)

    if valid_values is None:
        return value

    if value not in valid_values:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = ("Unknown value '{value}' for argument {arg}. "
                   "Valid values are {{{valid_values}}}.")
            msg = msg.format(value=value, arg=arg,
                             valid_values=iterable_to_str(valid_values))
        raise ValueError(msg)

    return value


DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': 'TrainVal/VOCdevkit/VOC2011'
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': 'VOCdevkit/VOC2010'
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': 'VOCdevkit/VOC2009'
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': 'VOCdevkit/VOC2008'
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': 'VOCdevkit/VOC2007'
    }
}


class VOCSegmentationWithBBox(VisionDataset):
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
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 mode='one-hot-overlap',
                 duplicate=1):
        super().__init__(root, transforms, transform, target_transform)
        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        self.image_set = verify_str_arg(image_set, "image_set",
                                        ("train", "trainval", "val"))

        # if overlap is true:
        # bbox targets will be in overlap-multi-hot style,
        # else bbox targets will be in one-hot style, small object will be set as primary
        assert mode in ('one-hot-overlap', 'one-hot', 'onlybox', 'onlybox0.2')
        self.mode = mode
        self.bbox_annotation_transform = VOCAnnotationTransform()

        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClass')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names] * duplicate
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names] * duplicate
        assert (len(self.images) == len(self.masks))

        annotation_dir = os.path.join(voc_root, 'Annotations')
        # annotation_splits_dir = os.path.join(voc_root, 'ImageSets/Main')
        # annotation_split_f = os.path.join(annotation_splits_dir, image_set.rstrip('\n') + '.txt')
        #
        # with open(os.path.join(annotation_split_f), "r") as f:
        #     file_names = [x.strip() for x in f.readlines()]
        #
        # self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names] * duplicate
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        width, height = img.size

        # bbox_target = self.parse_voc_xml(
        #     ET.parse(self.annotations[index]).getroot())

        bbox_target = self.bbox_annotation_transform(ET.parse(self.annotations[index]).getroot(), width, height)
        # print(bbox_target)
        res = []
        for i in bbox_target:
            # ['xmin', 'ymin', 'xmax', 'ymax']
            res.append({
                'cls': i[4],
                'area': int(i[2] - i[0] + 1) * int(i[3] - i[1] + 1),
                'bbox': i[:4]
            })
        res.sort(key=lambda x: x['area'], reverse=True)
        if self.mode in ('one-hot-overlap', 'one-hot'):
            mask_target = Image.open(self.masks[index])
            bbox_target = self._get_bbox_target(res, height, width, mode=self.mode)
        elif self.mode in ('onlybox', 'onlybox0.2'):
            bbox_target = self._get_bbox_target(res, height, width, mode='one-hot-overlap')
            mask_target = self._get_bbox_target(res, height, width, mode=self.mode)
        else:
            raise NotImplementedError

        if self.transforms is not None:
            img, mask_target, bbox_target = self.transforms(img, mask_target, bbox_target)
        return img, mask_target, bbox_target
        # return img, mask_target, torch.Tensor([0])

    def _get_bbox_target(self, res, height, width, mode):
        if mode == 'one-hot-overlap':
            mask = np.zeros((21, height, width))
            mask[0, :, :] = 1
            for inst in res:
                l, u, r, b = inst['bbox']
                mask[inst['cls'], u:b + 1, l:r + 1] = 1
                mask[0, u:b + 1, l:r + 1] = 0
            bbox_target = torch.from_numpy(mask).float()
        elif mode == 'one-hot':
            mask = np.zeros((height, width))
            for inst in res:
                l, u, r, b = inst['bbox']
                mask[u:b + 1, l:r + 1] = inst['cls']

            mask = label_to_one_hot(
                torch.from_numpy(np.expand_dims(mask, axis=0)).long(), n_class=21).numpy().squeeze(0)
            bbox_target = torch.from_numpy(mask).float()

        elif mode == 'onlybox':
            mask = np.zeros((height, width))
            for inst in res:
                l, u, r, b = inst['bbox']
                mask[u:b + 1, l:r + 1] = inst['cls']
            bbox_target = Image.fromarray(mask)

        elif mode == 'onlybox0.2':
            mask = np.zeros((height, width))
            for inst in res:
                l, u, r, b = inst['bbox']
                h = b - u + 1
                w = r - l + 1
                _h = int(0.227 * h)
                _w = int(0.227 * w)
                mask[u:b + 1, l:r + 1] = 255.
                mask[u + _h:b + 1 - _h, l + _w:r + 1 - _w] = inst['cls']
            bbox_target = Image.fromarray(mask)

        else:
            raise NotImplementedError
        return bbox_target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


VOC_CLASSES = (  # always index 0
    'background',
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)


class VOCAnnotationTransform(object):
    # 将VOC的标注转换为 (x,y,w,h,class), class为上面VOC_CLASSES的序号
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            # 将物体名称与0~class数量绑定
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    # 可调用对象
    def __call__(self, target, width, height, to_relative=False):
        """Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        # 对ET.Element 里面名字为'object'的对象进行遍历
        # 具体用法：https://www.cnblogs.com/ifantastic/archive/2013/04/12/3017110.html
        for obj in target.iter('object'):
            # difficult VOC文档里的含义，标为 1 表示难以辨认
            # ‘difficult’: an object marked as ‘difficult’ indicates that the object is considered
            # difficult to recognize, for example an object which is clearly visible but unidentifiable
            # without substantial use of context. Objects marked as difficult are currently ignored
            # in the evaluation of the challenge.
            difficult = int(obj.find('difficult').text) == 1
            # 检测目标为难以检测而且self.keep_difficult标记为1才继续进行操作
            if not self.keep_difficult and difficult:
                continue

            # 用法解释：
            # str = "00000003210Runoob01230000000";
            # print str.strip( '0' );  # 去除首尾字符 0
            name = obj.find('name').text.lower().strip()
            # 数据格式：
            # <bndbox>
            # 			<xmin>174</xmin>
            # 			<ymin>101</ymin>
            # 			<xmax>349</xmax>
            # 			<ymax>351</ymax>
            # 		</bndbox>
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            # 得到一组0~1.0范围的值
            for i, pt in enumerate(pts):
                # bbox 数值为像素点的位置，从1开始取所以要减去1？
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                if to_relative:
                    cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            # 查找name类别对应的标号
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]
        # res: tensor[ ,5] i.e. [xmin, ymin, xmax, ymax, label_ind], ... ]
        return res


def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)


def get_voc_data_set(args, sub_mean=True, stdize=True, year='2012', train=True,
                     del_ignore=False, aug='standard', mode='one-hot-overlap', duplicate=1):
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
        voc_train_dataset = VOCSegmentationWithBBox(root=args.voc_data_path,
                                                    year=year,
                                                    image_set='train',
                                                    download=False,
                                                    transform=None,
                                                    target_transform=None,
                                                    transforms=train_transform,
                                                    mode=mode, duplicate=duplicate)
        return voc_train_dataset
    else:
        voc_val_dataset = VOCSegmentationWithBBox(root=args.voc_data_path,
                                                  year=year,
                                                  image_set='val',
                                                  download=False,
                                                  transform=None,
                                                  target_transform=None,
                                                  transforms=val_transform,
                                                  mode=mode)
        return voc_val_dataset


from torch.utils.data import DataLoader


def get_voc_data_loader(args, sub_mean=True, stdize=True, year='2012', train=False,
                        del_ignore=False, aug='standard', mode='one-hot-overlap', duplicate=1):
    if train:
        voc_train_dataset = get_voc_data_set(args, sub_mean, stdize, year, train, del_ignore, aug, mode, duplicate)
        return DataLoader(dataset=voc_train_dataset,
                          batch_size=args.batch_size,
                          shuffle=args.shuffle,
                          num_workers=args.prefetch,
                          pin_memory=False,
                          drop_last=True)
    else:
        voc_val_dataset = get_voc_data_set(args, sub_mean, stdize, year, train)
        return DataLoader(dataset=voc_val_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=args.prefetch,
                          pin_memory=False,
                          drop_last=True)


def test():
    from train_utils import config
    from torchvision import transforms
    args = config()
    dataset = get_voc_data_loader(args, sub_mean=True, stdize=True, train=True, mode='onlybox0.2', duplicate=10)
    print(len(dataset))
    for _, (imgs, targets, bbox_targets) in enumerate(dataset):
        # print(f'imgs:{imgs}')
        print(f'bbox_targets:{bbox_targets.shape}')
        print(f'targets:{targets.shape}')
        # print(f'bbox_targets:{bbox_targets}')
        # print(f'targets:{targets[targets > 0]}')
        # print(f'bbox_targets:{bbox_targets[bbox_targets > 0]}')
        print(bbox_targets)
        img = transforms.ToPILImage()(imgs[0])
        target = transforms.ToPILImage()(targets[0].float()).convert('I')
        print(bbox_targets[bbox_targets > 0])
        print(bbox_targets.shape)
        bbox_target = transforms.ToPILImage()(bbox_targets[0].argmax(dim=0, keepdim=True).float()).convert('I')
        # bbox_target = transforms.ToPILImage()(bbox_targets[0].float())

        img.show()
        target.show()
        bbox_target.show()

        break


if __name__ == '__main__':
    test()
