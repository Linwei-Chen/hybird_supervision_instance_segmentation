# coding:UTF-8
import os
from random import shuffle
from PIL import Image
import numpy as np
from tqdm import tqdm

img_dir = '/Volumes/STORAGE/code/20190717FCN-Pytorch/PBDLHS/HS_50.99'


def from_dir_get_imgs_list(test_img_dir, only_root=False, use_shuffle: bool = False):
    r"""

    :param test_img_dir: str of img dir
    :param use_shuffle: use random.shuffle to shuffle the list or not
    :return: list of img_path

    """

    res = []
    file_format = ['jpg', 'png', 'jpeg', 'gif', 'tiff']
    for root, dirs, files in os.walk(test_img_dir, topdown=True):
        if only_root and test_img_dir != root:
            continue
        print(root, dirs, files)
        files = [i for i in files if any([j in i for j in file_format])]
        if use_shuffle:
            shuffle(files)
        for file in files:
            file_path = os.path.join(root, file)
            res.append(file_path)
    return res


if __name__ == '__main__':
    imgs_list = from_dir_get_imgs_list(img_dir)
    for i in tqdm(imgs_list):
        img_np = np.array(Image.open(i)).astype(np.uint8).transpose()
        Image.fromarray(img_np).save(i)
    print(imgs_list)
