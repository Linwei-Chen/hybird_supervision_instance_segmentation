import sys

sys.path.append('../')
import numpy as np
from tqdm import tqdm
from train_utils import get_device
import torch.nn.functional as F
from PIL import Image
import torch
import math
import gc


# torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None)

def validation(args, model, val_loader, multi_scale=False, flip=False, mode='ce'):
    # To solve the problem: received 0 items of ancdata
    torch.multiprocessing.set_sharing_strategy('file_system')

    def scale_with_strde(size: tuple, scale: float, stride=32):
        assert len(size) == 2
        size = [int(math.ceil(i * scale / stride) * stride) for i in size]
        return size

    def modify_out(out: torch.Tensor, mode):
        if mode is None:
            return out
        elif mode == 'ce':
            return out.softmax(dim=1)
        elif mode == 'bce':
            return out.sigmoid()
        else:
            raise ValueError(f'*** wrong mode!')

    scales = [0.5, 0.75, 1.25, 1.5, 1.75]
    model.eval()
    device = get_device(args)
    label_preds = []
    label_targets = []
    with torch.no_grad():
        for step, (img, target, bbox_target) in tqdm(enumerate(val_loader)):

            N, c, h, w = target.shape

            img = img.to(device)
            out = modify_out(model(img, bbox_target), mode)
            outs = F.interpolate(out, (h, w), mode='bilinear', align_corners=True).data.cpu().numpy()
            # del out
            target = target.cpu().numpy().reshape(N, h, w)
            if multi_scale:
                for scale in scales:
                    img_temp = F.interpolate(img, size=scale_with_strde((h, w), scale), mode='bilinear',
                                             align_corners=True)
                    bbox_target_temp = F.interpolate(bbox_target, size=scale_with_strde((h, w), scale), mode='nearest',
                                                     align_corners=True)
                    # print(f'scale img.shape:{img.shape}')
                    out = F.interpolate(modify_out(model(img_temp, bbox_target_temp), mode), size=(h, w),
                                        mode='bilinear', align_corners=True)
                    # del img_temp
                    # print(f'flip img.shape:{out.shape}')
                    outs += out.data.cpu().numpy()
                    # del out
                pass
            if flip:
                img_temp = torch.flip(img, dims=[3])
                bbox_target_temp = torch.flip(bbox_target, dims=[3])
                # print(f'flip img.shape:{img.shape}')
                out = F.interpolate(torch.flip(modify_out(model(img_temp, bbox_target_temp), mode), dims=[3]),
                                    size=(h, w), mode='bilinear', align_corners=True)
                # del img_temp
                # print(f'flip img.shape:{out.shape}')
                outs += out.data.cpu().numpy()
                # del out
                pass

            # print(target[target > 0])
            pred = outs.transpose(0, 2, 3, 1).reshape(-1, args.n_class).argmax(axis=1).reshape(N, h, w)
            label_preds.append(pred)
            label_targets.append(target)
            del img, outs
            gc.collect()
            # break
    acc, acc_cls, mean_iu, fwavacc, iu = label_accuracy_score(label_targets, label_preds, args.n_class)

    del label_preds, label_targets
    gc.collect()

    print('*** acc:{:.4f}\n'
          '*** acc_cls:{:.4f}\n'
          '*** mean_iu:{:.4f}\n'
          '*** fwavacc:{:.4f}\n'.format(acc, acc_cls, mean_iu, fwavacc))
    for _, i in enumerate(iu, 0):
        print(f'{_}:{i:.4f}')
    return acc, acc_cls, mean_iu, fwavacc, iu


def _fast_hist(label_true, label_pred, n_class):
    """

    # 我们可以看到x中最大的数为7，因此bin的数量为8，那么它的索引值为0->7
    x = np.array([0, 1, 1, 3, 2, 1, 7])
    # 索引0出现了1次，索引1出现了3次......索引5出现了0次......
    np.bincount(x)
    #因此，输出结果为：array([1, 3, 1, 1, 0, 0, 0, 1])

    :param label_true: [h, w]
    :param label_pred:[h, w]
    :param n_class:
    :return:
    """
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    # np.diag 提取对角数值
    # acc 像素精确度
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, list(iu)


if __name__ == '__main__':
    from model import get_model
    from logger import ModelSaver
    from train_utils import config, get_device
    from dataset.voc_dataset import get_voc_data_loader

    args = config()

    device = get_device(args)
    load_path = '../deeplabv3plusxception.pkl'
    # load_path = '/media/charles/STORAGE/code/20190717FCN-Pytorch/fcn8sberkeley/fcn8sberkeley_miou_0.5858_pa_0.9100.pkl'
    # load_path = '/media/charles/STORAGE/code/20190717FCN-Pytorch/fcn2sresnet101/fcn2sresnet101_miou_0.7014_pa_0.9340.pkl'
    model = get_model(name='deeplabv3plusxception', n_class=21)
    # model = get_model(name='fcn2sresnet101', n_class=21)
    # model = get_model(name='fcn8sberkeley', n_class=21)
    ModelSaver.load_tool(load_path=load_path, model=model)
    model = model.to(device)
    model.eval()

    # voc2011 = get_voc_data_loader(args, year='2011', train=False)
    voc2011 = get_voc_data_loader(args, sub_mean=True, stdize=True, year='2012', train=False)
    # for step, (imgs, targets) in enumerate(voc2011, start=1):
    #     model
    #     pass
    # validation(args, model, val_loader=voc2011, multi_scale=True, flip=True)
    # validation(args, model, val_loader=voc2011, multi_scale=True, flip=True, mode='bce')
    validation(args, model, val_loader=voc2011, multi_scale=False, flip=False, mode='bce')
