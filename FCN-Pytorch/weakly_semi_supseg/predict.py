import os
import os.path as osp
import sys

sys.path.append(osp.join(sys.path[0], '../'))
sys.path.append(osp.join(sys.path[0], '../../'))
sys.path.append(osp.join(sys.path[0], '../../../'))
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from tools.imgs_dir_reader import from_dir_get_imgs_list
import numbers
from dataset.voc_dataset import COLORMAP
from logger import ModelSaver
import math
from model import get_model
from dataset.segmentation_transform import MEAN, STD
from train_utils import config, get_device
from weakly_semi_supseg.weakly_semi_model import get_self_learning_model
from dataset.voc_dataset import get_voc_data_loader
from tqdm import tqdm


class Predict:
    def __init__(self, model, stride=32., sub_mean=True, stdize=True, cuda=False):
        self.model = model
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
        self.model = self.model.to(self.device)
        self.color_map = torch.Tensor(COLORMAP)
        self.stride = stride
        self.stride = stride
        self.sub_mean = sub_mean
        self.stdize = stdize

    def predict_one_img(self, img, resize=None):
        # if isinstance(img_path, Image.Image):
        #     pass
        if isinstance(img, str):
            img, img_tensor, (w, h) = self.from_PIL_path_to_Tensor(img, resize)
            img_tensor = img_tensor.unsqueeze(dim=0)
            img_tensor = img_tensor.to(self.device)
            print(f'img_tensor:{img_tensor.size()}')
        elif isinstance(img, torch.Tensor):
            img_tensor = img.to(self.device)
            print(img_tensor.device)
        else:
            raise NotImplementedError

        self.model.eval()
        self.model = self.model.to(self.device)

        with torch.no_grad():
            pred = self.model(img_tensor).detach()  # shape[1, n_class, h, w]
            # pred = pred * torch.randint(low=0, high=2, size=(1, 21, 1, 1)).float()
            try:
                if isinstance(self.model, nn.DataParallel):
                    pre = self.model.module.refine_model(pred, torch.zeros(1, 21, 32, 32).to(self.device))
                else:
                    pre = self.model.refine_model(pred, torch.zeros(1, 21, 32, 32).to(self.device))
                print('refine model', pre)
                probs, index = pre.max(keepdim=True, dim=1)
                print(index[index < 1])
                # transforms.ToPILImage()(index.squeeze(dim=0).float()).show()
            except:
                pass

            # pred = torch.sigmoid(pred)
            print(f'pred.size:{pred.size()}')
            print(f'pred:{pred}')
            probs, pred = pred.max(dim=1, keepdim=True)  # shape[1, 1, h, w]
            # pred[probs < 0.5] = 21
            pred = pred.squeeze()  # shape[ h, w]
            print(f'pred size:{pred.size()}\npred:{pred}')
            pred_img = self.from_pred_to_color_img(pred)
            # pred_img = pred_img.resize((w, h))
            # pred_img.show()
            # merge_img = self.merge_two_img(img, pred_img)
            # merge_img.show()
        return pred_img, None

    def from_PIL_path_to_Tensor(self, img_path, resize):
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        if resize is not None:
            assert isinstance(resize, numbers.Number) or (isinstance(resize, tuple) and len(resize) == 2)

            if isinstance(resize, numbers.Number):
                resize = (resize, resize)
        else:
            w_stride, h_stride = math.ceil(w / self.stride), math.ceil(h / self.stride)
            resize = (int(w_stride * self.stride), int(h_stride * self.stride))

        img_resized = img.resize(resize)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN if self.sub_mean else [0.] * 3,
                                 std=STD if self.sub_mean else [1.] * 3)
        ])
        img_tensor = transform(img_resized)
        return img, img_tensor, (w, h)

    def from_pred_to_color_img(self, pred):
        h, w = pred.size()
        # pred = pred.expand(3, -1, -1)
        temp = torch.zeros((3, h, w))
        for channel in range(3):
            for color in range(len(self.color_map)):
                temp[channel][pred == color] = self.color_map[color][channel]

        # another way of coloring the img
        # for i in range(h):
        #     for j in range(w):
        #         temp[:, i, j] = self.color_map[pred[i, j]]
        return transforms.ToPILImage()(temp / 255.0)

    @staticmethod
    def merge_two_img(img1, img2):
        return Image.blend(img1, img2, 0.5)


def create_dir(dir_path):
    if not osp.exists(dir_path):
        os.mkdir(dir_path)


def predict_on_test_dir():
    args = config()
    args.voc_data_path = '/Users/chenlinwei/dataset'
    device = get_device(args)
    voc2012 = get_voc_data_loader(args, sub_mean=False, stdize=False, year='2012', train=False)
    # load_path = '/Users/chenlinwei/code/20190717FCN-Pytorch/weakly/deeplabv3plusxception_miou_0.7719_pa_0.9485.pkl'
    # load_path = '/Users/chenlinwei/code/20190717FCN-Pytorch/deeplabv3pluslabelrefinelabel/deeplabv3pluslabelrefinelabel.pkl'
    # load_path = '/Users/chenlinwei/code/20190717FCN-Pytorch/weakly/deeplabv3plusxception_miou_0.7898_pa_0.9528.pkl'
    # load_path = '/Users/chenlinwei/code/20190717FCN-Pytorch/weakly/deeplabv3plusxceptionlabel_miou_0.7871_pa_0.9512.pkl'
    load_path = '/Users/chenlinwei/code/20190717FCN-Pytorch/deeplabv3plusxception_miou_0.7955_pa_0.9542_cpu.pkl'

    # model = get_self_learning_model(name='deeplabv3plusxceptionlabel', n_class=21)
    model = get_self_learning_model(name='deeplabv3pluslabelrefinelabel', n_class=21)
    # model = get_model('deeplabv3plusxception', 21)
    model = nn.DataParallel(model)
    # model.load_state_dict(torch.load(load_path, map_location='cpu'))
    ModelSaver.load_tool(load_path=load_path, model=model)
    # torch.save(model.state_dict(), 'deeplabv3plusxception.pkl')
    # ModelSaver.load_tool(load_path=load_path, key='model_state_dict', model=model)
    model = model.to(device)
    # model.module.normal_eval()
    # print(model)
    # predictor = Predict(model, sub_mean=False,stdize=False)
    predictor = Predict(model, sub_mean=True, stdize=True)

    img_dir = '../../test_pic'
    img_list = from_dir_get_imgs_list(test_img_dir=img_dir)
    for img_path in img_list:
        predictor.predict_one_img(img_path)
    pass


def visualize_and_save(load_path, save_path):
    create_dir(save_path)
    args = config()
    args.device = 1
    args.voc_data_path = '/dataset'
    device = get_device(args)

    val_loader = get_voc_data_loader(args, sub_mean=False, stdize=False, year='2012', train=False)
    model = get_self_learning_model(name='deeplabv3plusxception', n_class=21)
    model = nn.DataParallel(model)
    ModelSaver.load_tool(load_path=load_path, model=model)
    model = model.to(device)
    predictor = Predict(model, sub_mean=True, stdize=True, cuda=True)
    counter = 1
    for step, (img, target) in tqdm(enumerate(val_loader)):
        img = img.to(device)
        pred_img, merge_img = predictor.predict_one_img(img)
        pred_img.save(osp.join(save_path, f'{counter}.png'))
        counter += 1
        if counter > 100:
            break
    pass


if __name__ == '__main__':
    visualize_and_save(
        '/media/charles/STORAGE/code/20190717FCN-Pytorch/deeplabv3plusxception_miou_0.7618_pa_0.9442_105.pkl', './0.01')
    visualize_and_save('/media/charles/STORAGE/code/20190717FCN-Pytorch/deeplabv3plusxception_miou_0.7727_pa_0.9484_317.pkl', './0.03')
    visualize_and_save('/media/charles/STORAGE/code/20190717FCN-Pytorch/deeplabv3plusxception_miou_0.7825_pa_0.9506_529.pkl', './0.05')
    visualize_and_save('/media/charles/STORAGE/code/20190717FCN-Pytorch/deeplabv3plusxception_miou_0.8080_pa_0.9563_1058.pkl', './0.1')
    visualize_and_save('/media/charles/STORAGE/code/20190717FCN-Pytorch/deeplabv3plusxception_miou_0.8087_pa_0.9573_1464.pkl', './0.14')
    visualize_and_save('/media/charles/STORAGE/code/20190717FCN-Pytorch/deeplabv3plusxception_miou_0.8076_pa_0.9571.pkl', './1.00')
    pass
