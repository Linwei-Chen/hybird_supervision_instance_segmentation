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

    def predict_one_img(self, img_path, resize=None):
        img, img_tensor, (w, h) = self.from_PIL_path_to_Tensor(img_path, resize)
        img_tensor = img_tensor.unsqueeze(dim=0)
        img_tensor = img_tensor.to(self.device)
        print(f'img_tensor:{img_tensor.size()}')

        self.model.eval()
        self.model = self.model.to(self.device)

        pred = self.model(img_tensor)  # shape[1, n_class, h, w]
        pred = torch.sigmoid(pred)

        print(f'pred.size:{pred.size()}')
        print(f'pred:{pred}')
        pred = pred.argmax(dim=1, keepdim=True)  # shape[1, 1, h, w]
        pred = pred.squeeze()  # shape[ h, w]
        print(f'pred size:{pred.size()}\npred:{pred}')
        pred_img = self.from_pred_to_color_img(pred)
        pred_img = pred_img.resize((w, h))
        pred_img.show()
        merge_img = self.merge_two_img(img, pred_img)
        merge_img.show()

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


if __name__ == '__main__':
    from train_utils import config, get_device

    args = config()
    device = get_device(args)
    # load_path = '/Users/chenlinwei/code/20190717FCN-Pytorch/fcn8sberkeley/checkpoint.pth.tar'
    # load_path = '../deeplabv3plusxception.pkl'
    # load_path = '/Users/chenlinwei/code/20190717FCN-Pytorch/deeplabv3plusad/deeplabv3plusad.pkl'
    load_path = '/Users/chenlinwei/code/20190717FCN-Pytorch/deeplabv3plusxception/deeplabv3plusxception_miou_0.7955_pa_0.9542.pkl'
    # load_path = '/Users/chenlinwei/code/20190717FCN-Pytorch/deeplabv3plusdenseaspp/deeplabv3plusdenseaspp_miou_0.7421_pa_0.9413.pkl'
    # load_path = '/Users/chenlinwei/code/20190717FCN-Pytorch/deeplabv3pluspsall/deeplabv3pluspsallv2.2.pkl'
    # load_path = '/Users/chenlinwei/code/20190717FCN-Pytorch/deeplabv3pluspsall/deeplabv3pluspsallv2.1.pkl'
    # load_path = '/Users/chenlinwei/code/20190717FCN-Pytorch/deeplabv3pluspsall/deeplabv3pluspsallv3.pkl'
    # load_path = '/Users/chenlinwei/code/20190717FCN-Pytorch/deeplabv3plusad/deeplabv3plusad_miou_0.7843_pa_0.9526.pkl'
    # load_path = '/Users/chenlinwei/code/20190717FCN-Pytorch/deeplabv3plusdenseaspp/deeplabv3plusbdasppthick256_miou_0.7710_pa_0.9493.pkl'
    # load_path = '/Users/chenlinwei/code/20190717FCN-Pytorch/kc/deeplabv3pluskcaspp_miou_0.1089_pa_0.7292.pkl'
    # load_path = '/Users/chenlinwei/code/20190717FCN-Pytorch/kc/deeplabv3pluskcaspp.pkl'
    # load_path = '/Users/chenlinwei/code/20190717FCN-Pytorch/deeplabv3plusrf/deeplabv3plusrf_miou_0.5345_pa_0.8588.pkl'
    model = get_model(name='deeplabv3plusxception', n_class=21)
    # model = get_model(name='deeplabv3plusxception', n_class=21)
    # model = get_model(name='deeplabv3plusdenseaspp', n_class=21)
    # model = get_model(name='deeplabv3pluspsallv2.1', n_class=21)
    # model = get_model(name='deeplabv3plusad', n_class=21)
    # model = get_model(name='deeplabv3plusbdasppthick256', n_class=21)
    # model = get_model(name='deeplabv3pluskcaspp', n_class=21)
    model = nn.DataParallel(model)
    # model = get_model(name='fcn8satonceberkeley', n_class=21)
    ModelSaver.load_tool(load_path=load_path, key=None, model=model)
    # torch.save(model.state_dict(), 'deeplabv3plusxception.pkl')
    # ModelSaver.load_tool(load_path=load_path, key='model_state_dict', model=model)
    model = model.to(device)
    model.eval()
    # print(model)
    # predictor = Predict(model, sub_mean=False,stdize=False)
    predictor = Predict(model, sub_mean=True, stdize=True)

    img_dir = '../test_pic'
    img_list = from_dir_get_imgs_list(test_img_dir=img_dir)
    for img_path in img_list:
        predictor.predict_one_img(img_path)
    pass
