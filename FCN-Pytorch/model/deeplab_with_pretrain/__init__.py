from .DeepLab_v3_plus import DeepLabV3Plus
from .DeepLab_v3 import DeepLabv3


def get_deeplab_model(name: str, n_class=21):
    if name.lower() == 'deeplabv3':
        return DeepLabv3(num_classes=n_class, small=True, pretrained=True)
    elif name.lower() == 'deeplabv3plus':
        return DeepLabV3Plus(num_classes=n_class, small=True, pretrained=True)
    else:
        raise Exception(f'*** model name wrong, {name} not legal, '
                        f'please choose from fcn32s, fcn16s, fcn8s, fcns')
