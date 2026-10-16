from model.fcn_from_berkeley.fcn32s import FCN32sBerkeley
from model.fcn_from_berkeley.fcn16s import FCN16sBerkeley
from model.fcn_from_berkeley.fcn8s import FCN8sBerkeley, FCN8sAtOnceBerkeley
from model.fcn_from_berkeley.vgg import VGGNet


# 参考：
# https://github.com/shelhamer/fcn.berkeleyvision.org
def get_fcn_berkeley_model(name: str, n_class=21):
    if name.lower() == 'fcn32sberkeley':
        return FCN32sBerkeley(pretrained_net=VGGNet(), n_class=n_class)
    elif name.lower() == 'fcn16sberkeley':
        return FCN16sBerkeley(pretrained_net=VGGNet(), n_class=n_class)
    elif name.lower() == 'fcn8sberkeley':
        return FCN8sBerkeley(pretrained_net=VGGNet(), n_class=n_class)
    elif name.lower() == 'fcn8satonceberkeley':
        return FCN8sAtOnceBerkeley(pretrained_net=VGGNet(), n_class=n_class)
    else:
        raise Exception(f'*** model name wrong, {name} not legal, '
                        f'please choose from fcn32s, fcn16s, fcn8s, fcns')
