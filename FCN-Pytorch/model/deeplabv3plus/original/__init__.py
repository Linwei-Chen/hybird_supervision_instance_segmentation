class Configuration():
    def __init__(self):
        self.MODEL_NAME = 'deeplabv3plus'
        self.MODEL_BACKBONE = 'xception'
        self.MODEL_OUTPUT_STRIDE = 16
        self.MODEL_ASPP_OUTDIM = 256
        self.MODEL_SHORTCUT_DIM = 48
        self.MODEL_SHORTCUT_KERNEL = 1
        self.MODEL_NUM_CLASSES = 21
        self.TRAIN_BN_MOM = 0.0003


def get_deeplabv3plus_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.deeplabv3plus import deeplabv3plus
    if name.lower() == 'deeplabv3plusxception-8os':
        cfg.MODEL_OUTPUT_STRIDE = 8
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusxception' or name.lower() == 'deeplabv3plusxception-16os':
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusatrousresnset101-8os':
        cfg.MODEL_BACKBONE = 'res101_atrous'
        cfg.MODEL_OUTPUT_STRIDE = 8
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusatrousresnset101' or name.lower() == 'deeplabv3plusatrousresnset101-16os':
        cfg.MODEL_BACKBONE = 'res101_atrous'
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusatrousresnset152-8os':
        cfg.MODEL_BACKBONE = 'res152_atrous'
        cfg.MODEL_OUTPUT_STRIDE = 8
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusatrousresnset152' or name.lower() == 'deeplabv3plusatrousresnset152-16os':
        cfg.MODEL_BACKBONE = 'res152_atrous'
        return deeplabv3plus(cfg)
    else:
        raise Exception(f'*** model name wrong, {name} not legal, '
                        f'please choose from fcn32s, fcn16s, fcn8s, fcns')


def get_deeplab_denseaspp_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.MultiDenseASPP import deeplabv3plusdenseaspp
    if name == 'deeplabv3plusdenseaspp':
        return deeplabv3plusdenseaspp(cfg)

    raise Exception(f'*** model name wrong, {name} not legal')


def get_deeplabv3plusCA_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.deeplabv3plusCA import deeplabv3plusCA
    if name == 'deeplabv3plusca':
        return deeplabv3plusCA(cfg)

    raise Exception(f'*** model name wrong, {name} not legal')


def get_deeplabv3plusdeconv_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from model.deeplabv3plus.deeplabv3plusDeconv import deeplabv3plusdeconv
    if name == 'deeplabv3plusdeconv':
        return deeplabv3plusdeconv(cfg)

    raise Exception(f'*** model name wrong, {name} not legal')


if __name__ == '__main__':
    name_list = ['deeplabv3plusxception-8os', 'deeplabv3plusxception', 'deeplabv3plusxception-16os',
                 'deeplabv3plusatrousresnset101-8os', 'deeplabv3plusatrousresnset101-16os',
                 'deeplabv3plusatrousresnset101',
                 'deeplabv3plusatrousresnset152', 'deeplabv3plusatrousresnset152-16os',
                 'deeplabv3plusatrousresnset152-8s']
    for name in name_list:
        model = get_deeplabv3plus_model(name=name, n_class=21)
        print(model)
