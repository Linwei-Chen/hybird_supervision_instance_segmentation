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
        self.REFINE = False


def get_anc_model(name: str, n_class: int):
    if 'deeplabv3plus' in name.lower():
        return get_deeplabv3plusanc_model(name, n_class)
    else:
        raise Exception(f'*** model name wrong, {name} not legal')


def get_self_learning_model(name: str, n_class: int):
    if 'deeplabv3plus' in name.lower():
        return get_deeplabv3plus_selflearning_model(name, n_class)
    else:
        raise Exception(f'*** model name wrong, {name} not legal')


def get_anc_fore_back_model(name: str, n_class: int):
    if 'deeplabv3plus' in name.lower():
        return get_deeplabv3plusancforeback_model(name, n_class)
    else:
        raise Exception(f'*** model name wrong, {name} not legal')


def get_deeplabv3plusanc_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from weakly_semi_supseg.weakly_semi_model.deeplabv3plusanc import deeplabv3plus, deeplabv3pluslabel
    if name.lower() == 'deeplabv3plusxception-8os':
        cfg.MODEL_OUTPUT_STRIDE = 8
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusxception' or name.lower() == 'deeplabv3plusxception-16os':
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusxceptionlabel' or name.lower() == 'deeplabv3plusxceptionlabel-16os':
        return deeplabv3pluslabel(cfg)

    elif name.lower() == 'deeplabv3plusxceptionrefine':
        cfg.REFINE = True
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusxceptionlabelrefine':
        cfg.REFINE = True
        return deeplabv3pluslabel(cfg)

    elif name.lower() == 'deeplabv3plusxceptionthick' or name.lower() == 'deeplabv3plusthick':
        cfg.MODEL_ASPP_OUTDIM = 512
        cfg.MODEL_SHORTCUT_DIM = 96
        return deeplabv3plus(cfg)
    else:
        raise Exception(f'*** model name wrong, {name} not legal')


def get_deeplabv3plus_selflearning_model(name: str, n_class: int):
    cfg = Configuration()
    cfg.MODEL_NUM_CLASSES = n_class
    from weakly_semi_supseg.weakly_semi_model.deeplabv3plusselflearning import deeplabv3plus, deeplabv3pluslabel,\
        deeplabv3plus10kbbox,deeplabv3plusnolabel, deeplabv3pluslabelrefinelabel, deeplabv3pluslabelDIS
    if name.lower() == 'deeplabv3plusxception-8os':
        cfg.MODEL_OUTPUT_STRIDE = 8
        return deeplabv3plus(cfg)

    elif name.lower() == 'deeplabv3plusxception' or name.lower() == 'deeplabv3plusxception-16os':
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusxceptionlabel' or name.lower() == 'deeplabv3plusxceptionlabel-16os':
        return deeplabv3pluslabel(cfg)
    elif name.lower() == 'deeplabv3pluslabeldis':
        return deeplabv3pluslabelDIS(cfg)

    elif name.lower() == 'deeplabv3plusxceptionnolabel':
        return deeplabv3plusnolabel(cfg)
    elif name.lower() == 'deeplabv3plus10kbbox':
        return deeplabv3plus10kbbox(cfg)
    
    elif name.lower() == 'deeplabv3pluslabelrefinelabel':
        return deeplabv3pluslabelrefinelabel(cfg)

    # elif name.lower() == 'deeplabv3plusxceptionrefine':
    #     cfg.REFINE = True
    #     return deeplabv3plus(cfg)
    # elif name.lower() == 'deeplabv3plusxceptionlabelrefine':
    #     cfg.REFINE = True
    #     return deeplabv3pluslabel(cfg)

    elif name.lower() == 'deeplabv3plusxceptionthick' or name.lower() == 'deeplabv3plusthick':
        cfg.MODEL_ASPP_OUTDIM = 512
        cfg.MODEL_SHORTCUT_DIM = 96
        return deeplabv3plus(cfg)
    else:
        raise Exception(f'*** model name wrong, {name} not legal')


def get_deeplabv3plusancforeback_model(name: str, n_class: int):
    cfg = Configuration()
    assert n_class == 2
    cfg.MODEL_NUM_CLASSES = n_class
    from weakly_semi_supseg.weakly_semi_model.deeplabv3plusancforeback import deeplabv3plus
    if name.lower() == 'deeplabv3plusxception-8os':
        cfg.MODEL_OUTPUT_STRIDE = 8
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusxception' or name.lower() == 'deeplabv3plusxception-16os':
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusxceptionthick' or name.lower() == 'deeplabv3plusthick':
        cfg.MODEL_ASPP_OUTDIM = 512
        cfg.MODEL_SHORTCUT_DIM = 96
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusatrousresnet101-8os':
        cfg.MODEL_BACKBONE = 'res101_atrous'
        cfg.MODEL_OUTPUT_STRIDE = 8
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusatrousresnet101' or name.lower() == 'deeplabv3plusatrousresnet101-16os':
        cfg.MODEL_BACKBONE = 'res101_atrous'
        return deeplabv3plus(cfg)

    elif name.lower() == 'deeplabv3plusatrousresnext101-8os':
        cfg.MODEL_BACKBONE = 'resnext101_atrous'
        cfg.MODEL_OUTPUT_STRIDE = 8
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusatrousresnext101' or name.lower() == 'deeplabv3plusatrousresnext101-16os':
        cfg.MODEL_BACKBONE = 'resnext101_atrous'
        return deeplabv3plus(cfg)

    elif name.lower() == 'deeplabv3plusatrousresnet152-8os':
        cfg.MODEL_BACKBONE = 'res152_atrous'
        cfg.MODEL_OUTPUT_STRIDE = 8
        return deeplabv3plus(cfg)
    elif name.lower() == 'deeplabv3plusatrousresnet152' or name.lower() == 'deeplabv3plusatrousresnet152-16os':
        cfg.MODEL_BACKBONE = 'res152_atrous'
        return deeplabv3plus(cfg)
    else:
        raise Exception(f'*** model name wrong, {name} not legal')
