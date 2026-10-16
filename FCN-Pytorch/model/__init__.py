from model.fcn import get_fcn_model
from model.deeplab_with_pretrain import get_deeplab_model
from model.fcn_from_berkeley import get_fcn_berkeley_model
from model.fcn_resnet import get_fcn_resnet_model
from model.deeplabv3plus import get_deeplabv3plus_model, get_deeplab_multidenseaspp_model, \
    get_deeplabv3plusCA_model, get_deeplabv3plusdeconv_model, get_deeplabv3plusbalanceddenseaspp_model, \
    get_deeplabv3plusAD_model, get_deeplabv3plusPS_model, get_deeplabv3plusJPU_model, get_deeplabv3plusKCASPP_model, \
    get_tkcnet_model, get_danet_model, get_deeplabv3pluspspaspp_model, get_deeplabv3plusHR_model, \
    get_deeplabv3plusRF_model, get_deeplabv3plusTriAtt_model, \
    get_EncNet_model, get_deeplabv3plusgff_model, get_deeplabv3plusupsampleandcls_model, \
    get_deeplabv3plusDUsampling_model, get_deeplabv3plusExFuse_model

from model.erfnet import  get_erfnet_model


def get_model(name: str, n_class: int):
    if 'resnet' in name:
        model = get_fcn_resnet_model(name, n_class)

    elif 'erfnet' in name:
        model = get_erfnet_model(name, n_class)

    elif 'deeplabv3plusexfuse' in name:
        model = get_deeplabv3plusExFuse_model(name, n_class)
    elif 'deeplabv3plusupsampleandcls' in name:
        model = get_deeplabv3plusupsampleandcls_model(name, n_class)
    elif 'deeplabv3plustriatt' in name:
        model = get_deeplabv3plusTriAtt_model(name, n_class)

    elif 'deeplabv3plusdupsample' in name:
        model = get_deeplabv3plusDUsampling_model(name, n_class)

    elif 'deeplabv3plushr' in name:
        model = get_deeplabv3plusHR_model(name, n_class)
    elif 'deeplabv3plusrf' in name:
        model = get_deeplabv3plusRF_model(name, n_class)
    elif 'deeplabv3plusdeconv' in name:
        model = get_deeplabv3plusdeconv_model(name, n_class)
    elif 'deeplabv3pluspspaspp' in name:
        model = get_deeplabv3pluspspaspp_model(name, n_class)
    elif 'deeplabv3plusps' in name:
        model = get_deeplabv3plusPS_model(name, n_class)
    elif 'deeplabv3plusjpu' in name:
        model = get_deeplabv3plusJPU_model(name, n_class)
    elif 'deeplabv3plusca' in name:
        model = get_deeplabv3plusCA_model(name, n_class)
    elif 'deeplabv3plusad' in name:
        model = get_deeplabv3plusAD_model(name, n_class)
    elif 'deeplabv3pluskcaspp' in name:
        model = get_deeplabv3plusKCASPP_model(name, n_class)
    elif 'deeplabv3plusbdaspp' in name:
        model = get_deeplabv3plusbalanceddenseaspp_model(name, n_class)
    elif 'multidenseaspp' in name:
        model = get_deeplab_multidenseaspp_model(name, n_class)

    elif 'enc' in name:
        model = get_EncNet_model(name, n_class)
    elif 'danet' in name:
        model = get_danet_model(name, n_class)
    elif 'tkc' in name:
        model = get_tkcnet_model(name, n_class)
    elif 'gff' in name:
        model = get_deeplabv3plusgff_model(name, n_class)

    elif 'deeplabv3plus' in name:
        model = get_deeplabv3plus_model(name, n_class)
    elif 'deeplab' in name:
        model = get_deeplab_model(name, n_class)
    elif 'berkeley' in name:
        model = get_fcn_berkeley_model(name, n_class)
    elif 'fcn' in name:
        model = get_fcn_model(name, n_class)
    else:
        raise Exception(f'*** model name {name} not support')
    print(f'===> model name: {model.__class__.__name__}')
    return model
