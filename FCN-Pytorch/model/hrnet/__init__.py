from model.hrnet.models.cls_hrnet import get_hrnet_basebone

if __name__ == '__main__':
    get_hrnet_basebone(21, pretrain=False)
