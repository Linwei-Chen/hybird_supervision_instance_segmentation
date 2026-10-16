from model.hrnet.config import config
from yacs.config import CfgNode as CN


def hr48_config():
    c = CN()
    c.MODEL = CN()
    cm = c.MODEL
    cm.NAME = 'cls_hrnet'
    cm.EXTRA = CN()
    cm.EXTRA.FINAL_CONV_KERNEL = 1
    cm.EXTRA.STAGE2 = CN()

    cm.EXTRA.STAGE2.NUM_MODULES = 1
    cm.EXTRA.STAGE2.NUM_BRANCHES = 2
    cm.EXTRA.STAGE2.BLOCK = 'BASIC'
    cm.EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
    cm.EXTRA.STAGE2.NUM_CHANNELS = [48, 96]
    cm.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

    cm.EXTRA.STAGE3 = CN()
    cm.EXTRA.STAGE3.NUM_MODULES = 4
    cm.EXTRA.STAGE3.NUM_BRANCHES = 3
    cm.EXTRA.STAGE3.BLOCK = 'BASIC'
    cm.EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
    cm.EXTRA.STAGE3.NUM_CHANNELS = [48, 96, 192]
    cm.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

    cm.EXTRA.STAGE4 = CN()
    cm.EXTRA.STAGE4.NUM_MODULES = 3
    cm.EXTRA.STAGE4.NUM_BRANCHES = 4
    cm.EXTRA.STAGE4.BLOCK = 'BASIC'
    cm.EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
    cm.EXTRA.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
    cm.EXTRA.STAGE4.FUSE_METHOD = 'SUM'
    config.merge_from_other_cfg(c)
    return config

#
# config.CUDNN:
#   BENCHMARK=true
#   DETERMINISTIC=false
#   ENABLED=true
# DATASET:
#   DATASET='imagenet'
#   DATA_FORMAT='jpg'
#   ROOT='data/imagenet/'
#   TEST_SET='val'
#   TRAIN_SET='train'
# TEST:
#   BATCH_SIZE_PER_GPU=32
#   MODEL_FILE=''
# TRAIN:
#   BATCH_SIZE_PER_GPU=32
#   BEGIN_EPOCH=0
#   END_EPOCH=100
#   RESUME=true
#   LR_FACTOR=0.1
#   LR_STEP:
#   - 30
#   - 60
#   - 90
#   OPTIMIZER=sgd
#   LR=0.05
#   WD=0.0001
#   MOMENTUM=0.9
#   NESTEROV=true
#   SHUFFLE=true
# DEBUG:
#   DEBUG=false
