import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
print(os.environ['CUDA_VISIBLE_DEVICES'])

import torch

print(torch.cuda.device_count())

import torch.distributed as dist

import numpy as np
import cv2
import random
import os.path as osp

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, launch, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.engine.defaults import default_argument_parser


def train(cfg):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    return trainer.train()


# prepare dataset
# 1. train set
cityscapes_path = '~/dataset/cityscapes/'
cityscapes_train_img_path = osp.join(cityscapes_path, './leftImg8bit/train')
cityscapes_train_914_json_path = osp.join(cityscapes_path, './cityscapes_train_148.json')
register_coco_instances("cityscapes_train_148", {},
                        json_file=osp.expanduser(cityscapes_train_914_json_path),
                        image_root=osp.expanduser(cityscapes_train_img_path))

cityscapes_train_gen_2061_json_path = osp.join(cityscapes_path, './cityscapes_gen_mask_0.05.json')
register_coco_instances("cityscapes_train_gen_mask_0.05", {},
                        json_file=osp.expanduser(cityscapes_train_gen_2061_json_path),
                        image_root=osp.expanduser(cityscapes_train_img_path))

cityscapes_val_img_path = osp.join(cityscapes_path, './leftImg8bit/val')
cityscapes_val_json_path = osp.join(cityscapes_path, './cityscapes_val.json')
register_coco_instances("cityscapes_val", {},
                        json_file=osp.expanduser(cityscapes_val_json_path),
                        image_root=osp.expanduser(cityscapes_val_img_path))
MetadataCatalog.get("cityscapes_val").thing_classes = ['person',
                                                       'rider',
                                                       'car',
                                                       'truck',
                                                       'bus',
                                                       'train',
                                                       'motorcycle',
                                                       'bicycle', ]
MetadataCatalog.get("cityscapes_val").gt_dir = osp.expanduser('~/dataset/cityscapes/gtFine/val')


def train_and_eval():
    cfg = get_cfg()
    cfg.OUTPUT_DIR = './'
    cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    # cfg.DATASETS.TRAIN = ["cityscapes_train_148" for _ in range(19)] + ["cityscapes_train_gen_mask_0.05"]
    cfg.DATASETS.TRAIN = ["cityscapes_train_148", ]
    cfg.DATASETS.TEST = ("cityscapes_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    # Default
    # MODEL:
    #     WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-101.pkl"
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #     "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = 'detectron2://ImageNetPretrained/MSRA/R-50.pkl'
    cfg.SOLVER.IMS_PER_BATCH = 8
    # cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
    cfg.SOLVER.MAX_ITER = 26000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    # cfg.SOLVER.STEPS = (14000, )
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # only has one class (ballon)
    cfg.VIS_PERIOD = 10
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    args = default_argument_parser().parse_args()
    args.num_gpus = torch.cuda.device_count()
    args.dist_url = f'tcp://127.0.0.1:{random.randint(0, 65500)}'
    print("Command Line Args:", args)
    launch(
        train,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(cfg,),
    )

    # cfg.SOLVER.MAX_ITER = 26000
    # cfg.DATASETS.TRAIN = ("cityscapes_train_148",)
    # print("Command Line Args:", args)
    # launch(
    #     train,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(cfg,),
    # )

    from detectron2.evaluation import COCOEvaluator, inference_on_dataset, CityscapesInstanceEvaluator
    from detectron2.data import build_detection_test_loader
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    # trainer.resume_or_load(resume=False)
    # evaluator = COCOEvaluator("cityscapes_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    evaluator = CityscapesInstanceEvaluator("cityscapes_val")
    val_loader = build_detection_test_loader(cfg, "cityscapes_val")
    inference_on_dataset(trainer.model, val_loader, evaluator)


if __name__ == '__main__':
    # model_build_test()
    train_and_eval()
    pass
