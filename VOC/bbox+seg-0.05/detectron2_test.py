import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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


def predict_test():
    im = cv2.imread("/home/charles/Documents/test.png")
    cv2.imshow('temp', im)
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    print(outputs)
    # outputs["instances"].pred_classes
    # outputs["instances"].pred_boxe
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('result', v.get_image()[:, :, ::-1])
    cv2.waitKey()


def model_build_test():
    im = cv2.imread("/home/charles/Documents/test.png")
    cv2.imshow('temp', im)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    model = build_model(cfg)
    print(model)

    cfg.MODEL.WEIGHTS = '~/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth'
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('result', v.get_image()[:, :, ::-1])
    cv2.waitKey()

    # DetectionCheckpointer(model, osp.expanduser('~/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth'))

    pass


def train(cfg):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    return trainer.train()


# prepare dataset
# 1. train set
voc2012_path = '~/dataset/VOCdevkit/VOC2012/'
voc2012_train_img_path = osp.join(voc2012_path, './JPEGImages')
voc2012_train_json_path = osp.join(voc2012_path, './voc2012_train_529.json')
register_coco_instances("voc2012_train", {},
                        json_file=osp.expanduser(voc2012_train_json_path),
                        image_root=osp.expanduser(voc2012_train_img_path))

voc2012_train_gen_json_path = osp.join(voc2012_path, './voc2012_train_empty_mask.json')
register_coco_instances("voc2012_train_bbox", {},
                        json_file=osp.expanduser(voc2012_train_gen_json_path),
                        image_root=osp.expanduser(voc2012_train_img_path))


voc2012_val_json_path = osp.join(voc2012_path, './voc2012_val.json')
register_coco_instances("voc2012_val", {},
                        json_file=osp.expanduser(voc2012_val_json_path),
                        image_root=osp.expanduser(voc2012_train_img_path))

sbd_path = '~/dataset/SBD_FULL11355/dataset'
SBD_train_img_path = osp.join(sbd_path, './img')
#####
SBD_train_json_path = osp.join(sbd_path, './sbd_train_empty_mask.json')
register_coco_instances("sbd_train_bbox", {},
                        json_file=osp.expanduser(SBD_train_json_path),
                        image_root=osp.expanduser(SBD_train_img_path))


def train_and_eval():
    cfg = get_cfg()
    cfg.OUTPUT_DIR = './'
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ["voc2012_train" for _ in range(20)] + ["voc2012_train_bbox", "sbd_train_bbox"]
    # cfg.DATASETS.TRAIN = ["voc2012_train", ]
    cfg.DATASETS.TEST = ("voc2012_val",)
    cfg.DATALOADER.NUM_WORKERS = 20
    # Default
    # MODEL:
    #     WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-101.pkl"
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #     "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
    cfg.SOLVER.MAX_ITER = 24000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.SOLVER.STEPS = (16000,)
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # only has one class (ballon)
    cfg.VIS_PERIOD = 0
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
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

    cfg.SOLVER.MAX_ITER = 25000
    cfg.DATASETS.TRAIN = ("voc2012_train",)
    print("Command Line Args:", args)
    launch(
        train,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(cfg,),
    )

    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)

    evaluator = COCOEvaluator("voc2012_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "voc2012_val")
    inference_on_dataset(trainer.model, val_loader, evaluator)


if __name__ == '__main__':
    # model_build_test()
    train_and_eval()
    pass
