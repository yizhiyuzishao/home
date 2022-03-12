#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import torch
from torch import nn
from torch import tensor
import logging
import os
from collections import OrderedDict
import torch.nn.functional as F
import cv2
import detectron2.utils.comm as comm
from detectron2.engine.defaults import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets.coco import load_coco_json
import pycocotools
from detectron2.data import DatasetCatalog, MetadataCatalog
# DS path
# CLASS_NAMES =[]
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm
DATASET_ROOT = '/mnt/GZY/GZY1/detectron2/data/datasets/coco'
ANN_ROOT = os.path.join(DATASET_ROOT , 'annontations')

TRAIN_PATH = os.path.join(DATASET_ROOT, 'images')
VAL_PATH = os.path.join(DATASET_ROOT, 'images')
TRAIN_JSON = os.path.join(ANN_ROOT, 'train2022.json')
VAL_JSON = os.path.join(ANN_ROOT, 'test2022.json')

# DS data subset
PREDEFINED_SPLITS_DATASET = {
    "coco_train":(TRAIN_PATH, TRAIN_JSON),
    "coco_test":(VAL_PATH,VAL_JSON),
}
# register dataset
def plain_register_dataset():
    # register dataset to dataset catalog,register metadata to metadata catalog and set attribute
    DatasetCatalog.register("coco_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
    MetadataCatalog.get("coco_train").set(json_file=TRAIN_JSON, image_root=TRAIN_PATH,evaluator_type="coco")
    DatasetCatalog.register("coco_val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
    MetadataCatalog.get("coco_val").set(json_file=VAL_JSON, image_root=VAL_PATH,evaluator_type="coco")


def check_dataset_annotation(name="coco_val"):
    dataset_dicts = load_coco_json(TRAIN_JSON,TRAIN_PATH)
    print(len(dataset_dicts))
    for i,d in enumerate(dataset_dicts, 0):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata = MetadataCatalog.get(name), scale=1.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite('out/'+str(i) + '.jpg',vis.get_image()[:, :, ::-1])
        if i == 200:
            break


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    def build_evaluator(cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)



    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    args.config_file = "/mnt/GZY/GZY1/detectron2-main/configs/quick_schedules/retinanet_R_50_FPN_instant_test.yaml"
    cfg.DATASETS.TRAIN = ("coco_train",)
    cfg.DATASETS.TEST = ("coco_val",)
    cfg.DATALOADER.NUM_WORKERS = 10

    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.MAX_SIZE_TRAIN = 640 # train images
    cfg.INPUT.MAX_SIZE_TEST = 640 # test images
    cfg.INPUT.MIN_SIZE_TRAIN = (512, 768)
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range'

    cfg.MODEL.RETINANET.NUM_CLASSES = 8 # categories
    cfg.MODEL.WEIGHTS ="/mnt/GZY/GZY1/detectron2/tools/R-50.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 4
    batch_size=2;iter_in_one_epoch = batch_size
    ITERS_IN_ONE_EPOCH = int(300 / cfg.SOLVER.IMS_PER_BATCH)

    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 12) - 1 # 12EPOCHS
    cfg.SOLVER.BASE_LR = 0.002 # initial learning rate
    cfg.SOLVER.MOMENTUM = 0.9 # optimizer
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM =0.0 # weight
    cfg.SOLVER.GAMMA = 0.1 # learning rate decay rates
    cfg.SOLVER.STEPS = (7000,)
    cfg.SOLVER.WARMUP_FACTOR = 1.0/1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1

    cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH
  #  cfg.TEST.EVAL_PERIOD = 100'''

    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    plain_register_dataset()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
