#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""
from torch import Tensor
import logging
import os
from collections import OrderedDict
import torch
from torch import tensor
from torch.nn.parallel import DistributedDataParallel
import detectron2
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
import json
from detectron2.engine.defaults import DefaultTrainer
from detectron2.demo.predictor import VisualizationDemo
from numpy.core.fromnumeric import squeeze
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from torch.autograd import Variable
from detectron2.structures.boxes import Boxes
from detectron2.engine.defaults import DefaultTrainer
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets.coco import load_coco_json
import torch.nn.functional as F
import math
import cv2
import numpy
from detectron2.engine import HookBase
import torch.nn as nn
logger = logging.getLogger("detectron2")

DATASET_ROOT = '/mnt/GZY/GZY1/detectron2/data/datasets/coco'
ANN_ROOT = os.path.join(DATASET_ROOT , 'annontations')

TRAIN_PATH = os.path.join(DATASET_ROOT, 'images')
VAL_PATH = os.path.join(DATASET_ROOT, 'images')
TRAIN_JSON = os.path.join(ANN_ROOT, 'train2022.json')
VAL_JSON = os.path.join(ANN_ROOT, 'test2022.json')

# DS data subset
PREDEFINED_SPLITS_DATASET = {
    "coco_train": (TRAIN_PATH, TRAIN_JSON),
    "coco_test": (VAL_PATH, VAL_JSON),
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

def get_evaluator(cfg, dataset_name, output_folder=None):
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
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

class BeatCheckPointer(HookBase):
    def __init__(self):
        super().__init__()
    def after_step(self):
        curr_val = self.trainer.storage.latest().get('bbox/AP50', 0)
        if type(curr_val)!= int:
            curr_val = curr_val[0]
            if math.isnan(curr_val):
                curr_val = 0
        try:
            _ = self.trainer.storage.history('max_bbox/AP50')
        except:
            self.trainer.storage.put_scalar('max_bbox/AP50', curr_val)
        max_val = self.trainer.storage.history('max_bbox/AP50')._data[-1][0]
        if curr_val > max_val:
            print("\n%s>%s save!\n" % (curr_val, max_val))
            self.trainer.storage.put_scalar('max_bbox/AP50', curr_val)
            self.trainer.checkpointer.save("model_best")


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, teacher_model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
    criterion1 = nn.KLDivLoss()
    criterion2 = nn.CrossEntropyLoss()
    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    alpha = 0.5
    data_loader = build_detection_train_loader(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for epoch in range(200):
            for data, iteration in zip(data_loader, range(start_iter, max_iter)):
                storage.iter = iteration
                loss_dict = model(data)
                losses = sum(loss_dict.values())
            #train_loader = build_detection_train_loader(cfg)
                for i, data in enumerate(data_loader):
                    with torch.no_grad():
                        print(data)
                        data = json.load()
                        inputs = data[0]
                        output = model(inputs)
                        loss1 = criterion2(output, inputs)
                        teacher_outputs = teacher_model(inputs)
                        T=10
                        output_s = F.log_softmax(output/T, dim=1)
                        output_t = F.softmax(teacher_outputs/T, dim=1)
                        loss2 = criterion1(output_s, output_t)*T*T
                        loss= loss1*0.9+loss2*0.1
                assert torch.isfinite(losses).all(), loss_dict
                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
                optimizer.zero_grad()
                losses.backward()
                loss.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()
                if (
                        cfg.TEST.EVAL_PERIOD > 0
                        and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                        and iteration != max_iter - 1
                ):
                    do_test(cfg, model)
            # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

                if iteration - start_iter > 5 and (
                    (iteration + 1) % 20 == 0 or iteration == max_iter - 1
                 ):
                    for writer in writers:
                        writer.write()
                periodic_checkpointer.step(iteration)


def setup(args):
    """#git://github.com/daixinghome/Distill_GID_detectron2
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file("/mnt/GZY/GZY1/detectron2-main/configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml")
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TRAIN = ("coco_train",)
    cfg.DATASETS.TEST = ("coco_val",)
    cfg.DATALOADER.NUM_WORKERS = 1

    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.MAX_SIZE_TRAIN = 640  # train images
    cfg.INPUT.MAX_SIZE_TEST = 640  # test images
    cfg.INPUT.MIN_SIZE_TRAIN = (512, 768)
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range'

    cfg.MODEL.RETINANET.NUM_CLASSES = 8  # categories
    cfg.MODEL.WEIGHTS = "/mnt/GZY/GZY1/detectron2-main/tools/model_final_bfca0b.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 4
    batch_size = 2
    ITERS_IN_ONE_EPOCH = int(3000 / cfg.SOLVER.IMS_PER_BATCH)

    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 12) - 1  # 12EPOCHS
    cfg.SOLVER.BASE_LR = 0.002  # initial learning rate
    cfg.SOLVER.MOMENTUM = 0.9  # optimizer
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0  # weight
    cfg.SOLVER.GAMMA = 0.1  # learning rate decay rates
    cfg.SOLVER.STEPS = (7000,)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1
    cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def build_teacher_model(args):
    """
    Create configs and perform basic setups.
    """
    teacher_model = get_cfg()
    args.config_file = "/mnt/GZY/GZY1/detectron2-main/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml"
    teacher_model.merge_from_file(args.config_file)
    teacher_model .merge_from_list(args.opts)
    teacher_model.freeze()
    default_setup(
        teacher_model, args
    )  # if you don't
    return teacher_model

def main(args):
    cfg = setup(args)
    teacher_model = build_model(build_teacher_model(args))
    DetectionCheckpointer(teacher_model).load("/mnt/GZY/GZY1/detectron2/tools/output/model_final.pth")
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.TEACHER_MODEL.WEIGHTS, resume=args.resume)
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1



    do_train(cfg,model,teacher_model, resume=args.resume)
    return do_test(cfg, model)


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
