import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.build import build_detection_test_loader
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
from detectron2.engine.defaults import DefaultPredictor
from detectron2.evaluation.evaluator import inference_on_dataset
from detectron2.modeling import GeneralizedRCNNWithTTA
from Distill_GID_detectron2.GID.gid.config import add_distill_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
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


class DefaultTrainerDistill(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg)

        # checkpointers for Teacher & Student parameters init
        curr_model = self._trainer.model
        if isinstance(self._trainer.model, (DistributedDataParallel, DataParallel)):
            curr_model = curr_model.module
        self.checkpointer_te = DetectionCheckpointer(
            curr_model.teacher,
        )
        self.checkpointer_st = DetectionCheckpointer(
            curr_model.student,
        )

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        if resume:
            self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        else:
            self.checkpointer_te.resume_or_load(self.cfg.DISTILL.TEACHER_CFG.MODEL.WEIGHTS, resume=resume)
            self.checkpointer_st.resume_or_load(self.cfg.DISTILL.STUDENT_CFG.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1


class Trainer(DefaultTrainerDistill):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
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
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
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

    cfg.merge_from_file("/mnt/GZY/GZY1/Distill_GID_detectron2/GID/config/student/RetinaNet_2x_smooth_l1.yaml")

    cfg.merge_from_list(args.opts)

    cfg.DATASETS.TEST = ("coco_val",)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/mnt/GZY/GZY1/Distill_GID_detectron2/GID/output_retina_res101_Res50_2x/model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # cfg.TEST.EVAL_PERIOD = 5000
    # cfg.SOLVER.IMS_PER_BATCH = 4
    # cfg.DATALOADER.NUM_WORKERS = 1
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    predictor = DefaultPredictor(cfg)
    '''evaluator = COCOEvaluator("coco_val",("bbox","segm"),False,output_dir="/mnt/GZY/GZY1/Distill_GID_detectron2/GID/output")
    val_loader = build_detection_test_loader(cfg,"coco_val")
    print(inference_on_dataset(model,val_loader,evaluator))'''

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
