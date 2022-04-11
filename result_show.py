from pyexpat import model
import torch
import numpy as np
import cv2
import os 
from PIL import Image
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import random
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.modeling import build_model
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from Distill_GID_detectron2.GID.gid.config import add_distill_cfg
DATASET_ROOT = '/home/ps/DiskA/project/GZY1/detectron2/data/datasets/coco'
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
im_folder= '/home/ps/DiskA/project/GZY1/detectron2/data/datasets/coco/images'
save_folder = '/home/ps/DiskA/project/GZY1/output_res101-res18'
# register dataset
def plain_register_dataset():
    # register dataset to dataset catalog,register metadata to metadata catalog and set attribute
    DatasetCatalog.register("coco_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
    MetadataCatalog.get("coco_train").set(json_file=TRAIN_JSON, image_root=TRAIN_PATH,evaluator_type="coco")
    DatasetCatalog.register("coco_val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
    MetadataCatalog.get("coco_val").set(json_file=VAL_JSON, image_root=VAL_PATH,evaluator_type="coco")

plain_register_dataset()
'''#Visualizing the Train Dataset
dataset_dicts = DefaultTrainer.build_train_loader("coco_train")
#Randomly choosing 3 images from the Set
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata="coco_train")
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow(vis.get_image()[:, :, ::-1])'''

for im_file in os.listdir(im_folder):
  im = cv2.imread(os.path.join(im_folder,im_file))
  save_result_path = os.path.join(save_folder, im_file)
  height = im.shape[0]
  width = im.shape[1]
  dpi = 500
  cfg = get_cfg()
  cfg = add_distill_cfg(cfg)
  cfg.merge_from_file('/home/ps/DiskA/project/GZY1/Distill_GID_detectron2/GID/config/Distill_retinanet_T_res101_S_res18.yaml')
  cfg.DISTILL.TEACHER_CFG.merge_from_file(cfg.DISTILL.TEACHER_YAML)
  cfg.DISTILL.STUDENT_CFG.merge_from_file(cfg.DISTILL.STUDENT_YAML)
  cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  #模型阈值
  cfg.MODEL.RETINANET.NUM_CLASSES = 8
  cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True
  model = build_model(cfg)
  print(model)
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/home/ps/DiskA/project/GZY1/data/model_0044999.pth") 
  predictor = DefaultPredictor(cfg) 
  outsputs = predictor(im)

  pred_classes = outsputs["instances"].pred_classes
  pred_boxes = outsputs["instances"].pred_boxes
	#在原图上画出检测结果
  v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5,instance_mode=ColorMode.IMAGE_BW)
  v = v.draw_instance_predictions(outsputs["instances"].to("cpu"),0.5)

  plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
  plt.axis('off')
  plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
  plt.imshow(v.get_image())
  plt.savefig(save_result_path) #保存结果
