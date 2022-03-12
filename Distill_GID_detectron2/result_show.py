import numpy as np
import cv2
import os 
from PIL import Image
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import random
from detectron2.utils.logger import setup_logger
setup_logger()
 
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
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
save_folder = '/home/ps/DiskA/project/GZY1/output'
# register dataset
def plain_register_dataset():
    # register dataset to dataset catalog,register metadata to metadata catalog and set attribute
    DatasetCatalog.register("coco_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
    MetadataCatalog.get("coco_train").set(json_file=TRAIN_JSON, image_root=TRAIN_PATH,evaluator_type="coco")
    DatasetCatalog.register("coco_val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
    MetadataCatalog.get("coco_val").set(json_file=VAL_JSON, image_root=VAL_PATH,evaluator_type="coco")

plain_register_dataset()
for im_file in os.listdir(im_folder):
	im = cv2.imread(os.path.join(im_folder, im_file))
	save_result_path = os.path.join(save_folder, im_file)
	height = im.shape[0]
	width = im.shape[1]
	dpi = 500
	cfg = get_cfg()
	cfg.merge_from_file('/home/ps/DiskA/project/GZY1/Distill_GID_detectron2/RetinaNet_Res101/config/RetinaNet_1x_smooth_l1.yaml')
	cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  #模型阈值
	cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True
	cfg.MODEL.WEIGHTS = '/home/ps/DiskA/project/GZY1/Distill_GID_detectron2/RetinaNet_Res101/output_1x_smooth_l1/model_final.pth'
	predictor = DefaultPredictor(cfg) 
	outputs = predictor(im)
	pred_classes = outputs["instances"].pred_classes
	pred_boxes = outputs["instances"].pred_boxes
	#在原图上画出检测结果
	v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
	v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
	plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
	plt.axis('off')
	plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
	plt.imshow(v.get_image())
	plt.savefig(save_result_path) #保存结果
