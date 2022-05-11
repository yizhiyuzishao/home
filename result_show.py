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
import time
img_file = '/home/ps/DiskA/project/GZY1/Distill_GID_detectron2/datasets/coco/images/liangyou_ng_00001_0_0.jpg'
#im_folder= '/home/ps/DiskA/project/GZY1/Distill_GID_detectron2/datasets/PCBdataset/images'
save_folder = '/home/ps/DiskA/project/GZY1'

#for im_file in os.listdir(im_folder):
  #im = cv2.imread(os.path.join(im_folder,im_file))
im = cv2.imread(img_file)
img_file = os.path.basename(img_file)
save_result_path = os.path.join(save_folder, img_file)
 # save_result_path = os.path.join(save_folder, im_file)
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
cfg.DISTILL.TEACHER_CFG.MODEL.RETINANET.NUM_CLASSES = 9
cfg.DISTILL.STUDENT_CFG.MODEL.RETINANET.NUM_CLASSES = 9
cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/home/ps/DiskA/project/GZY1/result/101-18/small/model_final.pth") 
model = build_model(cfg)
print(model)
time_start = time.time()  # 记录开始时间
predictor = DefaultPredictor(cfg)
time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum) 
outsputs = predictor(im)
pred_classes = outsputs["instances"].pred_classes
pred_boxes = outsputs["instances"].pred_boxes
	#在原图上画出检测结果
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
v = v.draw_instance_predictions(outsputs["instances"].to("cpu"),0.5)

plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
plt.axis('off')
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.imshow(v.get_image())
plt.savefig(save_result_path) #保存结果
