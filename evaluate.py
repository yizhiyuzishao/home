import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import pylab
import json
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("/mnt/GZY/GZY1/detectron2/data/datasets/coco/annontations/test2022.json", "--gt",type=str,help="Assign the groud true path.",default=None)
    parser.add_argument("/mnt/GZY/GZY1/detectron2/tools/output/metrics.json", "--dt", type=str, help="Assign the detection result path.", default=None)
    args = parser.parse_args()
    cocoGt = COCO(args.gt)
    cocoDt = cocoGt.loadRes(args.dt)
    cocoEval = COCOeval(cocoGt,cocoDt,"keypoints")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
