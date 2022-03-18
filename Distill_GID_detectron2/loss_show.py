import json
import re
from pylab import *
from numpy import append
fig = figure(figsize=(8,6),dpi = 300)
y1 = fig.add_subplot(111)
y1.set_xlabel('Iterations')
y2 = y1.twinx()
y1.set_ylim(0,1,0)
parsed = []

with open('/home/ps/DiskA/project/GZY1/Distill_GID_detectron2/GID/output_retina_res101_Res50_2x/metrics.json') as f:
    try:
        for line in f:
            parsed.append(json.loads(line))
    except:
        print("json format is net correct")
        exit(1)
    _iter = [j['iteration'] for j in parsed]
    _loss_bbox = [j['loss_box_reg'] for j in parsed]
    _loss_cls = [j['loss_cls'] for j in parsed]
    _loss = [j['total_loss'] for j in parsed]
    #try:
     #   _accuracy_cls = [j[''] for j in parsed]
   # except:
     #   _accuracy_cls = None
    _lr = [j['lr']for j in parsed]
    y1.plot(_iter,_loss_bbox,color="green",linewidth=0.3,linestyle="-",label='loss_box_reg')
    y1.plot(_iter,_loss,color="blue",linewidth=0.3,linestyle="-",label='total_loss')
    y1.plot(_iter,  _loss_cls, color="red", linewidth=0.3, linestyle="-", label=' loss_box_cls')
    y2.set_ylim(0,max(_lr)/0.8)
    y2.plot(_iter,_lr,color="purple",linewidth = 1.0,linestyle="-",label='lr')
    y2.set_ylabel('lr')
    y1.legend()
    savefig('loss.png')
    show()