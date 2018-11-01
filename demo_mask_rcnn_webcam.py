
import os
import argparse
import mxnet as mx
import cv2
import time
import gluoncv as gcv
import sys
import datetime
import subprocess
import os
import numpy
import base64
import uuid
import datetime
import traceback
import math
import random, string
import base64
import json
from time import gmtime, strftime
import numpy as np
import cv2
import math
import random, string
import time
import numpy
import random, string
import time
import psutil
import scipy.misc
from time import gmtime, strftime

from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils

start = time.time()

net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)

# --- make a time library for this 
cap = cv2.VideoCapture(0)
time.sleep(1)  ### letting the camera autofocus
ret, frame = cap.read()
uuid = '{0}_{1}'.format(strftime("%Y%m%d%H%M%S",gmtime()),uuid.uuid4())
filename = 'images/maskrcnn_image_{0}.png'.format(uuid)
filename2 = 'images/maskrcnn_p_image_{0}.png'.format(uuid)
cv2.imwrite(filename, frame)
# ----

x, orig_img = data.transforms.presets.rcnn.load_test(filename)

# overlay segmentation masks.

ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]

# paint segmentation mask on images directly
width, height = orig_img.shape[1], orig_img.shape[0]
masks = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
orig_img = utils.viz.plot_mask(orig_img, masks)

# identical to Faster RCNN object detection
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids,
                         class_names=net.classes, ax=ax)
#plt.show()
plt.savefig(filename2)
