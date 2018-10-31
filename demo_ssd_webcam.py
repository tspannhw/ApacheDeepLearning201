"""SSD Demo script."""
import os
import argparse
import mxnet as mx
import gluoncv as gcv
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
import math
import random, string
import time
import numpy
import random, string
import time
import psutil
import scipy.misc
from time import gmtime, strftime
from gluoncv.data.transforms import presets
from matplotlib import pyplot as plt

start = time.time()

def parse_args():
    parser = argparse.ArgumentParser(description='Test with SSD networks.')
    parser.add_argument('--network', type=str, default='ssd_300_vgg16_atrous_voc',
                        help="Base network name")
    parser.add_argument('--images', type=str, default='',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.cpu()] 

    # grab some image if not specified
    cap = cv2.VideoCapture(0)
    time.sleep(1)  ### letting the camera autofocus
    ret, frame = cap.read()
    uuid = '{0}_{1}'.format(strftime("%Y%m%d%H%M%S",gmtime()),uuid.uuid4())
    filename = 'images/ssd_image_{0}.jpg'.format(uuid)
    filename2 = 'images/ssd_p_image_{0}.jpg'.format(uuid)
    cv2.imwrite(filename, frame)

    image = filename

    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gcv.model_zoo.get_model(args.network, pretrained=True)
    else:
        net = gcv.model_zoo.get_model(args.network, pretrained=False, pretrained_base=False)
        net.load_parameters(args.pretrained)
    net.set_nms(0.45, 200)
    net.collect_params().reset_ctx(ctx = ctx)

    ax = None
    x, img = presets.ssd.load_test(image, short=512)
    x = x.as_in_context(ctx[0])
    ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]
    ax = gcv.utils.viz.plot_bbox(img, bboxes, scores, ids,
                                    class_names=net.classes, ax=ax)
    plt.savefig(filename2)


