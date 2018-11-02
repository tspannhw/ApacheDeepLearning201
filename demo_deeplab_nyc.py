import time
from matplotlib import pyplot as plt
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
import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
# using cpu
ctx = mx.cpu(0)

uuid = '{0}_{1}'.format(strftime("%Y%m%d%H%M%S",gmtime()),uuid.uuid4())
filename2 = 'images/dl3_img_nyc_{0}.png'.format(uuid)
filename = 'images/dl3_org_nyc_{0}.png'.format(uuid)

# http://207.251.86.238/cctv787.jpg
url = sys.argv[1]
gluoncv.utils.download(url, filename, True)

##############################################################################
# load the image
img = image.imread(filename)

##############################################################################
# normalize the image using dataset mean
transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])
img = transform_fn(img)
img = img.expand_dims(0).as_in_context(ctx)

##############################################################################
# Load the pre-trained model and make prediction
# ----------------------------------------------
#
# get pre-trained model
model = gluoncv.model_zoo.get_model('deeplab_resnet101_ade', pretrained=True)

##############################################################################
# make prediction using single scale
output = model.demo(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

##############################################################################
# Add color pallete for visualization
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
mask = get_color_pallete(predict, 'ade20k')
#mask.save('output.png')
mask.save(filename2)
# show the predicted mask
#mmask = mpimg.imread('output.png')
#plt.imshow(mmask)
#plt.show()
#plt.savefig(filename2)
