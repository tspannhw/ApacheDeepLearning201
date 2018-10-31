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
from gluoncv.data.transforms import presets
from matplotlib import pyplot as plt

start = time.time()
import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
# using cpu
ctx = mx.cpu(0)

cap = cv2.VideoCapture(0)
time.sleep(1)  ### letting the camera autofocus
ret, frame = cap.read()
uuid = '{0}_{1}'.format(strftime("%Y%m%d%H%M%S",gmtime()),uuid.uuid4())
filename = 'images/fcn_image_{0}.png'.format(uuid)
filename2 = 'images/fcn_p_image_{0}.png'.format(uuid)
cv2.imwrite(filename, frame)


##############################################################################

#from matplotlib import pyplot as plt
#plt.imshow(img.asnumpy())
#plt.show()

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
model = gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True)

##############################################################################
# make prediction using single scale
output = model.demo(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

##############################################################################
# Add color pallete for visualization
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
mask = get_color_pallete(predict, 'pascal_voc')
mask.save(filename2)

##############################################################################
# show the predicted mask
mmask = mpimg.imread(filename2)
plt.imshow(mmask)
plt.show()

