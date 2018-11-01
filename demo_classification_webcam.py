import mxnet as mx
import gluoncv
import time
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

start = time.time()
cap = cv2.VideoCapture(0)   # 0 - laptop   #1 - monitor  #2 external cam
time.sleep(3)
ret, frame = cap.read()
uuid = '{0}_{1}'.format(strftime("%Y%m%d%H%M%S",gmtime()),uuid.uuid4())
filename = 'images/classify_image_{0}.jpg'.format(uuid)
filename2 = 'images/classify_image_p_{0}.jpg'.format(uuid)
cv2.imwrite(filename, frame)

# you may modify it to switch to another model. The name is case-insensitive
model_name = 'ResNet50_v1d'
#model_name = 'CIFAR_ResNeXt29_16x64d'
# download and load the pre-trained model
net = gluoncv.model_zoo.get_model(model_name, pretrained=True)
# load image
img = mx.image.imread(filename)
# apply default data preprocessing
transformed_img = gluoncv.data.transforms.presets.imagenet.transform_eval(img)
# run forward pass to obtain the predicted score for each class
pred = net(transformed_img)
# map predicted values to probability by softmax
prob = mx.nd.softmax(pred)[0].asnumpy()
# find the 5 class indices with the highest score
ind = mx.nd.topk(pred, k=5)[0].astype('int').asnumpy().tolist()
# print the class name and predicted probability
print('The input picture is classified to be')
for i in range(5):
    print('- [%s], with probability %.3f.'%(net.classes[ind[i]], prob[ind[i]]))
