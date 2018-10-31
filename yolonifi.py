# See https://gluon-cv.mxnet.io/build/examples_detection/demo_yolo.html#sphx-glr-build-examples-detection-demo-yolo-py
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
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
filename = 'images/gluoncv_image_{0}.jpg'.format(uuid)
filename2 = 'images/gluoncv_image_p_{0}.jpg'.format(uuid)
cv2.imwrite(filename, frame)

# requires gluoncv 0.3 which is in beta when I did this
# 24-sept-2018

net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

x, img = data.transforms.presets.yolo.load_test(filename, short=512)

class_IDs, scores, bounding_boxs = net(x)

ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0], class_IDs[0], class_names=net.classes)

plt.savefig(filename2)

classname = str(class_IDs[0,0,0][0]).strip('<NDArray 1 @cpu(0)>') 
classname = classname.strip('\n[')
classname = classname.strip('.]\n')
classname = int(classname)

matchingpct = str(scores[0][0][0]).strip('<NDArray 1 @cpu(0)>') 
matchingpct = matchingpct.strip('\n[')
matchingpct = matchingpct.strip('.]\n')
matchingpct = float(matchingpct)

end = time.time()
row = { }
row['imgname'] = filename
row['imgnamep'] = filename2
row['class1'] = str(net.classes[classname])
row['pct1'] = '{0}'.format( str((matchingpct) * 100))
row['host'] = os.uname()[1]
row['shape'] = str(x.shape)
row['end'] = '{0}'.format( str(end ))
row['te'] = '{0}'.format(str(end-start))
row['battery'] = psutil.sensors_battery()[0]
row['systemtime'] = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
row['cpu'] = psutil.cpu_percent(interval=1)
usage = psutil.disk_usage("/")
row['diskusage'] = "{:.1f} MB".format(float(usage.free) / 1024 / 1024)
row['memory'] = psutil.virtual_memory().percent
row['id'] = str(uuid)
json_string = json.dumps(row)
print(json_string)
