#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. Imports

import cv2
from PIL import Image
import sys
import os
import urllib
# import tensorflow.contrib.tensorrt as trt
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np
import time
from tf_trt_models.detection import download_detection_model, build_detection_graph

get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


MODEL = 'MobNv1_Grayscale/'
CONFIG_FILE = MODEL + 'Frozen/pipeline.config'
CHECKPOINT_FILE = MODEL + 'Frozen/model.ckpt'    


# In[12]:


# Generating The Frozen_Inference_Graph from the trained model (checkpoints)

frozen_graph, input_names, output_names = build_detection_graph(
    config = '/home/mohamed/Dev/Gradp/DL_Models/MobNv1_Grayscale/Frozen/pipeline.config',
    checkpoint = CHECKPOINT_FILE,
    score_threshold = 0.3,
    batch_size = 1
)


# In[13]:


# Generating The Optimized TensorRT Model from the Frozen model

trt_graph = trt.create_inference_graph(
    input_graph_def = frozen_graph,
    outputs = output_names,
    max_batch_size = 1,
    max_workspace_size_bytes = 1 << 25,
    precision_mode = 'FP16',
    minimum_segment_size = 50
)


# In[14]:


# Serializing and Writing the pb model to disk

with open( MODEL + '/ssd_inception_v2_coco_trt.pb', 'wb') as f:
    f.write(trt_graph.SerializeToString())


# In[16]:


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

tf_sess = tf.Session(config=tf_config)

tf.import_graph_def(trt_graph, name='')

tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')


# In[17]:


# Image Test.....................................
IMAGE_PATH = 'bumpy.jpg' 

image = Image.open(IMAGE_PATH)
plt.imshow(image)
image_resized = np.array(image.resize((300, 300)))
image = np.array(image)


# In[ ]:


# Testing the model on an sample image

import time
start_time = time.time()

scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], 
                                                      feed_dict={tf_input: image_resized[None, ...]})

boxes = boxes[0] # index by 0 to remove batch dimension
scores = scores[0]
classes = classes[0]
num_detections = num_detections[0]

# # GPIO Signal........................
# if scores[0] > 0.5:
#     GPIO.output(18, GPIO.HIGH) 
#     print('Bump :(')
# else:
#     print('No Bump :)')
#     GPIO.output(18, GPIO.LOW) 
# #....................................

print("--- %s seconds ---" % (time.time() - start_time))


# In[19]:


from matplotlib import pyplot as plt

fig = plt.figure()   # name of the figure ( as a whole)
ax = fig.add_subplot(1, 1, 1)

ax.imshow(image)

plt.axis('off')

# plot boxes exceeding score threshold
for i in range(int(num_detections)):  #####
    # scale box to image coordinates
    box = boxes[i] * np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])

    # display rectangle
    patch = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], color='g', alpha=0.3)
    ax.add_patch(patch)

    # display class index and score
    plt.text(x=box[1] + 10, y=box[2] - 10, s='%d (%0.2f) ' % (classes[i], scores[i]), color='w')

plt.ioff()
plt.show()
#plt.savefig("img9.png", bbox_inches='tight', pad_inches = 0)

plt.close(fig)

