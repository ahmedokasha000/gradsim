#!/usr/bin/env python
# coding: utf-8

# In[1]:


import helper_functions as hf
import cv2
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
import numpy as np
import os 


# In[2]:
current_dir=os.getcwd().split('/')
pb_fname="/home/"+current_dir[2]+"/catkin_ws/src/gradsim/src/bump_detect_Camera/ssd_inception_v2_coco_trt.pb"

# Model = 'DL_Models/MobNv1_Grayscale/trtOptimized/'
# pb_fname = Model + 'ssd_inception_v2_coco_trt.pb'


# In[4]:


trt_graph = hf.get_frozen_graph(pb_fname)

input_names = ['image_tensor']

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')

tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')
tf_input.shape.as_list()


# In[7]:


cap = cv2.VideoCapture("/home/grad20/Desktop/VID_20200807_190001.mp4")
width, height, fps = 2560, 720, 25   #cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

while True :

    retval, image = cap.read()
    left_right_image = np.split(image, 2, axis=1)
    image = left_right_image[1]
    
    # Detect Here...
    scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections],
                                                         feed_dict={tf_input: image[None, ...]})
    boxes = boxes[0] # index by 0 to remove batch dimension
    scores = scores[0]
    classes = classes[0]
    num_detections = int(num_detections[0])

    # GPIO Signal........................
    if scores[0] > 0.3:
        print('Bump :(')
#         hf.visualize_detection(image, num_detections, classes, boxes, scores)
    else:
        print('No Bump :)')
    #....................................
    
    cv2.imshow("Detection", image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

