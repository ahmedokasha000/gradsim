#!/usr/bin/env python3

#import Jetson.GPIO as GPIO

#GPIO.cleanup()
#GPIO.setmode(GPIO.BOARD)
#GPIO.setup(18, GPIO.OUT, initial=GPIO.LOW)  #sets deafult o/p value
#print('Done importing GPIO')
#####################
# 1. Imports
import time
import cv2
from PIL import Image
import sys
import os
import urllib
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np
import time
from tf_trt_models.detection import download_detection_model, build_detection_graph
from IPython.display import Image as DisplayImage
import rospy
from std_msgs.msg import String

rospy.init_node('deep_learning', anonymous=True)
resultPub = rospy.Publisher('/bump_detected_DL', String, queue_size=1)
detection_result=String()
print('Done importing')
#######################
#%cd /home/grad20/.virtualenvs/dg1.15/a_detectionTensorRT

MODEL = '/home/ahmed000/Desktop/Frozen_MobNv1GrSc_FT'

CONFIG_FILE = MODEL + '/pipeline.config'
CHECKPOINT_FILE = MODEL + '/model.ckpt'  

########################
frozen_graph, input_names, output_names = build_detection_graph(
    config = CONFIG_FILE,
    checkpoint = CHECKPOINT_FILE,
    score_threshold = 0.3,
    batch_size = 1
)

trt_graph = trt.create_inference_graph(
    input_graph_def = frozen_graph,
    outputs = output_names,
    max_batch_size = 1,
    max_workspace_size_bytes = 1 << 25,
    precision_mode = 'FP16',
    minimum_segment_size = 50
)

with open( MODEL + '/ssd_inception_v2_coco_trt.pb', 'wb') as f:
    f.write(trt_graph.SerializeToString())

print('Done imported frozed') 
########################
# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')
print('Done creat') 
########################
tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')
print('Done score') 
##########################
tf_input.shape.as_list()
print('Done input') 
#########################
#image = cv2.imread(IMAGE_PATH)
#image = cv2.resize(image, (300, 300))
#print('Done resize') 
########################
#scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, #tf_num_detections], feed_dict={
#    tf_input: image[None, ...]
#})
#boxes = boxes[0]  # index by 0 to remove batch dimension
#scores = scores[0]
#classes = classes[0]
#num_detections = int(num_detections[0])
#print('Done imported frozed') 
########################
def save_image(data, fname="/home/grad20/.virtualenvs/dg1.15/a_detectionTensorRT/_images/bump2.jpg", swap_channel=True):
    if swap_channel:
        data = data[..., ::-1]
    cv2.imwrite(fname, data)
print('Done save--------------------------------------------------------')
########################
def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.5, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]),
                  (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale,
                (255, 255, 255), thickness)
print('Done draw--------------------------------------------------------')
#########################
def non_max_suppression(boxes, probs=None, nms_threshold=0.3):
    """Non-max suppression

    Arguments:
        boxes {np.array} -- a Numpy list of boxes, each one are [x1, y1, x2, y2]
    Keyword arguments
        probs {np.array} -- Probabilities associated with each box. (default: {None})
        nms_threshold {float} -- Overlapping threshold 0~1. (default: {0.3})

    Returns:
        list -- A list of selected box indexes.
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > nms_threshold)[0])))
    # return only the bounding boxes indexes
    return pick
print('Done non_max_suppression--------------------------------------------------------')
#######################
print('afify')
print('afify1')
# Open the ZED camera
cap = cv2.VideoCapture(0)
#if cap.isOpened() == 0:
#    exit(-1)
print('afify2')
FILE_OUTPUT = 'output.avi'
width = 2560 #cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
# Get current height of frame
height = 720 #cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
fps = 25 #cap.get(cv2.CAP_PROP_FPS)
print('afify3')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
print('afify4')
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
print('afify5')
#out = cv2.VideoWriter(FILE_OUTPUT,fourcc, fps, (int(width/2),int(height)))# video
i =0
k = 0
print('hi')
while True :
    #print('bye')
    retval, image = cap.read()
    print('hi')
    #left_right_image = np.split(frame, 2, axis=1)
    #image = left_right_image[1]
    
    # Detect Here...
    scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
       tf_input: image[None, ...]
       })

    boxes = boxes[0] # index by 0 to remove batch dimension
    scores = scores[0]
    classes = classes[0]
    num_detections = int(num_detections[0])

    # GPIO Signal........................
    if scores[0] > 0.3:
        #GPIO.output(18, GPIO.LOW) 

        print('Bump :(')
        detection_result.data="Bump Detected"
        resultPub.publish(detection_result)
    else:
        #print('No Bump :)')
        detection_result.data="No Bump"
        resultPub.publish(detection_result)
        #GPIO.output(18, GPIO.HIGH) 
    #....................................
    
    
   
                
    cv2.imshow("Detection", image)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

#out.release()
cap.release()
cv2.destroyAllWindows()




