#!/usr/bin/env python3
# coding: utf-8

# In[1]:


from bump_detect_Camera import helper_functions as hf
import cv2
#import tensorflow.contrib.tensorrt as trt
#import tensorflow as tf
from tensorflow import ConfigProto,Session,import_graph_def
import numpy as np
import os 
import rospy
from std_msgs.msg import String

import requests
from time import sleep,time

url= "http://192.168.43.1:8080/shot.jpg"
#url= "http://192.168.1.4:8080/shot.jpg"

def preprocess(image, contrast_factor=0.85, intensity_thresh=200):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)    
    
    if image.std() > 45.0:
        image = np.array(image * contrast_factor, dtype=np.uint8)
        
    _, image = cv2.threshold(image, intensity_thresh, 255, cv2.THRESH_TRUNC)
    
    return image

if __name__ == '__main__':
    try:
        
        rospy.init_node('bump_detect_camera_dl', anonymous=True)
        rate = rospy.Rate(30)
        bumpNotficationPub = rospy.Publisher("/bump_detected_DL", String, queue_size=2)
        current_dir=os.getcwd().split('/')
        pb_fname="/home/"+current_dir[2]+"/catkin_ws/src/gradsim/src/bump_detect_Camera/ssd_inception_v2_coco_trt.pb"

        trt_graph = hf.get_frozen_graph(pb_fname)
        input_names = ['image_tensor']
        tf_config = ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_sess = Session(config=tf_config)
        import_graph_def(trt_graph, name='')
        tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
        tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
        tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
        tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
        tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')
        tf_input.shape.as_list()
        count = 0
        score_threshold = 0.5
        start = time()
        out = cv2.VideoWriter("/home/grad20/Desktop/grad_project"+str(start)+".avi",cv2.VideoWriter_fourcc(*'MJPG'),30,(640,480))
        while not rospy.is_shutdown():
            img_response =requests.get(url)
            img_arr=np.array(bytearray(img_response.content))
            image= cv2.imdecode(img_arr,-1)
            image = preprocess(image)
            # Detect Here...
            scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections],
                                                                feed_dict={tf_input: image[None, ...]})
            boxes = boxes[0] # index by 0 to remove batch dimension
            scores = scores[0]
            classes = classes[0]
            num_detections = int(num_detections[0])

            print('Processing...')
            if scores[0] > score_threshold:
                count += 1
                hf.visualize_detection(image, num_detections, classes, boxes, scores)
                print('Probably a Bump')
                
            if count > 4:
                count = 0
                print('Definetly a Bump')
                bumpNotficationPub.publish("Bump Detected")
                
                """ Produce Sound Here
                os.system('play -nq -t alsa synth {} sine {}'.format(0.1, 440))
                """
            else:
                bumpNotficationPub.publish("No Bump")    
            if time()-start > 1:
                count = 0
                start = time()  
            out.write(image)  
            #....................................
            #cv2.imshow("Detection", image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
                
            rate.sleep()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
