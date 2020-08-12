import time
import helper_functions as hf
import cv2
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
import numpy as np
import os 

""" Insert Image Device Here
cap = cv2.VideoCapture('OnFoot/test5.mp4')
"""

score_threshold = 0.5
def preprocess(image, contrast_factor=0.85, intensity_thresh=200):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)    
    
    if image.std() > 45.0:
        image = np.array(image * contrast_factor, dtype=np.uint8)
        
    _, image = cv2.threshold(image, intensity_thresh, 255, cv2.THRESH_TRUNC)
    
    return image


count = 0
start = time.time()
while True :
    ret, image = cap.read()
    
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

        """ Produce Sound Here
        os.system('play -nq -t alsa synth {} sine {}'.format(0.1, 440))
        """
        
    if time.time()-start > 1:
        count = 0
        start = time.time()
    
    cv2.imshow("Detection", image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# cap.release()
cv2.destroyAllWindows()