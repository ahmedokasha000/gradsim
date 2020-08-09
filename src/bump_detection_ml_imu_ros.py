#!/usr/bin/env python3
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import ros_numpy as r_np
from bumb_detect_IMU.bump_detection_ML import load_model,inference_2

def inference_model(msg,model):
    imu_buffer_np=r_np.numpify(msg)
    input_d=[imu_buffer_np[:,i] for i in range(6)]
    #####################   Feed Algorithm with the new buffer array ############
    result=inference_2(input_d,model)
    if(result[0]):
        ML_resultPub.publish("Bump Detected")
    else:
        ML_resultPub.publish("No Bump")
if __name__ == '__main__':
  try:
    model= load_model("/home/ahmed000/catkin_ws/src/gradsim/src/model_1")
    rospy.init_node('bump_detect_ML', anonymous=True)
    imu_buffSub = rospy.Subscriber("/imu_buffer", Image, inference_model,(model))
    ML_resultPub= rospy.Publisher("/bump_detected_ML",String,queue_size=1)
    rospy.spin()
  except rospy.ROSInterruptException:
    pass
 