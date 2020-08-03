#!/usr/bin/env python
import numpy as np
from bump_detection import BumpDetection
import witmotion_imu as im
import serial
import time
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Quaternion




serial_port = serial.Serial(
    port="/dev/rfcomm1",
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)

time.sleep(1)
class ImuToROS():
    def __init__(self):
        self.msg_seq=0
        self.imuPub = rospy.Publisher('/imu_data', Imu, queue_size=1)

    def imu_to_topic(self,imu_read):
        #print(imu_read)
        imu_data = Imu()
        
        acc,ang_v,eular=imu_read
        print(eular)
        eular[0] =eular[0]*((np.pi)/180.0)
        eular[1] = eular[1]*((np.pi)/180.0)
        eular[2] = eular[2]*((np.pi)/180.0)
        
        imu_data.linear_acceleration.x=acc[0]*9.8
        imu_data.linear_acceleration.y=acc[1]*9.8
        imu_data.linear_acceleration.z=acc[2]*9.8
        imu_data.angular_velocity.x=ang_v[0]*(np.pi/180.0)
        imu_data.angular_velocity.y=ang_v[1]*(np.pi/180.0)
        imu_data.angular_velocity.z=ang_v[2]*(np.pi/180.0)
        q=quaternion_from_euler(eular[0], eular[1], eular[2])
        quatern=Quaternion(q[0],q[1],q[2],q[3])
        imu_data.orientation=quatern
        imu_data.orientation_covariance=[0.,0.,0.,0.,0.,0.,0.,0.,0.]
        imu_data.angular_velocity_covariance=[0.,0.,0.,0.,0.,0.,0.,0.,0.]
        imu_data.linear_acceleration_covariance=[0.,0.,0.,0.,0.,0.,0.,0.,0.]
        timeob = rospy.get_rostime()
        imu_data.header.frame_id="imu_link"
        imu_data.header.stamp=timeob
        imu_data.header.seq=self.msg_seq
        self.imuPub.publish(imu_data)
        self.msg_seq+=1
        

if __name__ == '__main__':
  try:
    rospy.init_node('imu_interface', anonymous=True)
    bumpPub = rospy.Publisher('/bump_detection_imu', String, queue_size=1)
    imu_ros=ImuToROS()
    imu1=im.IMUDriver()
    rate = rospy.Rate(2000)
    #####################   Accident Detection Initialization   #####################  
    OUTPUT_RATE = 100
    Bump_ob = BumpDetection(OUTPUT_RATE,log_data=False)
    Bump_ob.DIFF_THRESHOLD = 1.0  # set the acceleration change threshold (3g set as default)
    Bump_ob.STD_THRESHOLD = 0.6  # set the standard deviation threshold
    BUFFER_SIZE= 50 # should contain 0.5s measurements, 50 measurements in case of 100Hz output rate
    ####################################################################################
    last_seq = 0
    buffer=[]
    while not rospy.is_shutdown():
        if serial_port.inWaiting() > 0:
            data = ord(serial_port.read())
            imu1.decode_format(data)
        if imu1.frame_seq > last_seq:
            row=(imu1.acceleration,imu1.angular_velocity,imu1.eular_angles)
            imu_ros.imu_to_topic(row)
            buffer.append([imu1.acceleration[0],imu1.acceleration[1],imu1.acceleration[2]])
            #csv_writer.writerow(info)
            last_seq += 1
        if len(buffer) >= BUFFER_SIZE:
            arr = np.array(buffer)
            #####################   Feed Algorithm with the new buffer array ############
            result = Bump_ob.process_buffer(arr)  
            if result["bool"]:
                bumpPub.publish("1")
            else:
                bumpPub.publish("0")
            #print(result)
            del buffer[:]
       

    rate.sleep()
  except rospy.ROSInterruptException:
    pass


