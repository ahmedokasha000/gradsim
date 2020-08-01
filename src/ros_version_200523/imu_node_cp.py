#!/usr/bin/env python
import time
import serial
import math as m
import numpy as np
import rospy
from sensor_msgs.msg import Imu
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Quaternion
imu_data = Imu()

rospy.init_node('imu_interface', anonymous=True)
commandPub = rospy.Publisher('/imu_data', Imu, queue_size=1)
timeob = rospy.get_rostime()
serial_port = serial.Serial(
    port="/dev/rfcomm4",
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)
# Wait a second to let the port initialize
time.sleep(1)

try:
    # Send a simple header

    counter = 0
    start_byte = 0
    Temp = 0.0
    acc = np.float_([0.,0.,0.])
    angVel = np.float_([0.,0.,0.])	
    angle = np.float_([0.,0.,0.])
    packet = [0]*11
    temparr = np.int16([0, 0, 0, 0])
    msg_seq=0
    def decodePacket():
        global packet, angle, angVel, acc, Temp, temparr,msg_seq
        temparr[0] = (packet[3] << 8 | packet[2])
        temparr[1] = (packet[5] << 8 | packet[4])
        temparr[2] = (packet[7] << 8 | packet[6])
        temparr[3] = (packet[9] << 8 | packet[8])
        if packet[1] == 0x51:

            acc[0] =temparr[0] / 32768.0 * 16
            acc[1] = temparr[1] / 32768.0 * 16
            acc[2] =temparr[2] / 32768.0 * 16

            Temp = temparr[3] / 340.0 + 36.25
        elif packet[1] == 0x52:
            angVel[0] = temparr[0] / 32768.0 * 2000
            angVel[1] = temparr[1] / 32768.0 * 2000
            angVel[2] = temparr[2] / 32768.0 * 2000
            Temp = temparr[3] / 340.0 + 36.25

        elif packet[1] == 0x53:
            angle[0] = temparr[0] / 32768.0 * (m.pi)
            angle[1] = temparr[1] / 32768.0 * (m.pi)
            angle[2] = temparr[2] / 32768.0 * (m.pi)
            imu_data.linear_acceleration.x=acc[0]*9.8
            imu_data.linear_acceleration.y=acc[1]*9.8
            imu_data.linear_acceleration.z=acc[2]*9.8
            imu_data.angular_velocity.x=angVel[0]*(m.pi/180.0)
            imu_data.angular_velocity.y=angVel[1]*(m.pi/180.0)
            imu_data.angular_velocity.z=angVel[2]*(m.pi/180.0)
            q=quaternion_from_euler(angle[0], angle[1], angle[2])
            quatern=Quaternion(q[0],q[1],q[2],q[3])
            imu_data.orientation=quatern
            imu_data.orientation_covariance=[0.,0.,0.,0.,0.,0.,0.,0.,0.]
            imu_data.angular_velocity_covariance=[0.,0.,0.,0.,0.,0.,0.,0.,0.]
            imu_data.linear_acceleration_covariance=[0.,0.,0.,0.,0.,0.,0.,0.,0.]
            timeob = rospy.get_rostime()
            imu_data.header.frame_id="imu_link"
            imu_data.header.stamp=timeob
            imu_data.header.seq=msg_seq
            commandPub.publish(imu_data)
            msg_seq+=1
            Temp = temparr[3] / 340.0 + 36.25
            #print("a :", acc, "w :", angVel, "angle :", angle, "Temp :", Temp)

    while True:
        
        if serial_port.inWaiting() > 0:
            data = ord(serial_port.read())
            # print(data)
            if (data == 0x55) and (counter == 0):
                #print ("start byte detected")
                start_byte = 1
            if start_byte == 1:
                packet[counter] = data
                counter += 1
                if counter == 11:
                    counter = 0
                    start_byte = 0
                    # print "packet received"
                    # print packet
                    decodePacket()
                    packet = [0]*11
            


except KeyboardInterrupt:
    print("Exiting Program")
finally:
    serial_port.close()
    pass
'''
except Exception as exception_error:
    print("Error occurred. Exiting Program")
    print("Error: " + str(exception_error))
'''

