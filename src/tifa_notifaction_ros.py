#!/usr/bin/env python

import numpy as np
import rospy
from std_msgs.msg import String
from TIFA_interface.i2c_script import init_i2c,writeNumber,readNumber



def get_notfication(msg):
    print("notification received")
    #dist =11
    dist=msg.data.split(',')[1]
    
    writeNumber(0x44,bus,address)
    writeNumber(25,bus,address)
    writeNumber(int(dist),bus,address)
    writeNumber(1,bus,address)
bus,address=init_i2c()

if __name__ == '__main__':
    try:
        rospy.init_node('TIFA_communication', anonymous=True)
        DriverSub = rospy.Subscriber("/driver_notification", String, get_notfication)
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            rate.sleep()
    except rospy.ROSInterruptException:
        pass