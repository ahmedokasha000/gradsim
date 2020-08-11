#!/usr/bin/env python3

import numpy as np
import rospy
from std_msgs.msg import String
from TIFA_interface.i2c_script import init_i2c,writeNumber,readNumber



def get_notfication(msg):
    print("notification received")
    recv_data=msg.data
    writeNumber(0x04,bus,address)

if __name__ == '__main__':
    try:
        rospy.init_node('TIFA_communication', anonymous=True)
        rate = rospy.Rate(30)
        
        GPS_filteredSub = rospy.Subscriber("/driver_notification", String, get_notfication)
        bus,address=init_i2c()
        print(bus)
        while not rospy.is_shutdown():
            rate.sleep()
    except rospy.ROSInterruptException:
        pass