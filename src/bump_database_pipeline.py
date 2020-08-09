#!/usr/bin/env python3

import numpy as np
import rospy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String
import csv
import os


latest_ML_result=None
latest_DL_result=None
GPS_coord=[None,None]
def get_gps_coordinates(msg):
    print("gps received")
    GPS_coord[0]=msg.latitude
    GPS_coord[1]=msg.longitude
def get_DL_results(msg):

    print("DL result received")
    global latest_DL_result
    latest_DL_result=msg.data

def get_ML_results(msg):
    global latest_ML_result
    latest_ML_result=msg.data

def save_coord_to_database(gps_coordinates):
    current_dir=os.getcwd().split('/')
    database_dir="/home/"+current_dir[2]+"/catkin_ws/src/gradsim/src/detected_bumps_coordinates.csv"
    with open(database_dir, 'a') as f1:
        writer = csv.writer(f1)
        writer.writerow(gps_coordinates)
        pass
    
if __name__ == '__main__':
  try:
    rospy.init_node('bump_database_pipeline', anonymous=True)
    rate = rospy.Rate(50)
    GPS_filteredSub = rospy.Subscriber("/gps/filtered", NavSatFix, get_gps_coordinates)
    DL_resultSub = rospy.Subscriber("/bump_detected_DL", String, get_DL_results)
    ML_resultSub= rospy.Subscriber("/bump_detected_ML",String,get_ML_results)
    duration = 0.4  # seconds
    freq = 1500  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    while not rospy.is_shutdown():
        if ((latest_ML_result=="Bump Detected") or (latest_DL_result=="Bump Detected")) :
            latest_ML_result=None
            latest_DL_result=None
            os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
            print("saving Bump location to database")
            save_coord_to_database(GPS_coord)
        rate.sleep()
  except rospy.ROSInterruptException:
    pass
 