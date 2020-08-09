#!/usr/bin/env python3

import numpy as np
import rospy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String
import csv
import os
import math

GPS_coord = [30.668732666666664,30.066891833333337]

def read_database(file_dir):
    database_data = []
    with open(file_dir, 'r') as db:
        csv_reader = csv.reader(db, delimiter=',')
        for row in csv_reader:
            print(row)
            row[0] = float(row[0])
            row[1] = float(row[1])
            database_data.append(row)
    return database_data

def send_notfication():
    pass
def check_near_bumps(current_coor, database_data):

    # 0 -> latitude
    # 1 -> longitude
    R = 6373.0
    i = 0
    current_coor = {'lon': math.radians(current_coor[1]),
                    'lat': math.radians(current_coor[0])}
    near_bumps_distance = []
    for location in database_data:
        location = {'lon': math.radians(location[1]),
                    'lat': math.radians(location[0])}

        dlon = location['lon'] - current_coor['lon']
        dlat = location['lat'] - current_coor['lat']
        a = math.sin(dlat / 2)**2 + math.cos(current_coor['lat']) * math.cos(
            location['lat']) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        meter = (int)(distance * 1000.0)
        i = i + 1
        print('distance' + str(i) + ' = ' + str(meter))
        if(meter <= 50):
            near_bumps_distance.append(meter)
    if near_bumps_distance:
        nearest_bump=np.min(near_bumps_distance)

        message="Bump within "+str(nearest_bump)+"m distance"
        bumpNotficationPub.publish(message)
        print('Send Notification')

def get_gps_coordinates(msg):
    print("gps received")
    GPS_coord[0] = msg.latitude
    GPS_coord[1] = msg.longitude

if __name__ == '__main__':
    try:
        current_dir = os.getcwd().split('/')
        database_dir = "/home/" + current_dir[2] + "/catkin_ws/src/gradsim/src/detected_bumps_coordinates.csv"
        rospy.init_node('bump_notfications', anonymous=True)
        rate = rospy.Rate(3)
        GPS_filteredSub = rospy.Subscriber("/gps/filtered", NavSatFix, get_gps_coordinates)
        bumpNotficationPub = rospy.Publisher("/driver_notification", String, queue_size=2)
        while not rospy.is_shutdown():
            database_d = read_database(database_dir)
            check_near_bumps(GPS_coord, database_d)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass