#!/usr/bin/env python3

import numpy as np
import rospy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String
import csv
import os
import math

GPS_coord = [None,None]
MAX_DIST_NOTIFY=50
latest_nearest_bump=MAX_DIST_NOTIFY

def read_database(file_dir):
    database_data = []
    with open(file_dir, 'r') as db:
        csv_reader = csv.reader(db, delimiter=',')
        for row in csv_reader:
            #print(row)
            row[0] = float(row[0])
            row[1] = float(row[1])
            database_data.append(row)
    db.close()
    return database_data

def send_notfication():
    pass
def check_near_bumps(current_coor, database_data,max_notify=50):

    # 0 -> latitude
    # 1 -> longitude
    R = 6373.0
    i = 0
    global latest_nearest_bump
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
        #print('distance' + str(i) + ' = ' + str(meter))
        if(meter <= max_notify):
            near_bumps_distance.append(meter)
    if near_bumps_distance:
        nearest_bump=np.min(near_bumps_distance)
        if(nearest_bump-latest_nearest_bump)<0:
            duration = 0.2  # seconds
            freq = 4500  # Hz
            os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
            message="Bump within "+str(nearest_bump)+"m distance"
            bumpNotficationPub.publish(message)
            print('Send Notification')
        #bumpNotficationPub.publish("hey")
        latest_nearest_bump=nearest_bump
    else:
        latest_nearest_bump=max_notify


def get_gps_coordinates(msg):
    GPS_coord[0] = msg.latitude
    GPS_coord[1] = msg.longitude

if __name__ == '__main__':
    try:
        current_dir = os.getcwd().split('/')
        database_dir = "/home/" + current_dir[2] + "/catkin_ws/src/gradsim/src/detected_bumps_coordinates.csv"
        rospy.init_node('bump_notfications', anonymous=True)
        rate = rospy.Rate(3)
        GPS_filteredSub = rospy.Subscriber("/gps_data", NavSatFix, get_gps_coordinates)
        bumpNotficationPub = rospy.Publisher("/driver_notification", String, queue_size=2)
        
        while not rospy.is_shutdown():
            try:
                database_d = read_database(database_dir)
            except IOError:
                print("waiting for database to be created")
                database_d=[]
            
            if GPS_coord[0]:
                check_near_bumps(GPS_coord, database_d,MAX_DIST_NOTIFY)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass