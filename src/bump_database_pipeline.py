#!/usr/bin/env python3

import numpy as np
import rospy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String
import csv
import os
import math


latest_ML_result=None
latest_DL_result=None
GPS_coord=[None,None]
def calc_dist_two_coor(current_coor,location):
    # 0 -> latitude
    # 1 -> longitude
    R = 6373.0
    i = 0
    current_coor = {'lon': math.radians(current_coor[1]),
                    'lat': math.radians(current_coor[0])}
    location = {'lon': math.radians(location[1]),
                'lat': math.radians(location[0])}
    dlon = location['lon'] - current_coor['lon']
    dlat = location['lat'] - current_coor['lat']
    a = math.sin(dlat / 2)**2 + math.cos(current_coor['lat']) * math.cos(
        location['lat']) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    meter = (int)(distance * 1000.0)
    return meter
def read_database(file_dir):
    database_data = []
    with open(file_dir, 'r') as db:
        csv_reader = csv.reader(db, delimiter=',')
        for row in csv_reader:
            #print(row)
            row[0] = float(row[0])
            row[1] = float(row[1])
            database_data.append(row)
    return database_data


def get_gps_coordinates(msg):
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
def overwrite_db(database_l,dirr):
    current_dir=os.getcwd().split('/')
    database_dir=dirr
    with open(database_dir, 'w') as f1:
        writer = csv.writer(f1)
        for loc in database_l: 
            writer.writerow(loc)
    print("database is overwritten")

def check_bump_exist(database_l,current_coor,min_dist=6):
    for ind in  range(len(database_l)):
        location=database_l[ind]
        dist=calc_dist_two_coor(current_coor,location)
        if (dist<min_dist):
            location[0]=(location[0]+current_coor[0])/2.
            location[1]=(location[1]+current_coor[1])/2.
            database_l[ind]=location
            print("bump already exist, average is taken")
            return(1,database_l)
    return(0,database_l)
def clean_db(database_l,min_dist=6):
    new_db=[]
    new_db=database_l
    for first_loc_ind in range(len(database_l)-2):
        for second_loc_ind in range(first_loc_ind+1,len(database_l)-1):
            first_loc=database_l[first_loc_ind]
            second_loc=database_l[second_loc_ind]
            dist=calc_dist_two_coor(second_loc,first_loc)
            #print(dist)
            if (dist<min_dist):
                first_loc[0]=(first_loc[0]+second_loc[0])/2.
                first_loc[1]=(first_loc[1]+second_loc[1])/2.
                database_l[first_loc_ind]=first_loc
                del new_db[second_loc_ind]
                print("bump already exist index %d, average is taken"%second_loc_ind)
    return new_db



if __name__ == '__main__':
  try:
    rospy.init_node('bump_database_pipeline', anonymous=True)
    rate = rospy.Rate(20)
    GPS_filteredSub = rospy.Subscriber("/gps/filtered", NavSatFix, get_gps_coordinates)
    DL_resultSub = rospy.Subscriber("/bump_detected_DL", String, get_DL_results)
    ML_resultSub= rospy.Subscriber("/bump_detected_ML",String,get_ML_results)
    duration = 0.4  # seconds
    freq = 1500  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    current_dir=os.getcwd().split('/')
    database_dir="/home/"+current_dir[2]+"/catkin_ws/src/gradsim/src/detected_bumps_coordinates.csv"
    database= read_database(database_dir)
    database=clean_db(database,min_dist=8)
    while not rospy.is_shutdown():
        if ((latest_ML_result=="Bump Detected") or (latest_DL_result=="Bump Detected")) :
            latest_ML_result=None
            latest_DL_result=None
            os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
            
            check_result,database=check_bump_exist(database,GPS_coord,min_dist=8)
            if (check_result==0):
                print("saving Bump location to database")
                database.append(GPS_coord)
                save_coord_to_database(GPS_coord)
            else:

                overwrite_db(database,database_dir)

        rate.sleep()
  except rospy.ROSInterruptException:
    pass
 