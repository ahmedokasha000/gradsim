#!/usr/bin/env python3

import numpy as np
import rospy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String
import csv
import os
import math

latest_ML_result=None
latest_DL_result=0
GPS_coord=[None,None]

def calc_dist_two_coor(current_coor,location):
    # 0 -> latitude
    # 1 -> longitude
    R = 6373.0
    current_coor = {'lon': math.radians(current_coor[1]),
                    'lat': math.radians(current_coor[0])}
    location = {'lon': math.radians(location[1]),
                'lat': math.radians(location[0])}
    dlon = location['lon'] - current_coor['lon']
    dlat = location['lat'] - current_coor['lat']
    a = math.sin(dlat / 2)**2 + math.cos(current_coor['lat']) * math.cos(location['lat']) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    meter =(distance * 1000.0)
    #print(meter)
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
        db.close()
    return database_data


def get_gps_coordinates(msg):
    GPS_coord[0]=msg.latitude
    GPS_coord[1]=msg.longitude
def get_DL_results(msg):

    print("DL result received")
    global latest_DL_result
    if (msg.data=="Bump Detected"):
        latest_DL_result+=1

def get_ML_results(msg):
    global latest_ML_result
    latest_ML_result=msg.data

def save_coord_to_database(gps_coordinates):
    current_dir=os.getcwd().split('/')
    database_dir="/home/"+current_dir[2]+"/catkin_ws/src/gradsim/src/detected_bumps_coordinates.csv"
    print("got here")
    with open(database_dir, 'a') as f1:
        writer = csv.writer(f1)
        writer.writerow(gps_coordinates)
        f1.close()
def overwrite_db(database_l,dirr):
    database_dir=dirr
    with open(database_dir, 'w') as f1:
        writer = csv.writer(f1)
        for loc in database_l: 
            writer.writerow(loc)
        f1.close()
    print("database is overwritten")

def check_bump_exist(database_l,current_coor,min_dist=6):
    #print(database_l)
    for ind in  range(len(database_l)):
        location=database_l[ind]
        dist=calc_dist_two_coor(location,current_coor)
        #print("distance equal %0.2f"%dist)
        if (dist<min_dist):
            location[0]=(location[0]+current_coor[0])/2.
            location[1]=(location[1]+current_coor[1])/2.
            database_l[ind]=location
            print("bump already exist, average is taken")
            
            return(1,database_l)
    return(0,database_l)
def clean_db(database_l,min_dist=6):
    ind_to_del=[]
    for first_loc_ind in range(len(database_l)-1):
        first_loc=database_l[first_loc_ind]
        for second_loc_ind in range(first_loc_ind+1,len(database_l)):
            second_loc=database_l[second_loc_ind]
            dist=calc_dist_two_coor(second_loc,first_loc)
            if (dist<min_dist):
                first_loc[0]=(first_loc[0]+second_loc[0])/2.
                first_loc[1]=(first_loc[1]+second_loc[1])/2.
                database_l[first_loc_ind]=first_loc
                ind_to_del.append(second_loc_ind)
                print("bump already exist index %d, average is taken"%second_loc_ind)
    ind_to_del=np.sort(ind_to_del)
    sub=0
    new_db=database_l.copy()
    for element_ind in ind_to_del :
        del new_db[element_ind-sub]
        sub+=1
    return new_db


if __name__ == '__main__':
  try:
    rospy.init_node('bump_database_pipeline', anonymous=True)
    rate = rospy.Rate(6)
    GPS_filteredSub = rospy.Subscriber("/gps_data", NavSatFix, get_gps_coordinates)
    DL_resultSub = rospy.Subscriber("/bump_detected_DL", String, get_DL_results)
    ML_resultSub= rospy.Subscriber("/bump_detected_ML",String,get_ML_results)
    duration = 0.1  # seconds
    freq = 1500  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    current_dir=os.getcwd().split('/')
    database_dir="/home/"+current_dir[2]+"/catkin_ws/src/gradsim/src/detected_bumps_coordinates.csv"
    try:
        database_r=read_database(database_dir)
        database=clean_db(database_r,min_dist=15)
        if len(database_r)>len(database):
            print ("repeated lines are detected")
            overwrite_db(database,database_dir)

    except IOError:
        print("File not accessible")
        database=[]


    while not rospy.is_shutdown():
        # for ML test only
        if ((latest_ML_result=="Bump Detected") or (latest_DL_result>0)) :
        # for ML & DL test
        #if ((latest_ML_result=="Bump Detected") or (latest_DL_result>0)) :

            latest_ML_result=None
            latest_DL_result=0
            os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
            if (GPS_coord[0]!=None)  and (GPS_coord[0]!=None):
                try:
                    database=read_database(database_dir)
                    check_result,database_2=check_bump_exist(database,GPS_coord,min_dist=15)
                except IOError:
                    check_result=0
                #check_result=0
                if (check_result==0):
                    print("saving Bump location to database")
                    database.append(GPS_coord)
                    save_coord_to_database(GPS_coord)
                elif(check_result==1):
                    overwrite_db(database_2,database_dir)

        rate.sleep()
  except rospy.ROSInterruptException:
    pass
 