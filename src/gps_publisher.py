#! /usr/bin/env python3

import rospy
import time
#import rospkg
#import roslib
import adafruit_gps
import serial
from sensor_msgs.msg import NavSatFix


uart = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=10)
gps = adafruit_gps.GPS(uart)     # Use UART/pyserial
#gps.send_command(b'PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
#gps.send_command(b'PMTK220,1000')
timestamp = time.monotonic()

rospy.init_node('gps_node')
pub = rospy.Publisher('/gps_data',NavSatFix , queue_size =1)
rate =rospy.Rate(5)

gps_data = NavSatFix()
cv = [0.]*9
msg_seq=0
while  not rospy.is_shutdown():
        data = gps.read(300)  # read up to 300 bytes
        if data is not None:
                data_string = ''.join([chr(b) for b in data])
                index_GNRMC = data_string.find('GNRMC')
                index_GNGGA = data_string.find('GNGGA')
                #print(result)
                if (index_GNRMC >= 0 ) & (index_GNRMC <(300-43)):
                        hour = int(data_string[index_GNRMC+6:index_GNRMC+8])
                        hour = hour +2
                        if hour == 25:
                                hour = 1
                        elif hour == 24:
                                hour = 0
                        minute = data_string[index_GNRMC+8:index_GNRMC+10]
                        sec = data_string[index_GNRMC+10:index_GNRMC+12]
                        ValidFlag = data_string[index_GNRMC+16]
                        print('\ntime is   : ',hour ,minute ,sec)
                        if ValidFlag == 'A':  #data is valid
                                print("Data is valid")
                                Latitude = data_string[index_GNRMC+18:index_GNRMC+20]+'.'+data_string[index_GNRMC+20:index_GNRMC+22]\
                                +data_string[index_GNRMC+23:index_GNRMC+28]
                                Longitude = data_string[index_GNRMC+31:index_GNRMC+34]+'.'+data_string[index_GNRMC+34:index_GNRMC+36]\
                                +data_string[index_GNRMC+37:index_GNRMC+41]
                                if (index_GNGGA >= 0 ) & (index_GNGGA <(300-58)):
                                	altitude = data_string[index_GNGGA+53:index_GNGGA+57]
                                else:
                                	altitude = '0'	
                                if (Latitude != '.') & (Longitude != '.') & (altitude != '.') :
                                        Longitude = float(Longitude)
                                        Latitude = float(Latitude)
                                        index = altitude.find(',')
                                        if index != -1:
                                        	altitude = altitude[:index]
                                        altitude = float(altitude)
										
                                        gps_data.latitude = Latitude
                                        gps_data.longitude = Longitude
                                        gps_data.altitude= altitude
                                        gps_data.header.frame_id = "base_link" #changed 0128 from base_footprint
                                        gps_data.status.status = 0
                                        gps_data.status.service = 1
                                        timeob = rospy.get_rostime()
                                        gps_data.header.stamp=timeob
                                        gps_data.header.seq=msg_seq
                                        gps_data.position_covariance = cv
                                        gps_data.position_covariance_type =0
                                        pub.publish(gps_data) 
                                        msg_seq+=1                           			
                                        print('Latitude  :',Latitude)
                                        print('Longitude :',Longitude)
                                        print('Altitude  :',altitude)
                                else:   
                                	print("Data is not valid")
                        else:
                        	print("Data didn't be sent")         	

        if time.monotonic() - timestamp > 5:
                #gps.send_command(b'PMTK605') 
                timestamp = time.monotonic() 
        rate.sleep()           
                
       # while not rospy.is_shutdown():
        	#pub.publish(gps_data)  
        #	     rate.sleep()   

                 



