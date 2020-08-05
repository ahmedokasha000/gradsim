#! /usr/bin/env python3
import time
import adafruit_gps
import serial
import rospy
from sensor_msgs.msg import NavSatFix


ports= ["/dev/ttyUSB"+str(i) for i in range(5)]

for port in ports:
    try:
        uart = serial.Serial(
            port=port,
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=10,
        )
        if (uart.isOpen()):
            print("port %s is open"%port)
            break
    except:
        print("port %s isn't found"%port)
    


rospy.init_node('gps_node')
pub = rospy.Publisher('/gps_data',NavSatFix , queue_size =1)
rate =rospy.Rate(5)
gps_data = NavSatFix()
cv = [0.]*9
msg_seq=0
# Create a GPS module instance.
gps = adafruit_gps.GPS(uart, debug=False)     # Use UART/pyserial



# Turn on the basic GGA and RMC info (what you typically want)
gps.send_command(b'PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
# Turn on just minimum info (RMC only, location):
#gps.send_command(b'PMTK314,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
# Turn off everything:
#gps.send_command(b'PMTK314,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
# Tuen on everything (not all of it is parsed!)
#gps.send_command(b'PMTK314,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0')

# Set update rate to once a second (1hz) which is what you typically want.
gps.send_command(b'PMTK220,1000')


# Main loop runs forever printing the location, etc. every second.
last_print = time.monotonic()
while  not rospy.is_shutdown():
    # Make sure to call gps.update() every loop iteration and at least twice
    # as fast as data comes from the GPS unit (usually every second).
    # This returns a bool that's true if it parsed new data (you can ignore it
    # though if you don't care and instead look at the has_fix property).
    gps.update()
    # Every second print out current location details if there's a fix.
    current = time.monotonic()
    if current - last_print >= 1.0:
        last_print = current
        if not gps.has_fix:
            # Try again if we don't have a fix yet.
            print('Waiting for fix...')
            continue
        # We have a fix! (gps.has_fix is true)
        # Print out details about the fix like location, date, etc.
        print('=' * 40)  # Print a separator line.
        print('Fix timestamp: {}/{}/{} {:02}:{:02}:{:02}'.format(
            gps.timestamp_utc.tm_mon,   # Grab parts of the time from the
            gps.timestamp_utc.tm_mday,  # struct_time object that holds
            gps.timestamp_utc.tm_year,  # the fix time.  Note you might
            gps.timestamp_utc.tm_hour,  # not get all data like year, day,
            gps.timestamp_utc.tm_min,   # month!
            gps.timestamp_utc.tm_sec))
        print('Latitude: {0:.6f} degrees'.format(gps.latitude))
        print('Longitude: {0:.6f} degrees'.format(gps.longitude))
        print('Fix quality: {}'.format(gps.fix_quality))
        # Some attributes beyond latitude, longitude and timestamp are optional
        # and might not be present.  Check if they're None before trying to use!
        if gps.satellites is not None:
            print('# satellites: {}'.format(gps.satellites))
        if gps.altitude_m is not None:
            print('Altitude: {} meters'.format(gps.altitude_m))
            gps_data.altitude= gps.altitude_m
        if gps.speed_knots is not None:
            print('Speed: {} knots'.format(gps.speed_knots))
        if gps.track_angle_deg is not None:
            print('Track angle: {} degrees'.format(gps.track_angle_deg))
        if gps.horizontal_dilution is not None:
            print('Horizontal dilution: {}'.format(gps.horizontal_dilution))
        if gps.height_geoid is not None:
            print('Height geo ID: {} meters'.format(gps.height_geoid))
        gps_data.latitude = gps.latitude
        gps_data.longitude = gps.longitude

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