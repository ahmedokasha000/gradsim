#!/usr/bin/env python3

import numpy as np
import math as m
class IMUDriver ():
    def __init__(self):
        self.acceleration=np.float_([0., 0., 0.])
        self.angular_velocity=np.float_([0., 0., 0.])
        self.eular_angles=np.float_([0., 0., 0.])
        self.frame_seq=0
        self.temerature=0.0
        self.byte_counter=0
        self.start_byte=0
        self.bytes_r = [0]*11
    def decode_bytes(self,bytes_r):
        temparr = np.int16([0, 0, 0, 0])
        temparr[0] = (bytes_r[3] << 8 | bytes_r[2])
        temparr[1] = (bytes_r[5] << 8 | bytes_r[4])
        temparr[2] = (bytes_r[7] << 8 | bytes_r[6])
        temparr[3] = (bytes_r[9] << 8 | bytes_r[8])
        if bytes_r[1] == 0x51:

            self.acceleration[0] = temparr[0] / 32768.0 * 16
            self.acceleration[1] = temparr[1] / 32768.0 * 16
            self.acceleration[2]  = temparr[2] / 32768.0 * 16

            self.temerature  = temparr[3] / 340.0 + 36.25
        elif bytes_r[1] == 0x52:
            self.angular_velocity[0]= temparr[0] / 32768.0 * 2000
            self.angular_velocity[1]= temparr[1] / 32768.0 * 2000
            self.angular_velocity[2]= temparr[2] / 32768.0 * 2000
            self.temerature = temparr[3] / 340.0 + 36.25
            self.frame_seq += 1
            #print("orientation angles around x y z= ",self.eular_angles)
            #print("acceleration around x y z= ",self.acceleration)

        elif bytes_r[1] == 0x53:
            self.eular_angles[0] = temparr[0] / 32768.0 * 180.0#(m.pi)
            self.eular_angles[1] = temparr[1] / 32768.0 * 180.0#(m.pi)
            self.eular_angles[2] = temparr[2] / 32768.0 * 180.0#(m.pi)
            self.temerature  = temparr[3] / 340.0 + 36.25

    def decode_format(self,data):
        if (data == 0x55) and (self.byte_counter == 0):
                self.start_byte = 1
        if self.start_byte == 1:
            self.bytes_r[self.byte_counter] = data
            self.byte_counter += 1
            if self.byte_counter == 11:
                self.byte_counter = 0
                self.start_byte = 0
                self.decode_bytes(self.bytes_r)
                self.bytes_r = [0]*11
