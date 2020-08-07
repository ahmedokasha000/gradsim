#!/usr/bin/env python3

import numpy as np
import csv
class BumpDetection:
    def __init__(self, sample_frequency,log_data=False):
        self.SAMPLE_FREQ = sample_frequency
        self.WINDOW_TIME =2.0 # moving windows length is over 0.5 second data
        self.window = int(sample_frequency * self.WINDOW_TIME)  
        self.last_seq = 0
        self.all_data = {
            "acc_x": [0] * self.window,
            "acc_y": [0] * self.window,
            "acc_z": [0] * self.window,
            "ang_vel_y":[0] * self.window,
            "angle_y":[0] * self.window,
        }
        self.STD_THRESHOLD = 0.6  #standard deviation threshold for accident detection
        self.DIFF_THRESHOLD = 3.0 #minimum change in the acceleration component window measurements
                                  #if change in measurements through the window is more than threshold value this is an indication 
                                  #of possible accident
        self.LOG_TO_CSV=log_data
        if self.LOG_TO_CSV :
            file_header=["std_acc_x","std_acc_y","std_acc_z","std_vel_y","std_angle_y"]
            self.file_name="data.csv"
            self.f1=open("data.csv","w")
            self.csv_writer=csv.writer(self.f1)
            self.csv_writer.writerow(file_header)
    def process_buffer(self, new_buffer):
        #this function takes the new buffer data as input and perform a moving window algorithm
        #to get the maximum standard deviation and difference in the measurements through this window
        final_output = {}
        results = {
            "std_acc_x": [],
            "diff_angle_y": [],
            "std_acc_z": [],
            "std_vel_y": [],
            "std_angle_y": [],
        }
        index = 0
        buff_size = len(new_buffer)
        for acc_measurement in new_buffer:
            self.all_data["acc_x"].append(acc_measurement[0]) 
            self.all_data["acc_y"].append(acc_measurement[1])
            self.all_data["acc_z"].append(acc_measurement[2])
            self.all_data["ang_vel_y"].append(acc_measurement[3])
            self.all_data["angle_y"].append(acc_measurement[4])
            window_x = self.all_data["acc_x"][-self.window :] #the window for each measurements represent all previous values including this measurement
            window_y = self.all_data["acc_y"][-self.window :] # with a length of a window size
            window_z = self.all_data["acc_z"][-self.window :]
            window_vel_y = self.all_data["ang_vel_y"][-self.window :]
            window_ang_y = self.all_data["angle_y"][-self.window :]
            results["std_acc_x"].append(np.std(window_x))
            results["diff_angle_y"].append(np.abs(np.max(window_ang_y)-np.min(window_ang_y)))
            results["std_acc_z"].append(np.abs(np.max(window_z)-np.min(window_z)))
            results["std_vel_y"].append(np.std(window_vel_y))
            results["std_angle_y"].append(np.std(window_ang_y))

            if self.LOG_TO_CSV:
                self.csv_writer.writerow([acc_measurement[0],
                                            acc_measurement[1],
                                            results["std_acc_x"][index],
                                            results["diff_angle_y"][index],
                                            results["std_acc_z"][index],
                                            results["std_vel_y"][index],
                                            results["std_angle_y"][index]])
            index += 1
        final_output = self.buffer_analize(results)
        del self.all_data["acc_x"][:buff_size]
        del self.all_data["acc_y"][:buff_size]
        del self.all_data["acc_z"][:buff_size]
        del self.all_data["ang_vel_y"][:buff_size]
        del self.all_data["angle_y"][:buff_size]
        return final_output

    def buffer_analize(self, buff_res):
        stdev_acc_x = np.max(buff_res["std_acc_x"])
        diff_angle_y = np.max(buff_res["diff_angle_y"])
        stdev_acc_z = np.max(buff_res["std_acc_z"])
        stdev_vel_y = np.max(buff_res["std_vel_y"])
        stdev_ang_y = np.max(buff_res["std_angle_y"])
        if (stdev_acc_z > 0.4) and (diff_angle_y > 6.0) and(stdev_ang_y>3.0):
            final_output = {
                "bool": 1,
                "message": "Bump Detected",
                "deff_acc_z": stdev_acc_z,
                "diff_angle_y": diff_angle_y,
                "stdev_ang_y": stdev_ang_y,
            }
        else:
            final_output = {
                "bool": 0,
                "message": "No Bump",
                "deff_acc_z": stdev_acc_z,
                "diff_angle_y": diff_angle_y,
                "stdev_ang_y": stdev_ang_y,
            }
        # final_output = {
        #     "std_acc_x": stdev_acc_x,
        #     "diff_angle_y": diff_angle_y,
        #     "std_acc_z": stdev_acc_z,
        #     "std_vel_y": stdev_vel_y,
        #     "std_angle_y": stdev_ang_y,
        # }
        return final_output

