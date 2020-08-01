#!/usr/bin/env python3

import numpy as np
import csv
class BumpDetection:
    def __init__(self, sample_frequency,log_data=False):
        self.SAMPLE_FREQ = sample_frequency
        self.WINDOW_TIME = 0.5 # moving windows length is over 0.5 second data
        self.window = int(sample_frequency * self.WINDOW_TIME)  
        self.last_seq = 0
        self.all_data = {
            "acc_x": [0] * self.window,
            "acc_y": [0] * self.window,
            "acc_z": [0] * self.window,
        }
        self.STD_THRESHOLD = 0.6  #standard deviation threshold for accident detection
        self.DIFF_THRESHOLD = 3.0 #minimum change in the acceleration component window measurements
                                  #if change in measurements through the window is more than threshold value this is an indication 
                                  #of possible accident
        self.LOG_TO_CSV=log_data
        if self.LOG_TO_CSV :
            file_header=["acc_x","acc_y","std_acc_x","std_acc_y","std_acc_all","diff_x","diff_y","diff_all"]
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
            "std_acc_y": [],
            "std_acc_z": [],
            "std_acc_all": [],
            "diff_z": [],
        }
        index = 0
        buff_size = len(new_buffer)
        for acc_measurement in new_buffer:
            self.all_data["acc_x"].append(acc_measurement[0]) 
            self.all_data["acc_y"].append(acc_measurement[1])
            self.all_data["acc_z"].append(acc_measurement[2])
            window_x = self.all_data["acc_x"][-self.window :] #the window for each measurements represent all previous values including this measurement
            window_y = self.all_data["acc_y"][-self.window :] # with a length of a window size
            window_z = self.all_data["acc_z"][-self.window :]
            results["std_acc_x"].append(np.std(window_x))
            results["std_acc_y"].append(np.std(window_y))
            results["std_acc_z"].append(np.std(window_z))
            #std_all=np.sqrt(results["std_acc_y"][index]**2+results["std_acc_x"][index]**2)
            #results["std_acc_all"].append(results["std_acc_z"][index])
            diff_z = np.max(window_z) - np.min(window_z)
            if abs(np.max(window_z)) < abs(np.min(window_z)):
                diff_z *= -1
                
            #results["diff_x"].append(diff_x)
            #results["diff_y"].append(diff_y)
            #diff_all=np.sqrt(results["diff_x"][index]**2+results["diff_y"][index]**2)
            results["diff_z"].append(diff_z)
            if self.LOG_TO_CSV:
                self.csv_writer.writerow([acc_measurement[0],
                                            acc_measurement[1],
                                            results["std_acc_x"][index],
                                            results["std_acc_y"][index],
                                            results["std_acc_z"][index],
                                            results["diff_x"][index],
                                            results["diff_y"][index],
                                            results["diff_z"][index]])
            index += 1
        final_output = self.buffer_analize(results)
        del self.all_data["acc_x"][:buff_size]
        del self.all_data["acc_y"][:buff_size]
        del self.all_data["acc_z"][:buff_size]
        return final_output

    def buffer_analize(self, buff_res):
        diff_max_comp = np.max(buff_res["diff_z"])
        diff_mcomp_ind = np.argmax(buff_res["diff_z"])
        #comp_angle = (np.arctan2(buff_res["diff_y"][diff_mcomp_ind], buff_res["diff_x"][diff_mcomp_ind])* 180/ np.pi)
        # severity=(diff_max_comp/self.DIFF_THRESHOLD)
        stedev_max = np.max(buff_res["std_acc_z"])
        if (stedev_max > self.STD_THRESHOLD) and (diff_max_comp > self.DIFF_THRESHOLD):
            final_output = {
                "bool": 1,
                "message": "Bump Detected",
                "accel_difference": diff_max_comp,
                "stdev": stedev_max,
            }
        else:
            final_output = {
                "bool": 0,
                "message": "No Bump",
                "accel_difference": diff_max_comp,
                "stdev": stedev_max,
            }
        return final_output

