#!/usr/bin/env python3
import numpy as np
import pandas as pd
import scipy.signal as sg
from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error,precision_recall_curve, precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize

from sklearn.model_selection import StratifiedKFold as StK

pd.set_option("display.max_rows",50)
pd.set_option("display.max_columns",50)
pd.set_option("max_colwidth",20)

def butter_bandpass(data, rang, fs):
    """
    bandpass the data withn certain frequency range for all sensors
    Args:
        data: a list of sensors measurements
        range: tuple contain the min, max frequency
        fs: sample frequence for the signal
    Returns:
        return a list arrays for all the filtered signals in the time domain
    """
    results = []
    zeros, poles = sg.butter(3, rang, btype="bandpass", fs=fs)
    # filter each sensor measurement
    for key in data:
        results.append(sg.filtfilt(zeros, poles, key))
    return results




# in your prediction file
def load_model(model_path):
    """
    Loads a saved sklearn regression model or train new model
    Args:
    Returns:
        return the saved model
    """
    path_n = model_path
    if os.path.isfile(path_n):
        fil = open(path_n, "rb")
        prediction_model = pickle.load(fil)
        print("model found")
    else:
        prediction_model = model_train()
        print("new model is created")
    return prediction_model

def load_all_dataset(dir, fs, window_size=5, window_overlab=2):

    """
    Load data, labels from all files, windowing on each 6 seconds data with
    4 sec overlab each sample data is a window of 6 seconds and shifted 2 s
    from the previoues one
    Args:
        data_fls: all the files for the dataset sensor measurements
        ref_fls: all the files for the dataset sensor labels
        fs: sample frequency
    Returns: list of array for all samples in the dataset, each sample data
     has all the sensors measurements in this sample in the time domain
    """

    head=["acc_x","acc_y","acc_z","ang_vel_x","ang_vel_y","ang_vel_z","angle_x","angle_y","angle_z","bump_class"]
    df=pd.read_csv(dir,sep=",")
    df.columns=head

    label_peaks, _ = sg.find_peaks(
                df["bump_class"].values, distance=200
            )

    for peak in label_peaks:
        df["bump_class"].iloc[peak-250:peak+250]=1
        
    X = []
    Y = []
    window_len = window_size * fs
    window_shift_len = (window_size - window_overlab) * fs
  
    # Convert dataframe to samples each one is shifted 2s
    number_samples = int(len(df) / window_shift_len)
    for window_ind in range(number_samples):
        sample_start = int(window_ind * window_shift_len)
        sample_end = int(sample_start + window_len)
        acc_x = df["acc_x"][sample_start:sample_end].values
        acc_y = df["acc_y"][sample_start:sample_end].values
        acc_z = df["acc_z"][sample_start:sample_end].values
        ang_vel_x=df["ang_vel_x"][sample_start:sample_end].values
        ang_vel_y=df["ang_vel_y"][sample_start:sample_end].values
        ang_vel_z=df["ang_vel_z"][sample_start:sample_end].values
        ang_x=df["angle_x"][sample_start:sample_end].values
        ang_y=df["angle_y"][sample_start:sample_end].values
        ang_z=df["angle_z"][sample_start:sample_end].values
        labels = df["bump_class"][sample_start:sample_end].values
        # [acc_x, acc_y, acc_z, ang_vel_x,ang_vel_y,ang_vel_z,ang_x,ang_y,ang_z]
        X.append(np.array([acc_x, acc_y, acc_z,ang_vel_y,ang_y,ang_z]))
        Y.append(np.array([labels]))
    return X, Y

def clean_datset(samples, labels, fs):
    """
    Clean each seample data by performing bandpass filter on range of 0.7-4
    Hz and add FFT for each sample in the dataset
    Args:
        samples: list of array for each sample. each sample data has several
         sensor measurements
        labels: list of arrays for each sample labels
    Returns: tuble of dataset time and frequency data for all sensors, labels
     for each sample
    """
    clean_x = []
    clean_y = []
    for sample, label in zip(samples, labels):
        bandpass_filtered = butter_bandpass(sample, (0.001, 10), fs)
        fft_sample = fft_sensors(bandpass_filtered, fs)
        clean_x.append([bandpass_filtered,(bandpass_filtered, fft_sample)])
        clean_y.append([int(np.mean(label)>0.5)])
    return (clean_x, clean_y)
def fft_sensors(data, fs):
    """
    fast fourier transform for the each sensor data
    Args:
        data: a list of sensors measurements
        fs: sample frequence for the signal
    Returns:
        return a list of arrays for all the signals in the frequency domaoin
        data is on this format data - > sensor1_freq, sensor1_powers
                                    - > sensor2_freq, sensor2_powers
                                    - > sensorn_freq, sensorn_powers
    """
    ff_data = []
    # FFT each sensor measurement
    for key in data:
        ff_freq = np.fft.rfftfreq(len(key), 1.0 / fs)
        ff_power = np.fft.rfft(key)
        ff_data.append(np.array([ff_freq, ff_power]))
    return ff_data

'''data is on this format data ->sample   time_data
                                             - > acc_x
                                             - > acc_y
                                             - > acc_z
                                             - > ang_vel_x
                                             - > ang_vel_y
                                             - > ang_vel_z
                                             - > ang_x
                                             - > ang_y
                                             - > ang_z
                                           freq_data
                                             - > acc_x freqs ,acc_x power 
                                             - > acc_y freqs, acc_y power
                                             - > acc_z freqs, acc_z power
                                             - > ang_vel_x freqs, ang_vel_x power
                                             - > ang_vel_y freqs, ang_vel_y power
                                             - > ang_vel_z freqs, ang_vel_z power
                                             - > ang_x freqs, ang_x power
                                             - > ang_y freqs, ang_y power
                                             - > ang_z freqs, ang_z power
'''                                                 


def featurize_samples(samples, fs):
    """
    time and freq featurization for each sample sensors data
    Args:
        samples: a list of the dataset samples
        fs: sample frequency
    Returns: return a list of arrays for each sample features
    """
    features = []
    for sample in samples:
        # extract each sample to each sensor time & freq data
        time_data = sample[0]
        freqs = np.abs(sample[1][0][0])
        freq_data=[np.abs(sensor_freq_power[1]) for sensor_freq_power in sample[1]]
        #  average freq power for all accel axes
        # Time features
        min_vals = [np.min(col_data) for col_data in time_data]
        max_vals = [np.max(col_data) for col_data in time_data]
        mean_vals = [np.mean(col_data) for col_data in time_data]
        median_vals=[np.median(col_data) for col_data in time_data]
        std_vals = [np.std(col_data) for col_data in time_data]
        var_vals = [np.var(col_data) for col_data in time_data]
        percentile_5=[np.percentile(col_data, 5) for col_data in time_data]
        percentile_10=[np.percentile(col_data, 10) for col_data in time_data]
        percentile_25=[np.percentile(col_data, 25) for col_data in time_data]
        percentile_75=[np.percentile(col_data, 75) for col_data in time_data]
        percentile_90=[np.percentile(col_data, 90) for col_data in time_data]
        percentile_95=[np.percentile(col_data, 95) for col_data in time_data]
        time_features =[]
        time_features.extend(min_vals)
        time_features.extend(max_vals)
        time_features.extend(median_vals)
        time_features.extend(mean_vals)
        time_features.extend(std_vals)
        time_features.extend(var_vals)
        time_features.extend(percentile_5)
        time_features.extend(percentile_10)
        time_features.extend(percentile_25)
        time_features.extend(percentile_75)
        time_features.extend(percentile_90)
        time_features.extend(percentile_95)

        total_features = time_features
        features.append(np.array(total_features))
    return(features)

def model_train(estimators=650, depth=14, file_path="model_1"):
    """
    train a random forest regressor on the dataset and save it in the
     provided path
    Args:
        estimators: number of trees in the model
        depth: single value for the depth of each tree of the model
        file_path: a path string to which model will be saved
    Returns: return the trained regression model
    """
    # Reading ref and sensors data, create timestamp for both
    fs_imu=100
    current_dir=os.getcwd().split('/')
    dir1="/home/"+current_dir[2]+"/catkin_ws/src/gradsim/src/bumb_detect_IMU/dataset/dataset_20_08_06.csv"
    data_x,data_y=load_all_dataset(dir1, fs_imu, window_size=5, window_overlab=2)
    clean_x,clean_y=clean_datset(data_x, data_y, fs_imu)
    dataset_feats=featurize_samples(clean_x, fs_imu)
#     train_x, test_x, train_y, test_y = train_test_split(
#         dataset_feats, clean_y, random_state=15, test_size=0.2
#     )
    #print(dataset_feats.shape)
    dataset_feats=np.array(dataset_feats)
    
    clean_y=np.ravel(clean_y)
    
    folds = StK(n_splits=5)
    y_true=[]
    y_pred=[]
    for train_index, test_index in folds.split(dataset_feats, clean_y):
        X_train, X_test = dataset_feats[train_index], dataset_feats[test_index]
        y_train, y_test = clean_y[train_index], clean_y[test_index]
        clf = RandomForestRegressor(
            n_estimators=estimators, max_depth=depth, random_state=15,
        )
        clf.fit(X_train,y_train)
        y_true.extend(list(y_test))
        y_pred.extend(clf.predict(X_test))
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    
    with open(file_path, "wb") as f:
        pickle.dump(clf, f)
        print("model saved in the following dir: %s" % file_path)
    return clf,{"y_true":y_true,"y_pred":y_pred}

def choose_best_threshold(results):
    precision, recall, thresholds = precision_recall_curve(results["y_true"],
                                                           results["y_pred"])
    F1 = 2 * (precision * recall) / (precision + recall)
    best_thresh_ind=np.argmax(F1)
    best_thresh=thresholds[best_thresh_ind]
    print("at best f-score")
    print("best thresh = ",best_thresh)
    return best_thresh

def evaluate_res(results):
    mae_val = mean_absolute_error(results["y_true"], results["y_pred"])  # *60 toconvert from Hz t BPM
    mse_val = mean_squared_error(results["y_true"], results["y_pred"])  # *60 to convert from Hz t BPM
    conf_mat=confusion_matrix(results["y_true"], results["y_pred"],labels=[1,0])
    
    print("Model Results : \n")
    print("mean absolute error", mae_val)
    print("mean square error", mse_val)
    print("\nConfusion matrix")
    print("True\Prediction\t\t1\t\t0")
    print("1\t\t\t%d\t\t%d"%(conf_mat[0,0],conf_mat[0,1]))
    print("0\t\t\t%d\t\t%d"%(conf_mat[1,0],conf_mat[1,1]))
    print("\n",classification_report(results["y_true"], results["y_pred"],labels=[0,1],
                            target_names=["No Bump","Bump"]))

def inference(imu_data, model_path="model_1"):
    """
    load regresssion model, perform prediction over latest 5 sec of imu data
    return prediction and confidence for each sample
    Args:
        data_fl: file name for sensor measurements
        model_path: a path string to which model will be saved
    Returns: a data for predictions and confidence of each 2s measurement
    """
    fs_imu = 100
    labels=np.zeros(len(imu_data))
    clean_x,clean_y=clean_datset([imu_data], [labels], fs_imu)
    dataset_feats=featurize_samples(clean_x, fs_imu)
    dataset_feats=np.array(dataset_feats[0]).reshape(1,-1)
    clean_y = np.ravel(clean_y)
    reg_model = load_model(model_path)
    samples_pred = reg_model.predict(dataset_feats)
    return (samples_pred>0.4116).astype(int)

def inference_2(imu_data, model):
    """
    load regresssion model, perform prediction over latest 5 sec of imu data
    return prediction and confidence for each sample
    Args:
        data_fl: file name for sensor measurements
        model_path: a path string to which model will be saved
    Returns: a data for predictions and confidence of each 2s measurement
    """
    fs_imu = 100
    labels=np.zeros(len(imu_data))
    clean_x,clean_y=clean_datset([imu_data], [labels], fs_imu)
    dataset_feats=featurize_samples(clean_x, fs_imu)
    dataset_feats=np.array(dataset_feats[0]).reshape(1,-1)
    clean_y = np.ravel(clean_y)
    reg_model = model
    samples_pred = reg_model.predict(dataset_feats)
    
    return (samples_pred>0.4116).astype(int)
