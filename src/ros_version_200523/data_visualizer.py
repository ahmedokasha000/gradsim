import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#plt.style.use('fivethirtyeight')

x_vals = []
y_vals = []

index = count()


# def animate(i):
#     data = pd.read_csv('data.csv')
#     x = data['x_value']
#     y1 = data['acc_value']

#     plt.cla()

#     plt.plot(x, y1, label='Channel 1')
#     plt.legend(loc='upper left')
#     plt.Figure(figsize=(14, 12))
#     plt.tight_layout()


# ani = FuncAnimation(plt.gcf(), animate, interval=200)
data = pd.read_csv('normal_drive.csv')
# seq= data['seq_val']
#acc_x = data['acc_x']
#acc_y = data['acc_y']
acc_z = data['acc_z']
#std_x= data['std_acc_x']
#std_y= data['std_acc_y']
std_z= data['std_acc_z']
#std_all= data['std_acc_all']
#diff_x=data["diff_x"]
#diff_y=data["diff_y"]
#diff_all=data["diff_all"]
# x_mean=data["x_mean"]
# y_mean=data["y_mean"]
#plt.plot( acc_x,label="accel_x",color='b')
#plt.plot( acc_y,label="accel_y",color="g")
plt.plot( acc_z,label="accel_z",color="r")
# plt.plot( std_x,label="std_acc_x",color='c')
# plt.plot( std_y,label="std_acc_y",color="k")
plt.plot( std_z,label="std_acc_z",color="y")
# plt.plot( std_all,label="std_acc_all",color='m')
# plt.plot( diff_x,label="diff_x",color='orange')
# plt.plot( diff_y,label="diff_y",color='grey')
# plt.plot( diff_all,label="diff_all",color='dodgerblue')
# plt.plot( x_mean,label="x_mean",color='dodgerblue')
# plt.plot( y_mean,label="y_mean",color='navy')
plt.legend()
plt.tight_layout()
plt.show()