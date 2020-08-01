import numpy as np
from acc_detection import AccDetection
import witmotion_imu as im
import serial

serial_port = serial.Serial(
    port="/dev/rfcomm2",
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)

# 0.2 diff
# 20 sample window and 0.2 stdev z
def main():
    #####################   Accident Detection Initialization   #####################  
    OUTPUT_RATE = 100
    Accident_ob = AccDetection(OUTPUT_RATE,log_data=False)
    Accident_ob.DIFF_THRESHOLD = 3.0  # set the acceleration change threshold (3g set as default)
    Accident_ob.STD_THRESHOLD = 0.6  # set the standard deviation threshold
    BUFFER_SIZE= 50 # should contain 0.5s measurements, 50 measurements in case of 100Hz output rate
    ####################################################################################
    last_seq = 0
    buffer=[]
    imu1=im.IMUDriver()
    while True:
        if serial_port.inWaiting() > 0:
            data = ord(serial_port.read())
            imu1.decode_format(data)
        if imu1.frame_seq > last_seq:
            row=[imu1.acceleration[0],imu1.acceleration[1],imu1.acceleration[2]]
            buffer.append(row)
            #csv_writer.writerow(info)
            last_seq += 1
        if len(buffer) >= BUFFER_SIZE:
            arr = np.array(buffer)
            #####################   Feed Algorithm with the new buffer array ############
            result = Accident_ob.process_buffer(arr)  
            print(result)
            buffer.clear()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting Program")
    finally:
        pass
