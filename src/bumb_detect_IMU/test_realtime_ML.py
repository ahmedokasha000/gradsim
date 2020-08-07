import numpy as np
#from bump_detection import BumpDetection
import witmotion_imu as im
import os
import serial
import time
from bump_detection_ML import inference_2,load_model

ports= ["/dev/rfcomm"+str(i) for i in range(10)]
ports.extend(["/dev/ttyTHS1","/dev/ttyTHS2","/dev/rfcomm1"])
for port in ports:
    try:
        serial_port = serial.Serial(
            port=port,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        if (serial_port.isOpen()):
            print("port %s is open"%port)
            break
    except:
        print("port %s isn't found"%port)


duration = 0.4  # seconds
freq = 1500  # Hz
os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

if __name__ == '__main__':
  try:
    imu1=im.IMUDriver()
    #####################   Accident Detection Initialization   #####################  
    OUTPUT_RATE = 100
    BUFFER_SIZE= 500 # should contain 0.5s measurements, 50 measurements in case of 100Hz output rate
    ####################################################################################
    last_seq = 0
    buffer=[]
    model= load_model("model_1")
    while (1):
        if serial_port.inWaiting() > 0:
            data = ord(serial_port.read())
            imu1.decode_format(data)
        if imu1.frame_seq > last_seq:
            row=(imu1.acceleration,imu1.angular_velocity,imu1.eular_angles)
            buffer.append([imu1.acceleration[0],imu1.acceleration[1],imu1.acceleration[2],imu1.angular_velocity[1],imu1.eular_angles[1],imu1.eular_angles[2]])
            #csv_writer.writerow(info)
            last_seq += 1
        if len(buffer) >= BUFFER_SIZE:
            
            arr = np.array(buffer)
            input_d=[arr[:,i] for i in range(6)]
            #rint(input_d)
            #####################   Feed Algorithm with the new buffer array ############
            result=inference_2(input_d,model)
            if(result[0]):
                print("Bump Detected")
                os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
            else:
                print("No Bump")
            del buffer[:300]
            
            # print(result["stdev"],result["accel_difference"])
        
       

  except :
    print("imu port closing...")
    serial_port.close()
    pass
