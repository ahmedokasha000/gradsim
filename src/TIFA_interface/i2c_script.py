import smbus
import time
# Nvidia Jetson Nano i2c Bus 0
bus = smbus.SMBus(0)

# This is the address we setup in the Tiva C Program
address = 0x0f

def writeNumber(value):
    bus.write_byte(address, value)
    # bus.write_byte_data(address, 0, value)
    return -1
 
def readNumber(i):
    m = 0
    while m<i:
        number = bus.read_byte(address)
        print(number)
        m=m+1
    # number = bus.read_byte_data(address, 1)
    return number

while True:
    var = input("")
    if not var:
        continue
    writeNumber(var)
    if var ==0x50:
        number = readNumber(8)
    if var == 0x43:
        number = readNumber(1)
   
