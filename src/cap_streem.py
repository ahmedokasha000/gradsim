import requests
import cv2
import numpy as np 
from time import sleep

url= "http://192.168.43.1:8080/shot.jpg"

while True :
    img_response =requests.get(url)
    img_arr=np.array(bytearray(img_response.content))
    img_res= cv2.imdecode(img_arr,-1)
    cv2.imshow('Phone',img_res)
    #sleep(0.1)
    if cv2.waitKey(1)==27 :
        break