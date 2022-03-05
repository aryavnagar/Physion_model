from Physion import *
import cv2
import os
count=1
import time 



start_time = time.time()

vidcap = cv2.VideoCapture(r"C:\Users\aryav\Desktop\Github\Physion\physiontestvideo.mp4")
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        dim = (640,360) # you can change image height and image width
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite("images/"+str(count)+".png", resized) # image write to image folder be sure crete image folder in same dir
    return hasFrames
sec = 0
frameRate = 0.3 # change frame rate as you wish, ex : 30 fps => 1/30

success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)

Plank()


print("My program took", time.time() - start_time, "to run")
  