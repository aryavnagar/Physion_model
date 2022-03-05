
import cv2
import os
count=1
import time 

start_time = time.time()

vidcap = cv2.VideoCapture(r"C:\Users\aryav\Downloads\yt5s.com-Our Story in 1 Minute(360p).mp4")
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        dim = (640,360) # you can change image height and image width
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite("images/"+str(count)+".png", resized) # image write to image folder be sure crete image folder in same dir
    return hasFrames
sec = 0
frameRate = 0.2 # change frame rate as you wish, ex : 30 fps => 1/30

success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
    


print("My program took", time.time() - start_time, "to run")
  
  
# ----------------------


import os
import time 

start_time = time.time()


folder = 'physionTest'  
os.mkdir(folder)
# use opencv to do the job
import cv2
print(cv2.__version__)  # my version is 3.1.0
vidcap = cv2.VideoCapture(r"C:\Users\aryav\Downloads\Y2Mate.is - Planks For Beginners  Proper Form + 10 Plank Exercises-DjEN3SKl0Eg-1080p-1645432239151_Trim.mp4")
count = 0
while True:
    success,image = vidcap.read()
    if not success:
        break
    cv2.imwrite(os.path.join(folder,"frame{:d}.jpg".format(count)), image)     # save frame as JPEG file
    count += 1
print("{} images are extacted in {}.".format(count,folder))


print("My program took", time.time() - start_time, "to run")
