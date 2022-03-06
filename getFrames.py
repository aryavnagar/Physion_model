import cv2

def Frames(video_path):
    
    vidcap = cv2.VideoCapture(video_path)
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            cv2.imwrite("images/"+f"{count:04d}.jpg", image)     # save frame as JPG file
        return hasFrames
    sec = 0
    frameRate = 0.3 #//it will capture image in each 0.5 second
    count=1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)