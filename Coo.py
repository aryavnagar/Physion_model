import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import numpy as np

anglekneerightarray = []
anglekneeleftarray = []

anglehiprightarray = []
anglehipleftarray = []

angleshoulderrightarray = []
angleshoulderleftarray = []

angleelbowrightarray = []
angleelbowleftright = []


def findPosition(image, draw=True):
    
    lmList = []
    
    if results.pose_landmarks:
        
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = image.shape
            # print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
            
            if draw:
                cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                
    return lmList


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

# For webcam input:
cap = cv2.VideoCapture(r"C:\Users\aryav\Desktop\Github\Physion\physiontestvideo.mp4")
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
   
    if results.pose_landmarks:
        
        lmList = findPosition(image, draw=False)

        landmarks = results.pose_landmarks.landmark
        
        left_pinky = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x,landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y]
        right_pinky = [landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]
        
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y] #11
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y] #13
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y] #15
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y] #23
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y] #25
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y] #27
        left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y] #29
        
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y] #11
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y] #13
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y] #15
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y] #23
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y] #25
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y] #27
        right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y] #
        
        
        # Check if any landmarks are found.
        if results.pose_landmarks:
            
            left_side_distance = ({results.pose_landmarks.landmark[mp_pose.PoseLandmark(13).value].z}) #left elbow
            right_side_distance = ({results.pose_landmarks.landmark[mp_pose.PoseLandmark(14).value].z}) #right elbow
            
            left_side_distance = list(left_side_distance)
            right_side_distance = list(right_side_distance)
            
            left_side_distance = left_side_distance[0]
            right_side_distance = right_side_distance[0]
            
            
            if (right_side_distance < left_side_distance):
                
                right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                right_wrist_angle = calculate_angle(right_elbow, right_wrist, right_pinky)
                right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                right_ankle_angle = calculate_angle(right_knee, right_ankle, right_heel)
                
                # print("right_sholder_angle: " + str(right_sholder_angle))
                # print("right_elbow_angle: " + str(right_elbow_angle))
                # print("right_wrist_angle: " + str(right_wrist_angle))
                # print("right_hip_angle: " + str(right_hip_angle))
                # print("right_knee_angle: " + str(right_knee_angle))
                # print("right_ankle_angle: " + str(right_ankle_angle))
                
                anglekneerightarray.append(right_knee_angle)
                anglehiprightarray.append(right_hip_angle)
                angleshoulderrightarray.append(right_shoulder_angle)
                angleelbowrightarray.append(right_elbow_angle)
                
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y] #11
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y] #13
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y] #15
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y] #23
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y] #25
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y] #27
                right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
                
                cv2.line(image,(lmList[30][1],lmList[30][2]),(lmList[28][1],lmList[28][2]),(255,255,255),1)
                cv2.line(image,(lmList[28][1],lmList[28][2]),(lmList[26][1],lmList[26][2]),(255,255,255),1)
                cv2.line(image,(lmList[26][1],lmList[26][2]),(lmList[24][1],lmList[24][2]),(255,255,255),1)
                cv2.line(image,(lmList[24][1],lmList[24][2]),(lmList[12][1],lmList[12][2]),(255,255,255),1)
                cv2.line(image,(lmList[12][1],lmList[12][2]),(lmList[14][1],lmList[14][2]),(255,255,255),1)
                cv2.line(image,(lmList[14][1],lmList[14][2]),(lmList[16][1],lmList[16][2]),(255,255,255),1)
                cv2.line(image,(lmList[16][1],lmList[16][2]),(lmList[22][1],lmList[22][2]),(255,255,255),1)
                cv2.line(image,(lmList[16][1],lmList[16][2]),(lmList[18][1],lmList[18][2]),(255,255,255),1)
                cv2.line(image,(lmList[16][1],lmList[16][2]),(lmList[20][1],lmList[20][2]),(255,255,255),1)
                cv2.line(image,(lmList[18][1],lmList[18][2]),(lmList[20][1],lmList[20][2]),(255,255,255),1)
                cv2.line(image,(lmList[30][1],lmList[30][2]),(lmList[32][1],lmList[32][2]),(255,255,255),1)
                cv2.line(image,(lmList[32][1],lmList[32][2]),(lmList[28][1],lmList[28][2]),(255,255,255),1)
            
#
                right_hip_half = (lmList[12][1]-lmList[24][1])/8
                right_hip_half = right_hip_half + lmList[24][1]

                right_hip_calc1 = (lmList[12][2]-lmList[24][2])/8
                right_hip_calc1 = right_hip_calc1 + lmList[24][2]

                right_hip_calc2 = (lmList[26][1]-lmList[24][1])/8
                right_hip_calc2 = right_hip_calc2 + lmList[24][1]

                right_hip_calc3 = (lmList[26][2]-lmList[24][2])/8
                right_hip_calc3 = right_hip_calc3 + lmList[24][2]
                
                if right_hip_angle < 134 or right_hip_angle > 142:

                    cv2.line(image, (lmList[24][1],lmList[24][2]), (int(right_hip_half), int(right_hip_calc1)) , (0, 255, 0), 3)
                    cv2.line(image, (lmList[24][1],lmList[24][2]), (int(right_hip_calc2), int(right_hip_calc3)) , (0, 255, 0), 3)
                    
                else:
                    cv2.line(image, (lmList[24][1],lmList[24][2]), (int(right_hip_half), int(right_hip_calc1)) , (255, 0, 0), 3)
                    cv2.line(image, (lmList[24][1],lmList[24][2]), (int(right_hip_calc2), int(right_hip_calc3)) , (255, 0, 0), 3)
#

#
                right_knee_half = (lmList[24][1]-lmList[26][1])/8
                right_knee_half = right_knee_half + lmList[26][1]

                right_knee_calc1 = (lmList[24][2]-lmList[26][2])/8
                right_knee_calc1 = right_knee_calc1 + lmList[26][2]

                right_knee_calc2 = (lmList[28][1]-lmList[26][1])/8
                right_knee_calc2 = right_knee_calc2 + lmList[26][1]

                right_knee_calc3 = (lmList[28][2]-lmList[26][2])/8
                right_knee_calc3 = right_knee_calc3 + lmList[26][2]
                
                if right_knee_angle < 153 or right_knee_angle > 160:

                    cv2.line(image, (lmList[26][1],lmList[26][2]), (int(right_knee_half), int(right_knee_calc1)) , (0, 255, 0), 3)
                    cv2.line(image, (lmList[26][1],lmList[26][2]), (int(right_knee_calc2), int(right_knee_calc3)) , (0, 255, 0), 3)
                    
                else:
                    
                    cv2.line(image, (lmList[26][1],lmList[26][2]), (int(right_knee_half), int(right_knee_calc1)) , (255, 0, 0), 3)
                    cv2.line(image, (lmList[26][1],lmList[26][2]), (int(right_knee_calc2), int(right_knee_calc3)) , (255, 0, 0), 3)
#

#
                right_sholder_half = (lmList[14][1]-lmList[12][1])/8
                right_sholder_half = right_sholder_half + lmList[12][1]

                right_sholder_calc1 = (lmList[14][2]-lmList[12][2])/8
                right_sholder_calc1 = right_sholder_calc1 + lmList[12][2]

                right_sholder_calc2 = (lmList[24][1]-lmList[12][1])/8
                right_sholder_calc2 = right_sholder_calc2 + lmList[12][1]

                right_sholder_calc3 = (lmList[24][2]-lmList[12][2])/8
                right_sholder_calc3 = right_sholder_calc3 + lmList[12][2]

                if right_shoulder_angle < 93 or right_shoulder_angle > 105:

                    cv2.line(image, (lmList[12][1],lmList[12][2]), (int(right_sholder_half), int(right_sholder_calc1)) , (0, 255, 0), 3)
                    cv2.line(image, (lmList[12][1],lmList[12][2]), (int(right_sholder_calc2), int(right_sholder_calc3)) , (0, 255, 0), 3)    
                    
                else:
                    
                    cv2.line(image, (lmList[12][1],lmList[12][2]), (int(right_sholder_half), int(right_sholder_calc1)) , (0, 255, 0), 3)
                    cv2.line(image, (lmList[12][1],lmList[12][2]), (int(right_sholder_calc2), int(right_sholder_calc3)) , (0, 255, 0), 3)  
#

#
                right_elbow_half = (lmList[12][1]-lmList[14][1])/8
                right_elbow_half = right_elbow_half + lmList[14][1]

                right_elbow_calc1 = (lmList[12][2]-lmList[14][2])/8
                right_elbow_calc1 = right_elbow_calc1 + lmList[14][2]

                right_elbow_calc2 = (lmList[16][1]-lmList[14][1])/8
                right_elbow_calc2 = right_elbow_calc2 + lmList[14][1]

                right_elbow_calc3 = (lmList[16][2]-lmList[14][2])/8
                right_elbow_calc3 = right_elbow_calc3 + lmList[14][2]

                if right_elbow_angle < 68 or right_elbow_angle > 75:

                    cv2.line(image, (lmList[14][1],lmList[14][2]), (int(right_elbow_half), int(right_elbow_calc1)) , (0, 255, 0), 3)
                    cv2.line(image, (lmList[14][1],lmList[14][2]), (int(right_elbow_calc2), int(right_elbow_calc3)) , (0, 255, 0), 3) 
                
                else:
                    
                    cv2.line(image, (lmList[14][1],lmList[14][2]), (int(right_elbow_half), int(right_elbow_calc1)) , (0, 255, 0), 3)
                    cv2.line(image, (lmList[14][1],lmList[14][2]), (int(right_elbow_calc2), int(right_elbow_calc3)) , (0, 255, 0), 3) 
#

            elif (left_side_distance < right_side_distance):
            
                left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                left_wrist_angle = calculate_angle(left_elbow, left_wrist, left_pinky)
                left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                left_ankle_angle = calculate_angle(left_knee, left_ankle, left_heel)
        
                # print("left_sholder_angle: " + str(left_sholder_angle))
                # print("left_elbow_angle: " + str(left_elbow_angle))
                # print("left_wrist_angle: " + str(left_wrist_angle))
                # print("left_hip_angle: " + str(left_hip_angle))
                # print("left_knee_angle: " + str(left_knee_angle))
                # print("left_ankle_angle: " + str(left_ankle_angle))
                
                anglekneeleftarray.append(left_knee_angle)
                anglehipleftarray.append(left_hip_angle)
                angleshoulderleftarray.append(left_shoulder_angle)
                angleelbowleftright.append(left_elbow_angle)
                
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y] #11
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y] #13
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y] #15
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y] #23
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y] #25
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y] #27
                left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y] #29
                
                cv2.line(image,(lmList[29][1],lmList[29][2]),(lmList[27][1],lmList[27][2]),(255,255,255),1)
                cv2.line(image,(lmList[27][1],lmList[27][2]),(lmList[25][1],lmList[25][2]),(255,255,255),1)
                cv2.line(image,(lmList[25][1],lmList[25][2]),(lmList[23][1],lmList[23][2]),(255,255,255),1)
                cv2.line(image,(lmList[23][1],lmList[23][2]),(lmList[11][1],lmList[11][2]),(255,255,255),1)
                cv2.line(image,(lmList[11][1],lmList[11][2]),(lmList[13][1],lmList[13][2]),(255,255,255),1)
                cv2.line(image,(lmList[13][1],lmList[13][2]),(lmList[15][1],lmList[15][2]),(255,255,255),1)
                cv2.line(image,(lmList[15][1],lmList[15][2]),(lmList[21][1],lmList[21][2]),(255,255,255),1)
                cv2.line(image,(lmList[15][1],lmList[15][2]),(lmList[17][1],lmList[17][2]),(255,255,255),1)
                cv2.line(image,(lmList[15][1],lmList[15][2]),(lmList[19][1],lmList[19][2]),(255,255,255),1)
                cv2.line(image,(lmList[17][1],lmList[17][2]),(lmList[19][1],lmList[19][2]),(255,255,255),1)
                cv2.line(image,(lmList[29][1],lmList[29][2]),(lmList[31][1],lmList[31][2]),(255,255,255),1)
                cv2.line(image,(lmList[31][1],lmList[31][2]),(lmList[27][1],lmList[27][2]),(255,255,255),1) 
        
#
                left_hip_half = (lmList[11][1]-lmList[23][1])/8
                left_hip_half = left_hip_half + lmList[23][1]
    
                left_hip_calc1 = (lmList[11][2]-lmList[23][2])/8
                left_hip_calc1 = left_hip_calc1 + lmList[23][2]
    
                left_hip_calc2 = (lmList[25][1]-lmList[23][1])/8
                left_hip_calc2 = left_hip_calc2 + lmList[23][1]
    
                left_hip_calc3 = (lmList[25][2]-lmList[23][2])/8
                left_hip_calc3 = left_hip_calc3 + lmList[23][2]

                if left_hip_angle < 134 or left_hip_angle > 142:

                    cv2.line(image, (lmList[23][1],lmList[23][2]), (int(left_hip_half), int(left_hip_calc1)) , (0, 255, 0), 3)
                    cv2.line(image, (lmList[23][1],lmList[23][2]), (int(left_hip_calc2), int(left_hip_calc3)) , (0, 255, 0), 3)
                    
                else:
                    
                    cv2.line(image, (lmList[23][1],lmList[23][2]), (int(left_hip_half), int(left_hip_calc1)) , (255, 0, 0), 3)
                    cv2.line(image, (lmList[23][1],lmList[23][2]), (int(left_hip_calc2), int(left_hip_calc3)) , (255, 0, 0), 3)
#

#
                left_knee_half = (lmList[23][1]-lmList[25][1])/8
                left_knee_half = left_knee_half + lmList[25][1]

                left_knee_calc1 = (lmList[23][2]-lmList[25][2])/8
                left_knee_calc1 = left_knee_calc1 + lmList[25][2]

                left_knee_calc2 = (lmList[27][1]-lmList[25][1])/8
                left_knee_calc2 = left_knee_calc2 + lmList[25][1]

                left_knee_calc3 = (lmList[27][2]-lmList[25][2])/8
                left_knee_calc3 = left_knee_calc3 + lmList[25][2]
                
                if left_knee_angle < 153 or left_knee_angle > 160:

                    cv2.line(image, (lmList[25][1],lmList[25][2]), (int(left_knee_half), int(left_knee_calc1)) , (0, 255, 0), 3)
                    cv2.line(image, (lmList[25][1],lmList[25][2]), (int(left_knee_calc2), int(left_knee_calc3)) , (0, 255, 0), 3)
                    
                else:
                    
                    cv2.line(image, (lmList[25][1],lmList[25][2]), (int(left_knee_half), int(left_knee_calc1)) , (255, 0, 0), 3)
                    cv2.line(image, (lmList[25][1],lmList[25][2]), (int(left_knee_calc2), int(left_knee_calc3)) , (255, 0, 0), 3)
#

#
                left_sholder_half = (lmList[13][1]-lmList[11][1])/8
                left_sholder_half = left_sholder_half + lmList[11][1]

                left_sholder_calc1 = (lmList[13][2]-lmList[11][2])/8
                left_sholder_calc1 = left_sholder_calc1 + lmList[11][2]

                left_sholder_calc2 = (lmList[23][1]-lmList[11][1])/8
                left_sholder_calc2 = left_sholder_calc2 + lmList[11][1]

                left_sholder_calc3 = (lmList[23][2]-lmList[11][2])/8
                left_sholder_calc3 = left_sholder_calc3 + lmList[11][2]

                if left_shoulder_angle < 93 or left_shoulder_angle > 105:

                    cv2.line(image, (lmList[11][1],lmList[11][2]), (int(left_sholder_half), int(left_sholder_calc1)) , (0, 255, 0), 3)
                    cv2.line(image, (lmList[11][1],lmList[11][2]), (int(left_sholder_calc2), int(left_sholder_calc3)) , (0, 255, 0), 3)  
                    
                else:
                    
                    cv2.line(image, (lmList[11][1],lmList[11][2]), (int(left_sholder_half), int(left_sholder_calc1)) , (255, 0, 0), 3)
                    cv2.line(image, (lmList[11][1],lmList[11][2]), (int(left_sholder_calc2), int(left_sholder_calc3)) , (255, 0, 0), 3) 
#

#
                left_elbow_half = (lmList[11][1]-lmList[13][1])/8
                left_elbow_half = left_elbow_half + lmList[13][1]

                left_elbow_calc1 = (lmList[11][2]-lmList[13][2])/8
                left_elbow_calc1 = left_elbow_calc1 + lmList[13][2]

                left_elbow_calc2 = (lmList[15][1]-lmList[13][1])/8
                left_elbow_calc2 = left_elbow_calc2 + lmList[13][1]

                left_elbow_calc3 = (lmList[15][2]-lmList[13][2])/8
                left_elbow_calc3 = left_elbow_calc3 + lmList[13][2]
                
                if left_elbow_angle < 68 or left_elbow_angle > 75:

                    cv2.line(image, (lmList[13][1],lmList[13][2]), (int(left_elbow_half), int(left_elbow_calc1)) , (0, 255, 0), 3)
                    cv2.line(image, (lmList[13][1],lmList[13][2]), (int(left_elbow_calc2), int(left_elbow_calc3)) , (0, 255, 0), 3)  
                
                else:
                    
                    cv2.line(image, (lmList[13][1],lmList[13][2]), (int(left_elbow_half), int(left_elbow_calc1)) , (255, 0, 0), 3)
                    cv2.line(image, (lmList[13][1],lmList[13][2]), (int(left_elbow_calc2), int(left_elbow_calc3)) , (255, 0, 0), 3) 





   
    
    # Draw the pose annotation on the image.
    # image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # mp_drawing.draw_landmarks(
    #     image,
    #     results.pose_landmarks,
    #     mp_pose.POSE_CONNECTIONS,
    #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()