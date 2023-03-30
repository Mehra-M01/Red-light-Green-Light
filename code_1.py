import mediapipe as mp
import cv2
import numpy as np
import time
from playsound import playsound

capture = cv2.VideoCapture(0) # captureing video
cpos = 0 # Position of user
t0= 0 # start time of run
t1=0 # end time od run
usersum = 0 # captruring all user position and storing its sum of 32 points
duration = 0 # duration for which user is allowed to move
isAlive = 1 # pointer for user alive or not
isInit = False # initialization of start and end time
cstart, cEnd= 0,0 # time for which user is not allowed to move
isCinit = False #initialization of cstart and cend
tempsum = 0 # comparing to usersum to check for movement
winner = 0 # if cpos=100 then user won (crossed the line)
inFrame = 0 # user in frame or not
inFramecheck = False
threshold = 180 # minimum height between points 24 and 28 to check for movement

def calc_sum(landmarkList): # sum of all 32 landmark positions
    tsum = 0
    for i in range(11,33):
        tsum += landmarkList[i].x * 480

    return tsum

def calc_dist(landmarkList):
    return (landmarkList[28].y*640 - landmarkList[24].y*640)

def isVisible(landmarkList):
    if(landmarkList[28].visibility > 0.7) and (landmarkList[24].visibility > 0.7): # 0.7 or 70% confidence in visibility of 24th and 28th landmark
        return True
    return False

mp_pose = mp.solutions.pose  # drawing landmark on screen to show visuals on screen 
pose = mp_pose.Pose()
drawing = mp.solutions.drawing_utils

image_1=cv2.imread('im1.png') # For green light
image_2=cv2.imread('im2.png') # For red light

currWindow = image_1

while True:
    _, frm = capture.read() # capturing the frame from visual input
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    frm = cv2.blur(frm,(5,5))
    drawing.draw_landmarks(frm, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if not(inFramecheck):
        try:
            if isVisible(res.pose_landmarks.landmark):
                inFrame = 1
                inFramecheck = True
            else:
                inFrame = 0
        except:
            print("You are not visible at all")

            if inFrame == 1: # if user in frame
                if not(isInit):
                    playsound('greenlight.mp3')
                    currWindow = image_1
                    t0= time.time()
                    t1= t0
                    dur= np.random.randint(1,5) #setting random duration of time
                    isInit= True  # setting to true to play the sound only once

                if (t1-t0) <= dur:
                    try:
                        m = calc_dist(res.pose_landmarks.landmark) # calculating distance between 24th and 28th landmark
                        if m < threshold:
                            cpos += 1

                        print("current progress is : ", cpos)
                    except:
                        print("Not visible")

                        t1= time.time()

                else:

                    if cpos >= 100:
                        print("YOU HAVE CLEARED THE STAGE AND ARE ALIVE TILL NOW")
                        winner = 1

                    else:
                        if not(isCinit):
                            isCinit = True
                            cstart = time.time()
                            cEnd = cstart
                            currWindow = image_2
                            playsound('redlight.mp3')
                            

