import mediapipe as mp
import cv2
import numpy as np
import time
from playsound import playsound

capture = cv2.VideoCapture(0) # capturing video
cpos = 0 # Position of user
t0= 0 # start time of run
t1=0 # end time of run
usersum = 0 # capturing all user position and storing its sum of 33 points
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

def calc_sum(landmarkList): # sum of all 33 landmark positions
    tsum = 0
    for i in range(11,33):
        tsum += landmarkList[i].x * 480

    return tsum

def calc_dist(landmarkList):
    return (landmarkList[28].y*640 - landmarkList[24].y*640)

def isVisible(landmarkList):
    # 0.7 or 70% confidence in detection visibility of 24th and 28th landmark
    if(landmarkList[28].visibility > 0.6) and (landmarkList[24].visibility > 0.6): 
        return True
    return False

mp_pose = mp.solutions.pose  # initializing mediapipe pose 
 # drawing landmark on screen to show visuals on screen
pose = mp_pose.Pose()
# Initializing the drawing utils for drawing the facial landmarks on image
drawing = mp.solutions.drawing_utils 

image_1=cv2.imread('image_1.png') # For green light
image_2=cv2.imread('image_2.png') # For red light

currWindow = image_1

while True:
    _, frm = capture.read() # capturing the frame from video input
    # frm is the current image whose color is to be changed from BGR to RBG as cv2 reads images in BGR format
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    # now that format is RBG ml pipeline can read frm 
    #here we are passing image to pose detection ml pipeline
    res = pose.process(rgb)
    # cv2.blur to remove noise from krenel with 5x5 size
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
            playsound('greenLight.mp3')
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
                print("WINNER")
                winner = 1

            else:
                if not(isCinit):
                    isCinit = True
                    cstart = time.time()
                    cEnd = cstart
                    currWindow = image_2
                    # to play sounds
                    playsound('redLight.mp3')
                    usersum = calc_sum(res.pose_landmarks.landmark)
                if (cEnd - cstart) <=3:
                    tempsum = calc_sum(res.pose_landmarks.landmark)
                    cEnd = time.time()
                    if abs(tempsum - usersum) >150:
                        print("DEAD" , abs(tempsum - usersum))
                        isAlive = 0 # if true,user is dead

                else:
                    isInit = False # so again these values can be updated after one stop
                    isCinit = False
        
        #cv2.circle(image, center_coordinates, radius, color, thickness)                        
        cv2.circle(currWindow, ((55+6*cpos),280),15, (0,0,255), -1)
        
        #cv2.resize(source,fx, fy)
        mainWin = np.concatenate(cv2.resize((frm, (800,400)), currWindow), axis = 0)
        cv2.imshow("MAIN WINDOW", mainWin)
                    
    else:
        #cv2.putText() method is used to draw a text string on any image.
        #cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
        #(20,200) - coordinates of bottom left corner of screen
        cv2.putText(frm, "PEASE MAKE SURE YOU ARE IN FRAME", (20,200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 4)
        cv2.imshow("WINDOW",frm)
        #cv2.waitkey will wait for 1ms only before taking another capture.read() from webcam feed
    if cv2.waitKey(1) == 27 or isAlive == 0 or winner == 1:  # IF any of the following happens program ends like if you press esc key
        #Python Opencv destroyAllWindows() function allows users to destroy or close all windows at any time after exiting the script
        cv2.destroyAllWindows()
        #capture.release releases both s/w and h/w resources consumed by program
        capture.release()
        break

frm = cv2.blur(frm, (5,5))

if isAlive == 0:
    cv2.putText(frm, "YOU ARE DEAD", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
    cv2.imshow("MAIN WINDOW ", frm)

if winner == 1:
    cv2.putText(frm, "YOU ARE WINNER", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
    cv2.imshow("MAIN WINDOW ", frm)

cv2.waitKey(0)