# Red-light Green-light
Red light Green light (Squid Game) using python  | machine learning project

This project runs in this following fashion, firstly it checks if player is in the frame or not, once player is in frame  audio for greenLight is played which allows player to move until some random time gap after which redLight is played. Now checks if player has made any movement or not, if yes then player declared Dead else game progresses until player gets more than 100 points or he is dead midway.
 
MediaPipe Pose

Human pose estimation from video plays a critical role in various applications such as quantifying physical exercises, sign language recognition,and full-body gesture control. We will use mediapipe pose for detecting motion of user in the frame.The solution utilizes a two-step detector-tracker ML pipeline. The tracker subsequently predicts the pose landmarks and segmentation mask within the ROI using the ROI-cropped frame as input. Note that for video use cases the detector is invoked only as needed, i.e., for the very first frame and when the tracker could no longer identify body pose presence in the previous frame. 

Model used-
Pose Landmark Model (BlazePose GHUM 3D)
The landmark model in MediaPipe Pose predicts the location of 33 pose landmarks (figure in landmarks.png).

POSE_LANDMARKS
A list of pose landmarks. Each landmark consists of the following:
x and y: Landmark coordinates normalized to [0.0, 1.0] by the image width and height respectively.
z: Represents the landmark depth with the depth at the midpoint of hips being the origin, and the smaller the value the closer the 
landmark is to the camera. The magnitude of z uses roughly the same scale as x.
visibility: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.

OPENCV

To capture a video, we need to create a VideoCapture object. VideoCapture have the device index or the name of a video file. Device 
index is just the number to specify which camera. If we pass 0 then it is for first camera, 1 for second camera so on. We capture the 
video frame by frame.

cv2.VideoCapture(0): Means first camera or webcam.
cv2.VideoCapture(1):  Means second camera or webcam.
cv2.VideoCapture(“file name.mp4”): Means video file
