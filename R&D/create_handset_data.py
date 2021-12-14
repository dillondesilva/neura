# This file will create a training dataset for gesture recognition
import cv2 as cv
import os

# Note: Change the variable below to the path you want for labels to be stored
PATH_FOR_CREATING_SET = "~/projects/neura/R&D/"

# create camera object and ensure it works
camera = cv.VideoCapture(0)

if not camera.isOpened():
    print("The Camera is not Opened....Exiting")
    exit()

gesture_labels = ["Grip", "C_Shape", "Closed", "Flat", "Point"]
 
# Create dir for each of our gesture labels
for label in gesture_labels:
    if not os.path.exists(os.path.abspath(PATH_FOR_CREATING_SET + label + "/")):
        os.mkdir(label)

# Take 200 photos of each gesture
for label in gesture_labels:
    img_count = 0
    start = input("Press any key to start capturing for " + label + "...")
    
    if start:
        while img_count <= 200:
            status, frame = camera.read()

            if status:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                cv.imwrite(PATH_FOR_CREATING_SET + label + '/img'+ str(img_count)+'.png', gray)
                img_count = img_count + 1

            if cv.waitKey(1) == ord('q'):
                break

camera.release()
cv.destroyAllWindows()