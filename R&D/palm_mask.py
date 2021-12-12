import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0);

while True:
    ret, frame = cap.read();

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hsvim = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    skinRegionHSV = cv.inRange(hsvim, lower, upper)
    blurred = cv.blur(skinRegionHSV, (2,2))
    ret,thresh = cv.threshold(blurred,0,255,cv.THRESH_BINARY)
    cv.imshow("thresh", thresh)

    if cv.waitKey(1) == ord('q'):
        break