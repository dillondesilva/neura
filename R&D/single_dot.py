import cv2 as cv

cap = cv.VideoCapture(0);

hand_cascade = cv.CascadeClassifier()

if not hand_cascade.load(cv.samples.findFile("./TestData/palm_v4.xml")):
    print('--(!)Error loading hand cascade')
    exit(0)

def detectHand(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Detect hands
    hands = hand_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in hands:
        center = (x + w//2, y + h//2)
        cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]

    cv.imshow('Capture - Hand detection', frame)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detectHand(frame)
    # Display the resulting frame
    if cv.waitKey(1) == ord('q'):
        break