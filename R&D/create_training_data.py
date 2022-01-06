import cv2
import csv
import mediapipe as mp

mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)

CSV_HEADER = ["Gesture", "WRIST_X", "WRIST_Y", "WRIST_Z",
  "THUMB_CMC_X", "THUMB_CMC_Y", "THUMB_CMC_Z", "THUMB_MCP_X", "THUMB_MCP_Y",
  "THUMB_MCP_Z", "THUMB_IP_X", "THUMB_IP_Y", "THUMB_IP_Z", "THUMB_TIP_X",
  "THUMB_TIP_Y", "THUMB_TIP_Z", "INDEX_FINGER_MCP_X", "INDEX_FINGER_MCP_Y",
  "INDEX_FINGER_MCP_Z", "INDEX_FINGER_PIP_X", "INDEX_FINGER_PIP_Y", "INDEX_FINGER_PIP_Z",
  "INDEX_FINGER_DIP_X", "INDEX_FINGER_DIP_Y", "INDEX_FINGER_DIP_Z", "INDEX_FINGER_TIP_X",
  "INDEX_FINGER_TIP_Y", "INDEX_FINGER_TIP_Z", "MIDDLE_FINGER_MCP_X", "MIDDLE_FINGER_MCP_Y",
  "MIDDLE_FINGER_MCP_Z", "MIDDLE_FINGER_PIP_X", "MIDDLE_FINGER_PIP_Y", "MIDDLE_FINGER_PIP_Z",
  "MIDDLE_FINGER_DIP_X", "MIDDLE_FINGER_DIP_Y", "MIDDLE_FINGER_DIP_Z", "MIDDLE_FINGER_TIP_X",
  "MIDDLE_FINGER_TIP_Y", "MIDDLE_FINGER_TIP_Z", "RING_FINGER_MCP_X", "RING_FINGER_MCP_Y",
  "RING_FINGER_MCP_Z", "RING_FINGER_PIP_X", "RING_FINGER_PIP_Y", "RING_FINGER_PIP_Z",
  "RING_FINGER_DIP_X", "RING_FINGER_DIP_Y", "RING_FINGER_DIP_Z", "RING_FINGER_TIP_X",
  "RING_FINGER_TIP_Y", "RING_FINGER_TIP_Z", "PINKY_MCP_X", "PINKY_MCP_Y", "PINKY_MCP_Z",
  "PINKY_PIP_X", "PINKY_PIP_Y", "PINKY_PIP_Z", "PINKY_DIP_X", "PINKY_DIP_Y", "PINKY_DIP_Z",
  "PINKY_TIP_X", "PINKY_TIP_Y", "PINKY_TIP_Z"]
  
CSV_DATA = []

gestures = ["Palm_Open", "Grip"]

hands = mp_hands.Hands(
  model_complexity=0,
  min_detection_confidence=0.5,
  min_tracking_confidence=0.5
)

# Captures numerical data from mediapipe and loads to respective csv
def capture_data(gesture_name):
  success, image = cap.read()

  if success:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        csv_row = [gesture_name]
        for idx in range(0, 20):
          x = hand_landmarks.landmark[idx].x
          y = hand_landmarks.landmark[idx].y
          z = hand_landmarks.landmark[idx].z

          csv_row.append(x)
          csv_row.append(y)
          csv_row.append(z)
          print(x, y, z)

          CSV_DATA.append(csv_row)

for gesture in gestures:
  user_ready = input("Press enter to record gestures for {}...".format(gesture))
  capture_count = 0

  while capture_count <= 200:
    capture_data(gesture)
    capture_count += 1

with open("TestData/gesture_train.csv", "w") as f:
  writer = csv.writer(f)

  writer.writerow(CSV_HEADER)
  writer.writerows(CSV_DATA)

cap.release()