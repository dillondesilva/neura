import cv2
import csv
import os
import mediapipe as mp

class DataCreator():
  def __init__(self):

    self._capture_features = ["Gesture", "WRIST_X", "WRIST_Y",
    "THUMB_CMC_X", "THUMB_CMC_Y", "THUMB_MCP_X", "THUMB_MCP_Y",
    "THUMB_IP_X", "THUMB_IP_Y", "THUMB_TIP_X","THUMB_TIP_Y", "INDEX_FINGER_MCP_X", 
    "INDEX_FINGER_MCP_Y", "INDEX_FINGER_PIP_X", "INDEX_FINGER_PIP_Y", "INDEX_FINGER_DIP_X", 
    "INDEX_FINGER_DIP_Y", "INDEX_FINGER_TIP_X", "INDEX_FINGER_TIP_Y", "MIDDLE_FINGER_MCP_X", 
    "MIDDLE_FINGER_MCP_Y", "MIDDLE_FINGER_PIP_X", "MIDDLE_FINGER_PIP_Y", "MIDDLE_FINGER_DIP_X", 
    "MIDDLE_FINGER_DIP_Y", "MIDDLE_FINGER_TIP_X","MIDDLE_FINGER_TIP_Y", "RING_FINGER_MCP_X", 
    "RING_FINGER_MCP_Y", "RING_FINGER_PIP_X", "RING_FINGER_PIP_Y", "RING_FINGER_DIP_X", 
    "RING_FINGER_DIP_Y", "RING_FINGER_TIP_X",  "RING_FINGER_TIP_Y", "PINKY_MCP_X", "PINKY_MCP_Y",
    "PINKY_PIP_X", "PINKY_PIP_Y", "PINKY_DIP_X", "PINKY_DIP_Y",
    "PINKY_TIP_X", "PINKY_TIP_Y"]

    self._captured_data = []
    self._capture_gestures = ["Palm_Open", "Grip", "Index_Point"]
    self._mp_hands = mp.solutions.hands.Hands(
                    model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    max_num_hands = 1
                  )

    self._num_hand_landmarks = 21

  def take_live_snapshot(self, gesture_name, success, image):
    if success:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = self._mp_hands.process(image)
      if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        gesture_points_data = [gesture_name]
        for idx in range(0, self._num_hand_landmarks):
          x = hand_landmarks.landmark[idx].x
          y = hand_landmarks.landmark[idx].y
          z = hand_landmarks.landmark[idx].z

          gesture_points_data.append(x)
          gesture_points_data.append(y)

        self._captured_data.append(gesture_points_data)

  def take_static_snapshot(self, gesture_name, file_path):
    image = cv2.flip(cv2.imread(file_path), 1)
    results = self._mp_hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
      hand_landmarks = results.multi_hand_landmarks[0]
      gesture_points_data = [gesture_name]
      for idx in range(0, self._num_hand_landmarks):
        x = hand_landmarks.landmark[idx].x
        y = hand_landmarks.landmark[idx].y
        z = hand_landmarks.landmark[idx].z

        gesture_points_data.append(x)
        gesture_points_data.append(y)

      self._captured_data.append(gesture_points_data)

  def capture_via_cam(self, output_file):
    cap = cv2.VideoCapture(0)
    
    for gesture in self._capture_gestures:
      user_ready = input(f"Press enter to record gestures for {gesture}...")
      capture_count = 0

      while capture_count <= 200:
        success, image = cap.read()
        self.take_live_snapshot(gesture, success, image)
        capture_count += 1

    with open(f"TestData/{output_file}.csv", "w", encoding="utf-8") as training_data_file:
      writer = csv.writer(training_data_file)
      writer.writerow(self._capture_features)
      writer.writerows(self._captured_data)

    cap.release()

  def capture_via_static(self, output_file, input_dir):
    accumulated_capture_data = []
    
    for gesture in self._capture_gestures:
      for image_file in os.listdir(f"{input_dir}/{gesture}"):
        filename = os.fsdecode(image_file)
        path = f"{input_dir}/{gesture}/{filename}"
        self.take_static_snapshot(gesture, path)
    
    with open(f"TestData/{output_file}.csv", "w") as training_data_file:
      writer = csv.writer(training_data_file)
      writer.writerow(self._capture_features)
      writer.writerows(self._captured_data)