from sklearn.model_selection import train_test_split
from sklearn import metrics

import pandas as pd
import joblib

gestures_test = pd.read_csv("Testdata/gesture_test.csv")

X= gestures_test[["WRIST_X", "WRIST_Y",
  "THUMB_CMC_X", "THUMB_CMC_Y", "THUMB_MCP_X", "THUMB_MCP_Y",
  "THUMB_IP_X", "THUMB_IP_Y", "THUMB_TIP_X","THUMB_TIP_Y", "INDEX_FINGER_MCP_X", 
  "INDEX_FINGER_MCP_Y", "INDEX_FINGER_PIP_X", "INDEX_FINGER_PIP_Y", "INDEX_FINGER_DIP_X", 
  "INDEX_FINGER_DIP_Y", "INDEX_FINGER_TIP_X", "INDEX_FINGER_TIP_Y", "MIDDLE_FINGER_MCP_X", 
  "MIDDLE_FINGER_MCP_Y", "MIDDLE_FINGER_PIP_X", "MIDDLE_FINGER_PIP_Y", "MIDDLE_FINGER_DIP_X", 
  "MIDDLE_FINGER_DIP_Y", "MIDDLE_FINGER_TIP_X","MIDDLE_FINGER_TIP_Y", "RING_FINGER_MCP_X", 
  "RING_FINGER_MCP_Y", "RING_FINGER_PIP_X", "RING_FINGER_PIP_Y", "RING_FINGER_DIP_X", 
  "RING_FINGER_DIP_Y", "RING_FINGER_TIP_X",  "RING_FINGER_TIP_Y", "PINKY_MCP_X", "PINKY_MCP_Y",
  "PINKY_PIP_X", "PINKY_PIP_Y", "PINKY_DIP_X", "PINKY_DIP_Y",
  "PINKY_TIP_X", "PINKY_TIP_Y"]]

target = gestures_test["Gesture"]

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.1)

gesture_recognition_model = joblib.load(open("gesture-recogniser", "rb"))

y_pred = gesture_recognition_model.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))