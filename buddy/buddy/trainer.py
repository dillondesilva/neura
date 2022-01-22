from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import pandas as pd
import joblib

# Trainer() is an abstracted tool that allows for the creation
# of an arsenal of effective ML Models
class Trainer():
    def __init__(self, training_data_file, output_dir, output_fname):
        self._training_data_file = training_data_file

        self._features = ["WRIST_X", "WRIST_Y",
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

        self._target_feature = ["Gesture"]

        self._gestures_df = pd.read_csv(training_data_path)
        self._target_feature = self._gestures_df["Gesture"]
        self._training_features = self._gestures_df[self._features]
        self._output = f"{output_dir}/{output_fname}.pkl"

    # Creates a model using a random forest classifier
    def create_rf_model():
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(self._training_features, self._target_feature)

        joblib.dump(rf_classifier, self._output)

    def create_knn_model():
        pass

    def create_neural_net_model():
        pass