import cv2
import mediapipe as mp
import joblib

class InteractiveRunner():
    def __init__(self, model_path):
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        self._mp_hands = mp.solutions.hands.Hands(model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self._model_path = model_path

    def extract_data(self, hand_landmarks):
        csv_row = [[]]
        for idx in range(0, 21):
            x = hand_landmarks.landmark[idx].x
            y = hand_landmarks.landmark[idx].y

            csv_row[0].append(x)
            csv_row[0].append(y)
            
        return csv_row

    def run(self):
        # For webcam input:
        cap = cv2.VideoCapture(0)
        with self._mp_hands as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                gesture_recognition_model = joblib.load(open(self._model_path, "rb"))

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        data = self.extract_data(hand_landmarks)
                        predictions = gesture_recognition_model.predict_proba(data)
                        gesture = gesture_recognition_model.predict(data)
                        print(f"{gesture}: {predictions}")
                    
                        self._mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            self._mp_drawing_styles.get_default_hand_landmarks_style(),
                            self._mp_drawing_styles.get_default_hand_connections_style())

                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                    