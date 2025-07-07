import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import os

# Load model
MODEL_PATH = "models/gesture_classifier.pkl"
LABEL_MAP_PATH = "models/label_map.json"

clf = joblib.load(MODEL_PATH)
label_map = {}

if os.path.exists(LABEL_MAP_PATH):
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
        label_map = {int(k): v for k, v in label_map.items()}

# Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
drawing = mp.solutions.drawing_utils

# Which landmarks to keep
landmarks_to_keep = set([
    11, 12, 13, 14, 15, 16,
    23, 24, 25, 26, 27, 28,
    29, 30, 31, 32
])

def extract_keypoints(results):
    sample = []
    for i in range(33):
        if i in landmarks_to_keep:
            lm = results.pose_landmarks.landmark[i]
            sample.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            sample.extend([0, 0, 0, 0])
    return np.array(sample).reshape(1, -1)

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("[INFO] Starting live gesture classification...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        # Extract features and predict
        keypoints = extract_keypoints(results)
        prediction = clf.predict(keypoints)[0]
        prob = clf.predict_proba(keypoints).max()

        label = label_map.get(prediction, prediction)
        text = f"{label} ({prob:.2f})"

        # Draw on frame
        drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("TacticalPose - Live Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
