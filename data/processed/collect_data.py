import cv2
import mediapipe as mp
import numpy as np
import os

# ======== CONFIG ========
GESTURE_LABEL = "salute"           # Change this for each gesture
SAMPLES_TO_COLLECT = 100           # How many samples to record
SAVE_DIR = "data/processed"        # Folder for saved data
# =========================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Indices to include (no face)
allowed_landmark_indices = set([
    11, 12, 13, 14, 15, 16,         # Shoulders & arms
    23, 24, 25, 26, 27, 28,         # Hips & legs
    29, 30, 31, 32                  # Feet
])

# Open webcam
cap = cv2.VideoCapture(0)
collected = []

print(f"Collecting gesture: {GESTURE_LABEL}")

while cap.isOpened() and len(collected) < SAMPLES_TO_COLLECT:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        sample = []

        for i in range(33):
            if i in allowed_landmark_indices:
                lm = landmarks[i]
                sample.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                sample.extend([0, 0, 0, 0])  # Placeholder for ignored points

        collected.append(sample)

        # Optional: draw with simplified style
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # Progress text
    cv2.putText(frame, f"Samples: {len(collected)}/{SAMPLES_TO_COLLECT}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Collecting Gesture', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
        break

cap.release()
cv2.destroyAllWindows()

# Save as NumPy array
os.makedirs(SAVE_DIR, exist_ok=True)
filename = os.path.join(SAVE_DIR, f"{GESTURE_LABEL}.npy")
np.save(filename, np.array(collected))
print(f"Saved {len(collected)} samples to {filename}")
