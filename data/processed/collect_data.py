import cv2
import mediapipe as mp
import numpy as np
import os

# =========================================================
GESTURE_LABELS = "salute"
SAMPELS_TO_COLLECT = 100
SAVE_DIR = "data/processed"
# =========================================================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp.drawing = mp.solutions.drawing_utils

#Indices to include (no face)
allowed_landmark_indices = set([
    11, 12, 13, 14, 15, 16,
    23, 24, 25, 26, 27, 28,
    31, 32 
])

#open camera
cap = cv2.VideoCapture(0)
collected = []

print("Collecting samples for gesture:", GESTURE_LABELS)

while cap.isOpened() and len(collected) < SAMPELS_TO_COLLECT:
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
                sample.extend([0, 0, 0, 0])  # Fill with zeros for excluded landmarks

        collected.append(sample)

        #draw with simple styled 
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
        )

        #progress text
        cv2.putText(frame, f"Collected: {len(collected)}/{SAMPELS_TO_COLLECT}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Collecting Samples", frame)
        if cv2.waitKey(1) & 0xFF == 27: # Exit on ESC key
            break
        
        cap.release()
        cv2.destroyAllWindows()

        #save collected data
        os.makedirs(SAVE_DIR, exist_ok=True)
        filename = os.path.join(SAVE_DIR, f"{GESTURE_LABELS}.npy")
        np.save(filename, np.array(collected)
        print(f"Saved {len(collected)} samples to {filename}"))
