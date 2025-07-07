import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import numpy as np
import threading
import os
from PIL import Image, ImageTk

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
landmarks_to_keep = set([
    11, 12, 13, 14, 15, 16,
    23, 24, 25, 26, 27, 28,
    29, 30, 31, 32
])

# Globals
collecting = False
collected_data = []
cap = None

SAVE_DIR = "data/processed"

# Make sure save folder exists
os.makedirs(SAVE_DIR, exist_ok=True)

def start_recording():
    global collecting, collected_data
    label = label_entry.get().strip().lower()
    if not label:
        messagebox.showerror("Error", "Please enter a gesture label.")
        return
    collected_data = []
    collecting = True
    status_label.config(text=f"Recording '{label}'...")

def stop_recording():
    global collecting
    collecting = False
    label = label_entry.get().strip().lower()
    if collected_data:
        npy_path = os.path.join(SAVE_DIR, f"{label}.npy")
        np.save(npy_path, np.array(collected_data))
        status_label.config(text=f"Saved {len(collected_data)} samples to {npy_path}")
    else:
        status_label.config(text="No data recorded.")

def update_frame():
    global cap, collecting, collected_data
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        sample = []
        for i in range(33):
            if i in landmarks_to_keep:
                lm = results.pose_landmarks.landmark[i]
                sample.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                sample.extend([0, 0, 0, 0])
        if collecting:
            collected_data.append(sample)

        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Convert frame to PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.imgtk = imgtk
    canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    # Refresh every 10ms
    root.after(10, update_frame)

def start_camera():
    global cap
    cap = cv2.VideoCapture(0)
    update_frame()

# ========== UI Setup ==========
root = tk.Tk()
root.title("TacticalPose - Gesture Data Recorder")

label_entry = tk.Entry(root, font=("Arial", 14))
label_entry.pack(pady=5)
label_entry.insert(0, "salute")

start_button = tk.Button(root, text="Start Recording", command=start_recording, bg="green", fg="white")
start_button.pack(pady=5)

stop_button = tk.Button(root, text="Stop & Save", command=stop_recording, bg="red", fg="white")
stop_button.pack(pady=5)

canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

status_label = tk.Label(root, text="Enter a gesture and press start", font=("Arial", 12))
status_label.pack(pady=5)

# Start camera in background thread
threading.Thread(target=start_camera, daemon=True).start()

# Run app
root.mainloop()

# Cleanup
if cap:
    cap.release()
cv2.destroyAllWindows()
