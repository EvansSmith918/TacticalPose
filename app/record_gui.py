import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import numpy as np
import threading
import os
from PIL import Image, ImageTk, ImageOps

# Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
landmarks_to_keep = {
    11, 12, 13, 14, 15, 16,
    23, 24, 25, 26, 27, 28,
    29, 30, 31, 32
}

# Globals
collecting = False
collected_data = []
cap = None
frame_count = 0
SAVE_DIR = "data/processed"

# Ensure save folder exists
os.makedirs(SAVE_DIR, exist_ok=True)

def start_recording():
    global collecting, collected_data
    label = label_entry.get().strip().lower()
    if not label:
        messagebox.showerror("Error", "Please enter a gesture label.")
        return
    collected_data = []
    collecting = True
    status_label.config(text=f"🎙️ Recording '{label}'...")

def stop_recording():
    global collecting
    collecting = False
    label = label_entry.get().strip().lower()
    if collected_data:
        filename = os.path.join(SAVE_DIR, f"{label}.npy")
        np.save(filename, np.array(collected_data))
        status_label.config(text=f"Saved {len(collected_data)} samples to {filename}")
    else:
        status_label.config(text="No data recorded.")

def update_frame():
    global cap, collecting, collected_data, frame_count

    ret, frame = cap.read()
    if not ret:
        status_label.config(text="Webcam read failed.")
        return

    frame = cv2.flip(frame, 1)

    # Run pose every 2 frames
    if frame_count % 2 == 0:
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

    frame_count += 1

    # Resize for smoother preview
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize((960, 540), Image.Resampling.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=img)

    canvas.imgtk = imgtk
    canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    root.after(10, update_frame)

def start_camera():
    global cap
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Optional FPS limit
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        messagebox.showerror("Webcam Error", "Could not access the webcam.")
        return

    update_frame()

# ===== UI Setup =====
root = tk.Tk()
root.title("TacticalPose — Smooth Gesture Recorder")

label_frame = tk.Frame(root)
label_frame.pack(pady=5)
tk.Label(label_frame, text="Gesture Label:", font=("Arial", 12)).pack(side=tk.LEFT)
label_entry = tk.Entry(label_frame, font=("Arial", 12), width=20)
label_entry.insert(0, "salute")
label_entry.pack(side=tk.LEFT)

button_frame = tk.Frame(root)
button_frame.pack(pady=5)
tk.Button(button_frame, text="Start Recording", bg="green", fg="white", command=start_recording).pack(side=tk.LEFT, padx=10)
tk.Button(button_frame, text="Stop & Save", bg="red", fg="white", command=stop_recording).pack(side=tk.LEFT, padx=10)

canvas = tk.Canvas(root, width=960, height=540, bg="black")
canvas.pack()

status_label = tk.Label(root, text="Enter a gesture and press Start", font=("Arial", 10))
status_label.pack(pady=4)

# Start camera thread
threading.Thread(target=start_camera, daemon=True).start()

# Run app
root.geometry("1000x680")
root.mainloop()

# Cleanup
if cap:
    cap.release()
cv2.destroyAllWindows()
