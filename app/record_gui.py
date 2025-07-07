import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import numpy as np
import threading
import os
from PIL import Image, ImageTk, ImageOps

# MediaPipe pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
landmarks_to_keep = set([
    11, 12, 13, 14, 15, 16,
    23, 24, 25, 26, 27, 28,
    29, 30, 31, 32
])

# Global state
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
        filename = os.path.join(SAVE_DIR, f"{label}.npy")
        np.save(filename, np.array(collected_data))
        status_label.config(text=f"Saved {len(collected_data)} samples to {filename}")
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

    # Convert frame to PIL Image and resize smoothly
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    if canvas_width > 0 and canvas_height > 0:
        img = ImageOps.contain(img, (canvas_width, canvas_height), method=Image.Resampling.LANCZOS)

    imgtk = ImageTk.PhotoImage(image=img)
    canvas.imgtk = imgtk
    canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    root.after(10, update_frame)

def start_camera():
    global cap
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for better compatibility on Windows

    # Force HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    update_frame()

# ========== UI ==========
root = tk.Tk()
root.title("TacticalPose — Gesture Recorder (HD)")

# Input label
label_frame = tk.Frame(root)
label_frame.pack(pady=5)
tk.Label(label_frame, text="Gesture Label:", font=("Arial", 12)).pack(side=tk.LEFT)
label_entry = tk.Entry(label_frame, font=("Arial", 12), width=20)
label_entry.insert(0, "salute")
label_entry.pack(side=tk.LEFT)

# Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=5)
tk.Button(button_frame, text="Start Recording", bg="green", fg="white", command=start_recording).pack(side=tk.LEFT, padx=10)
tk.Button(button_frame, text="Stop & Save", bg="red", fg="white", command=stop_recording).pack(side=tk.LEFT, padx=10)

# Camera feed
canvas = tk.Canvas(root, bg="black")
canvas.pack(fill="both", expand=True)

# Status
status_label = tk.Label(root, text="Enter a gesture and press start", font=("Arial", 10))
status_label.pack(pady=4)

# Launch camera in a thread
threading.Thread(target=start_camera, daemon=True).start()

# Run app
root.geometry("900x700")
root.mainloop()

# Cleanup
if cap:
    cap.release()
cv2.destroyAllWindows()
