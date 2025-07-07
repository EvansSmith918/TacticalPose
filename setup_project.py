import os

# Project folders
folders = [
    "scripts",
    "models",
    "data/raw",
    "data/processed",
    "app",
    "results",
]

# Basic files
files = {
    "README.md": "# TacticalPose\n\nReal-time human gesture recognition project.",
    ".gitignore": """__pycache__/
*.pyc
.env/
.venv/
venv/
*.log
.vscode/
*.DS_Store
""",
    "requirements.txt": """opencv-python==4.9.0.80
mediapipe==0.10.9
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
matplotlib==3.8.4
streamlit==1.35.0
""",
    "scripts/live_pose.py": """# Live pose detection script
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
""",
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for path, content in files.items():
    with open(path, "w") as f:
        f.write(content)

print("✅ TacticalPose project structure created.")
