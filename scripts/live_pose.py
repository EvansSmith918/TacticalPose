import cv2
import mediapipe as mp

# ========== Setup ==========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Zoom config
zoom = 1.0          # default zoom level
zoom_step = 0.1
max_zoom = 2.0
min_zoom = 1.0

# Only draw these landmarks (no face)
allowed_landmark_indices = set([
    11, 12, 13, 14, 15, 16,         # shoulders & arms
    23, 24, 25, 26, 27, 28,         # hips & legs
    29, 30, 31, 32                  # feet
])
pose_connections = [
    conn for conn in mp_pose.POSE_CONNECTIONS
    if conn[0] in allowed_landmark_indices and conn[1] in allowed_landmark_indices
]

# Drawing style
landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
connection_style = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
# ============================


def apply_zoom(frame, zoom_factor):
    """Crop and scale frame to simulate zoom."""
    if zoom_factor == 1.0:
        return frame

    h, w = frame.shape[:2]
    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = apply_zoom(frame, zoom)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        # Remove face joints by zeroing them out
        filtered_landmarks = results.pose_landmarks
        for i, lm in enumerate(filtered_landmarks.landmark):
            if i not in allowed_landmark_indices:
                lm.x = lm.y = lm.z = lm.visibility = 0

        mp_drawing.draw_landmarks(
            frame,
            filtered_landmarks,
            pose_connections,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style
        )

    # Show output
    cv2.imshow("Pose Detection (Zoom + No Face)", frame)

    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('+') or key == ord('='):
        zoom = min(zoom + zoom_step, max_zoom)
    elif key == ord('-') or key == ord('_'):
        zoom = max(zoom - zoom_step, min_zoom)

cap.release()
cv2.destroyAllWindows()
