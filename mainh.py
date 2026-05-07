import cv2
import os
import torch
from ultralytics import YOLO

# ================= CONFIGURATION =================
video_path = "b1.mp4"
model_path = "best_4.pt"

frame_skip = 3
resize_width = 640
resize_height = 360

# ================= CHECK FILES =================
if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found.")
    exit()

if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    exit()

# ================= LOAD VIDEO =================
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

print("Video opened successfully!")

# ================= DEVICE =================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ================= LOAD MODEL =================
model = YOLO(model_path)

# 🔥 IMPORTANT: Get class names directly from model
class_names = model.names
print("Model Classes:", class_names)

# ================= CLASS COLORS =================
class_colors = {
    "cycle": (255, 255, 0),
    "four-wheeler": (255, 0, 255),
    "helmet": (0, 255, 0),
    "motor-bike": (255, 0, 0),
    "no helmet": (0, 0, 255),
    "Pedestral": (0, 255, 255),
    "tripple-ridding": (0, 165, 255)
}

# ================= PROCESS VIDEO =================
frame_id = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video.")
        break

    frame_id += 1

    # Skip frames
    if frame_id % frame_skip != 0:
        continue

    # Resize
    frame = cv2.resize(frame, (resize_width, resize_height))

    # ================= YOLO PREDICTION =================
    results = model.predict(frame, device=device, conf=0.4)

    # ================= PROCESS RESULTS =================
    for result in results:

        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:

            # Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Confidence
            conf = float(box.conf[0])

            # Class ID
            cls_id = int(box.cls[0])

            # ✅ Correct class name
            label = class_names[cls_id]

            # Color
            color = class_colors.get(label, (255, 255, 255))

            # ================= DRAW BOX =================
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label text with confidence
            text = f"{label} {conf:.2f}"

            # Text size
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Background
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 10, y1), color, -1)

            # Text
            cv2.putText(
                frame,
                text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    # ================= SHOW OUTPUT =================
    cv2.imshow("Traffic Violation Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ================= RELEASE =================
cap.release()
cv2.destroyAllWindows()