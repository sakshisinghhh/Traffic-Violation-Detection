import cv2

cap = cv2.VideoCapture("b5.mp4")

# Get FPS of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print("Video FPS:", fps)

# To extract 10 FPS → save frame every (fps / 10) frames
frame_step = int(fps / 10)

count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save 10 frames per second
    if frame_step > 0 and count % frame_step == 0:
        cv2.imwrite(f"C:/Users/DELL/Desktop/helmet/yolov8helmetdetection-main/images/b5_{saved}.jpg", frame)
        print("Saved frame:", saved)
        saved += 1

    count += 1

cap.release()
print("Done! Total frames saved:", saved)


