import os
print("Current working directory:", os.getcwd())
print("Files in this directory:", os.listdir())
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize YOLOv8 model (using a pretrained model, e.g., yolov8n.pt)
model = YOLO('yolov8n.pt')

# Initialize DeepSort tracker
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, embedder="mobilenet", half=True)

# Open video file
cap = cv2.VideoCapture('C:/Users/user/OneDrive/Desktop/github/python_work-1/Chatbot_FAQS/my_vid.mp4')

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    # Prepare DeepSort input: list of [ [x, y, w, h], confidence, class_id ]
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw results
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('YOLOv8 + DeepSORT Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
