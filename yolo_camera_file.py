import cv2
from ultralytics import YOLO
from IPython.display import Video

# Load detection model (IMPORTANT CHANGE)
model = YOLO('yolov8n.pt')

VIDEO_PATH = "/content/drive/MyDrive/Edge_AI/Cars.mp4"
OUTPUT_PATH = "/content/output_video.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Error reading video file"

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25.0

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run detection
    results = model(frame)

    if results[0].boxes is not None:
        boxes = results[0].boxes

        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(xyxy, cls_ids, confs):
            x1, y1, x2, y2 = map(int, box)

            class_name = model.names[cls_id]
            label = f"{class_name} {conf:.2f}"

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    out.write(frame)

cap.release()
out.release()

print("Done!")
