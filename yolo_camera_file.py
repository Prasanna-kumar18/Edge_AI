import cv2
from ultralytics import YOLO
from IPython.display import Video

model = YOLO('yolov8n-cls.pt')

VIDEO_PATH = "/content/drive/MyDrive/Edge_AI/Cars.mp4"  # change if needed
OUTPUT_PATH = "/content/output_cls.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Error reading video file"

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25.0

# Output writer
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

    results = model(frame)

    probs = results[0].probs
    top1 = probs.top1
    class_name = model.names[top1]
    confidence = probs.top1conf.item()

    label = f"{class_name}: {confidence:.2f}"
    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
  
    out.write(frame)

cap.release()
out.release()

print("Processing complete!")
