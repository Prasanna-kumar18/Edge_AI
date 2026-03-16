import cv2
import math
from ultralytics import YOLO

VIDEO_PATH = "cars.mp4"
MODEL_PATH = "yolo26n.pt"   
OUTPUT_PATH = "speed_estimation.mp4"

CONFIDENCE = 0.35
LINE_Y = 300
METER_PER_PIXEL = 0.032   

VEHICLE_CLASSES = {2, 3, 5, 7}   


model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Error reading video file"

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25.0

video_writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)


prev_centers = {}         
prev_speeds = {}           
crossed_line = {}         
display_speed = {}         
prev_side = {}             


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or processing completed.")
        break

    cv2.line(frame, (0, LINE_Y), (w, LINE_Y), (255, 0, 0), 3)
    cv2.putText(
        frame,
        "Detection Line",
        (10, LINE_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    results = model.track(frame, persist=True, conf=CONFIDENCE, verbose=False)

    if results and results[0].boxes is not None:
        boxes = results[0].boxes

        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
        cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        if track_ids is not None:
            for box, cls_id, track_id in zip(xyxy, cls_ids, track_ids):
                if cls_id not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                class_name = model.names.get(cls_id, str(cls_id))

                current_side = cy < LINE_Y  

                if track_id in prev_centers:
                    px, py = prev_centers[track_id]

                    pixel_distance = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                    distance_m = pixel_distance * METER_PER_PIXEL
                    speed_mps = distance_m * fps
                    speed_kmph = speed_mps * 3.6

                    if track_id in prev_speeds:
                        speed_kmph = 0.7 * prev_speeds[track_id] + 0.3 * speed_kmph

                    prev_speeds[track_id] = speed_kmph

                if track_id in prev_side:
                    if prev_side[track_id] != current_side and not crossed_line.get(track_id, False):
                        crossed_line[track_id] = True
                        display_speed[track_id] = prev_speeds.get(track_id, 0.0)

                prev_side[track_id] = current_side
                prev_centers[track_id] = (cx, cy)

                if crossed_line.get(track_id, False):
                    label = f"ID {track_id} {class_name} {display_speed[track_id]:.1f} km/h"
                else:
                    label = f"ID {track_id} {class_name}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

    video_writer.write(frame)
    cv2.imshow("Vehicle Speed Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
