from ultralytics import YOLO
from picamera2 import Picamera2
import cv2

# Load YOLO model (nano model recommended for Pi)
model = YOLO("yolo26n.pt")

# Initialize Picamera2
picam2 = Picamera2()

camera_config = picam2.create_preview_configuration(
    main={"size": (640, 480),"format": "RGB888"}
)

picam2.configure(camera_config)
picam2.start()

print("Camera started... Press Q to quit")

while True:

    # Capture frame from Pi camera
    frame = picam2.capture_array()

    # Run YOLO detection
    results = model(frame, imgsz=320, conf=0.4, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot()

    # Display frame
    cv2.imshow("YOLO Edge AI - Raspberry Pi 5", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
