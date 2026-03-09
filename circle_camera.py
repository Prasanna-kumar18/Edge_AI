from picamera2 import Picamera2
import cv2
import numpy as np

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=100
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            x, y, r = c
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

    cv2.imshow("Circle Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
picam2.stop()