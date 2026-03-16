from ultralytics import YOLO
import cv2

model=YOLO("yolo26n.pt")

cap=cv2.VideoCapture("cars.mp4")

assert cap.isOpened(), "Error Reading the video"


w, h, fps=(int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer=cv2.VideoWriter("Tracked Vehicle.mp4",cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))


while cap.isOpened():

	success, frame=cap.read()

	if not success:
		print("Video is not opened/done")
		break

	tracking=model.track(frame,persist=True,tracker="bytetrack.yaml")	#ByteTrack(Fast & Accurate), Bot-SORT track(Default)

	annotate_frame=tracking[0].plot()

	video_writer.write(annotate_frame)


cap.release()
video_writer.release()
cv2.destroyAllWindows()



