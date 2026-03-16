import cv2
from ultralytics import solutions

cap=cv2.VideoCapture("cars.mp4")

assert cap.isOpened(), "Error reading the video"

w, h, fps=(int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS))

video_writer=cv2.VideoWriter("speed_estimation.mp4",cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))


line_points=[(0,200),(w,200)]

speed_esti=solutions.SpeedEstimator(model="yolo26n.pt",fps=fps,show=True,region=line_points,meter_per_pixel=0.2)

while cap.isOpened():
	success, img=cap.read()

	if not success:
		print("Video is Empty or  done")
		break

	result=speed_esti(img)
	video_writer.write(result.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()







