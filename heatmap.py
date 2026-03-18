import cv2
import numpy as np
import time
from ultralytics import YOLO

model = YOLO("yolo26n.pt")  #YOLO Model

video_path = "traffic_2.mp4"  #input Video
cap = cv2.VideoCapture(video_path)  #Capture input Frame

if not cap.isOpened():
    print("Error opening video file")
    exit()

width, height = 1080, 640

#Output_Video_Configuration
fps_input = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_path = "output.mp4"
out=cv2.VideoWriter(output_path,fourcc,fps_input,(width,height))

#Heatmap Intensity
global_img_array = np.zeros((height,width),dtype=np.float32)

previous_time = 0

vehicle_ids = [2,3,5,7] # 2 -car, 3- motorcycle, 5- bus, 7- truck

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break

    frame=cv2.resize(frame,(width,height))

    results=model(frame,conf=0.35)[0]

    for box in results.boxes:
        cls_id=int(box.cls[0])

        if cls_id not in vehicle_ids:
            continue

        conf=float(box.conf[0])

        x1,y1,x2,y2=map(int,box.xyxy[0])

        #Bounding Box
        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(width,x2)
        y2 = min(height,y2)

        #Draw the detection
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)
        label = f"{model.names[cls_id]}: {conf:.2f}"
        cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2)

        #center point Intensity method
        cx = (x1+x2)//2
        cy = (y1+y2)//2

        #add intensity to the center point
        global_img_array[cy, cx] += 50  

    #apply Gaussian blur to the heatmap
    blurred = cv2.GaussianBlur(global_img_array,(25,25),0)

    #normalize the heatmap (0-255)
    heatmap_norm = cv2.normalize(blurred,None,0,255,cv2.NORM_MINMAX)

    #convert the image to 8-bit
    heatmap_uint8 = heatmap_norm.astype(np.uint8)

    #apply color map
    heatmap_color = cv2.applyColorMap(heatmap_uint8,cv2.COLORMAP_JET)

    #overlay the heatmap on the original frame
    super_imposed_img = cv2.addWeighted(heatmap_color,0.6,frame,0.4,0)

    #Display the output
    cv2.imshow("Heatmap",super_imposed_img)
    out.write(super_imposed_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

