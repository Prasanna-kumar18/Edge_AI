from ultralytics import YOLO
import cv2
import os

model=YOLO("yolo26n.pt")

image=input("Enter the path of image: ")

result=model(image,conf=0.6)

boxes=result[0].plot()

cv2.imshow("Detected Image",boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
