import cv2
import time

WIDTH=620
HEIGHT=480

image=cv2.imread("image1.jpg")

resize_img=cv2.resize(image,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)

cv2.imshow("Original Image",image)

cv2.imshow("Resized Image",resize_img)
time.sleep(3)
cv2.waitKey(0)
cv2.destroyAllWindows()