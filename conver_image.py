import cv2
import time

image=cv2.imread("image1.jpg")

rgb_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#gray_image=cv2.cvtColor(rgb_image,cv2.COLOR_RGB2GRAY)

hsv_image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

hsl_image=cv2.cvtColor(image,cv2.COLOR_BGR2HLS)

cv2.imshow("BGR image",image)

cv2.imshow("RGB Image",rgb_image)
time.sleep(3)

cv2.imshow("GRAY Image",gray_image)
time.sleep(3)

cv2.imshow("HSV Image",hsv_image)
time.sleep(3)

cv2.imshow("HSL Image",hsl_image)
time.sleep(3)
cv2.waitKey(0)
