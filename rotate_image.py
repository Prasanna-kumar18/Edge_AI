import cv2
import time

image=cv2.imread("image1.jpg")

rotate_image=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
#rotate_image=cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
#rotate_image=cv2.rotate(image,cv2.ROTATE_180)

cv2.imshow("Original Image",image)

cv2.imshow("Rotated Image",rotate_image)
time.sleep(2)

cv2.waitKey(0)
cv2.destroyAllWindows()