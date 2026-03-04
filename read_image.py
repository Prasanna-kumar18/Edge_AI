import cv2

print(cv2.__version__)

image=cv2.imread("image.jpeg")

print(image)

cv2.imshow("Aston martin",image)

cv2.waitKey(0)