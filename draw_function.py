import cv2

image=cv2.imread("image1.jpg")

image=cv2.line(image,(293,283),(613,300),(0,255,0),4)

image=cv2.circle(image,center=(613,300),radius=30,color=(255,0,0),thickness=4)
image=cv2.circle(image,center=(293,283),radius=30,color=(255,0,0),thickness=4)

image=cv2.rectangle(image,(495,140),(580,190),(0,255,255),4)

cv2.imshow("Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

