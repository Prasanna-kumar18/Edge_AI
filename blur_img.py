import cv2
import time

image=cv2.imread("images.jpg")

blur_image=cv2.blur(image,(5,5))

Gauss_image=cv2.GaussianBlur(image,(5,5),0)

median_blur=cv2.medianBlur(image,5)

bilateral_blur=cv2.bilateralFilter(image,7,100,100)

cv2.imshow("Original Image",image)
cv2.imshow("Blurred Image",blur_image)
cv2.imshow("Gaussian Blur Image",Gauss_image)
cv2.imshow("Median Blur Image",median_blur)
cv2.imshow("Bilateral Blur Image",bilateral_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()