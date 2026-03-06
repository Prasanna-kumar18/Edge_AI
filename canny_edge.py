import cv2

image=cv2.imread("images.jpg")

#gray scale image to find the edges
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

edge_image=cv2.Canny(gray_image,150,255)

#Applied Gaussian blur to gray scale and finding edges

Gauss_image=cv2.GaussianBlur(gray_image,(5,5),0)

edge_blur_image=cv2.Canny(Gauss_image,150,255)

cv2.imshow("Original Image",image)
cv2.imshow("Gray Image",gray_image)
cv2.imshow("Edge Image",edge_image)
cv2.imshow("Gaussian Edge Image",edge_blur_image)

cv2.waitKey(0)
cv2.destroyAllWindows()