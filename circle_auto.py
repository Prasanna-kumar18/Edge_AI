import cv2
import numpy as np

image=cv2.imread("image.jpg")

gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

gray_image = cv2.GaussianBlur(gray_image, (9,9), 2)

circles=cv2.HoughCircles(gray_image,cv2.HOUGH_GRADIENT,dp=1,minDist=100,param1=100,param2=50,minRadius=10,maxRadius=60)

if circles is not None:
    circles=np.uint16(np.around(circles))
    # x,y,z=circles[0][0]
    # cv2.circle(image,(x,y),z,(0,255,0),2)
    # cv2.circle(image,(x,y),2,(0,0,255),3)

    for circle in circles[0, :]:
        x, y, r = circle
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        cv2.circle(image, (x, y), 2, (0, 0, 255), 3)


    cv2.imshow("Detected Circle",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No circles detected")