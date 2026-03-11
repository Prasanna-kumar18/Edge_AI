from picamera2 import Picamera2
import cv2
import numpy as np

picam2=Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")

while True:

    frame=picam2.capture_array()

    gray_image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Detecting Face
    faces=face_cascade.detectMultiScale(gray_image,scaleFactor=1.1,minNeighbors=5,minSize=(80,80))
    #it is going to return [x, y, w, h]

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),3)

    #Detecting Eye

    eyes=eye_cascade.detectMultiScale(gray_image,scaleFactor=1.1,minNeighbors=8,minSize=(20,20))
    #it is going to return [ex, ey, ew, eh]

    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


    cv2.imshow("Detected Face & eyes",frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
picam2.stop()