import cv2 as cv
import numpy as np
path = r'C:\Users\prabh\Downloads\haarcascade_frontalface_default.xml'
classifier = cv.CascadeClassifier(path)

cam = cv.VideoCapture(0)
while True:
    _,img = cam.read()
    img = cv.flip(img,1)
    faces = classifier.detectMultiScale(img,1.1,5)

    for f in faces:
        if f[-1] == max(faces[:,-1]):
            break
    
        x = f[0]
        y = f[1]
        w = f[2]
        h = f[3]

        circle_x = x + int(w/2)
        circle_y = y + int(h/2)

        cv.circle(img, (circle_x,circle_y),int(w/1.7),(110,180,68),1)

        cv.imshow('Frame', img)

        if cv.waitKey(1) == 27:
            cam.release()
            break

        