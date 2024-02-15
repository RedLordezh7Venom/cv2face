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
    
    if len(faces) >= 1:
        x = f[0]
        y = f[1]
        w = f[2]
        h = f[3]

        face = img[y:y+h,x:x+w]
        face = cv.blur(face,(32,32))
        img[y:y+h,x:x+w] = face
    cv.imshow("frame",img)
    cv.imshow('face',face)
    if cv.waitKey(1) == 27:
        cam.release()
        break
