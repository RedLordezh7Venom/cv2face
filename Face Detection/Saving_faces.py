import os 
import cv2 as cv
import numpy as np

path = r'C:\Users\prabh\Downloads\haarcascade_frontalface_default.xml'
classifier = cv.CascadeClassifier(path)
img = cv.imread('img.png')
faces = classifier.detectMultiScale(img,1.1,5)

cam = cv.VideoCapture(0)


def save(frame, folder_name): 
    name_img = len(os.listdir(folder_name)) + 1
    name_img = folder_name + "/IMG_" + str(name_img)+'.png'#Dynamic Naming
    cv.imwrite(name_img, frame) #Saving
    print(name_img ,'is exported')

for (x,y,w,h) in faces:

    face = img[y:y+h, x:x+w]
    cv.imshow('Face'   , face)
    
    
    if cv.waitKey(0) == 13:         # Save the Image | 13  = Enter Key
        save(face, 'People')
    
    elif cv.waitKey(0) == 127:      # Skip the Image | 127 = Delete Key
        pass

