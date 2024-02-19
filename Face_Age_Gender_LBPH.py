#!/usr/bin/env python
# coding: utf-8

# # Age, Gender and Emotion Detection
# 
# ### Let's load our classfiers

# In[2]:


from pathlib import Path
import cv2
import dlib
import sys
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import faceRecognition as fr


face_classifier = cv2.CascadeClassifier(r'C:\Python37\Major\haarcascade_frontalface_alt.xml')
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'C:\Python37\Major\trainingData.yml')
name={0:"Haritha",1:"Mom",2:"dheeru",3:"sanju"} 

def face_detector(img):
    # Convert image to grayscale for faster detection
    ret,test_img=cap.read()
    faces_detected,gray_img=fr.faceDetection(test_img)
    print("face Detected: ",faces_detected)
    gray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return False ,(0,0,0,0), np.zeros((1,48,48,3), np.uint8), img
    
    allfaces = []   
    rects = []
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi = img[y:y+h, x:x+w]
        allfaces.append(roi)
        rects.append((x,w,y,h))
    return True, rects, allfaces, img

# Define our model parameters
depth = 16
k = 8
weight_file = None
margin = 0.4
image_dir = None

img_size = 64
model = WideResNet(img_size, depth=depth, k=k)()
model.load_weights(r'C:\Python37\Major\weights.28_age_gender.hdf5')

# Initialize Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    ret, rects, faces, test_img = face_detector(frame)
    preprocessed_faces_ag = []
    faces_detected,gray_img=fr.faceDetection(test_img)
    
    
    if ret:
        for (i,face) in enumerate(faces):
            face_ag = cv2.resize(face, (64, 64), interpolation = cv2.INTER_AREA)
            preprocessed_faces_ag.append(face_ag)

        for face in faces_detected:
            (x,y,w,h)=face
            roi_gray=gray_img[y:y+h,x:x+h]
            label,confidence=face_recognizer.predict(roi_gray)
            print ("Confidence :",confidence)
           # print("label :",label)
            #fr.draw_rect(test_img,face)
            predicted_name=name[label]
            
            if(confidence>47):
               continue
            print(predicted_name)
            #face_label=fr.put_text(test_img,predicted_name,x,y)
            

        # make a prediction for Age and Gender
        results = model.predict(np.array(preprocessed_faces_ag))
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        # draw results, for Age and Gender and Emotion
        for (i, face) in enumerate(faces):
            label = "{}, {},{}".format(int(predicted_ages[i]),
                                        "F" if predicted_genders[i][0] > 0.2 else "M",
                                        predicted_name if confidence< 47 else "Unknown")
            
        #Overlay our detected emotion on our pic
        for (i, face) in enumerate(faces):
            label_position = (rects[i][0] + int((rects[i][1]/2)), abs(rects[i][2] - 10))
            cv2.putText(test_img, label, label_position , cv2.FONT_HERSHEY_PLAIN,1, (0,255,0), 2)

    cv2.imshow("Detedtion", test_img)
    
    
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()      


