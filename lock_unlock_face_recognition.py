import cv2
import sys
import numpy as np
import ctypes
import os
import time

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def main_recognize():
    counter_correct = 0 # counter variable to count number of times loop runs
    counter_wrong = 0
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    assure_path_exists("trainer/")
    recognizer.read('trainer/trainer.yml')  #load training model
    cascadePath = "haarcascade_frontalface_default.xml"  #cascade path
    faceCascade = cv2.CascadeClassifier(cascadePath);  #load cascade
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL  # Set the font style
    cam = cv2.VideoCapture(0)
    tester = False
    ll = []
    count_TRUE = 0
    count_TOTAL = 0

    start_time = time.time()
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3,5)

        if count_TOTAL > 1000:
            #if  count_TRUE > 99:
            #    break
            #else:
            #    count_TOTAL = 0
            #    count_TRUE = 0
            #    start_time = time.time()
            break

        count_TOTAL += 1
        
        for(x,y,w,h) in faces:

           
            cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)       
            Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])   # Recognize the face belongs to which ID
            if(confidence>80):                 #confidence usually comes greater than 80 for strangers
                counter_wrong += 1
                print("Wrong")
                Id = "Unknown + {0:.2f}%".format(round(100 - confidence, 2)) 
                print(confidence)
                print("counter_wrong - " + str(counter_wrong))
                cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,0,255), -1)
                cv2.putText(im, str(Id), (x,y-40), font, 1, (0,0,0), 2)
            else:                              #confidence usually comes less than 80 for correct user(s)
                Id = "Dima + {0:.2f}%".format(round(100 - confidence, 2)) 
                print("Verified")
                print(confidence)
                ll.append(confidence)
                count_TRUE += 1
                print("counter_correct - " + str(counter_correct))
                cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (255,255,255), -1)
                cv2.putText(im, str(Id), (x,y-40), font, 1, (0,0,0), 2)                
        cv2.imshow('Webcam',im) 
        if tester: 
            break
        if cv2.waitKey(10) & 0xFF == ord('*'):  # If '*' is pressed, terminate the  program
            break
        #if counter_correct > 600:
        #   break
        #if counter_wrong > 100:
        #   break
    cam.release()
    cv2.destroyAllWindows()
    print(f"Time -->{time.time() - start_time}")
    print(f'True/Total{count_TRUE/count_TOTAL}')
    f = open("result_1.txt", "w")
    f.write(str(ll))
    f.close()
    return tester

main_recognize()
