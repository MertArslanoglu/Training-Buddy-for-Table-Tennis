import cv2
import time
import argparse
import torch
from detect_analysis import detect

cap = cv2.VideoCapture(1)
check_point = 0
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
video = cv2.VideoWriter("D:\\YOLOv7\\yolov7-custom\\metalurji\\vidin.avi", fourcc, 60.0, (int(cap.get(3)), int(cap.get(4))))
flag = False
print(cap.isOpened())
while cap.isOpened():
    with open('./comm/comm.txt') as f:
        a = f.readlines()[0]


    holder = check_point
    check_point = time.time()
    ret, frame = cap.read()
    if ret:
        print(1/(check_point - holder))
        if flag:
            video.write(frame)



    #cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    cv2.imshow("Frame", frame)
    with open('./comm/comm.txt') as f:
        if a == "start":
            flag = True
        if a == "stop": ##esc
            break
cap.release()
video.release()
cv2.destroyAllWindows()

detect()