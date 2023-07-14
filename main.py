from ultralytics import YOLO
import cv2
import math
import os
import glob
import numpy as np
import streamlit as st

# Load a model
model = YOLO('yolov8x-pose.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom trained


video_path = 0
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    results = model(frame, imgsz=256)
    annotated_frame = results[0].plot()
    print(results[0].tojson('data.json'))
    cv2.imshow("YOLOv8 pose inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Quit!")
        break

cap.release()
cv2.destroyAllWindows()
