# -*- coding: utf-8 -*-
# https://medium.com/@alimustoofaa/how-to-load-model-yolov8-onnx-cv2-dnn-3e176cde16e6
# https://learnopencv.com/ultralytics-yolov8/#How-to-Use-YOLOv8?
from ultralytics import YOLO

model = YOLO("yolov8s.pt")  
model.train(data="C:\\LicensePlate_Yolov8_Filters_PaddleOCR\\licence_data.yaml", epochs=100,batch=8)  # train the model
model.val()  # evaluate model performance on the validation set
