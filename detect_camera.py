from flask import Flask, render_template, Response
import cv2
import time
import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
model1 = load_model(r'C:\projects\deepfake_detection_app\deepfake\25_3.h5')


def detect_barcodes(camera):
    global is_detection_running
    model_images = load_model(r'C:\projects\deepfake_detection_app\deepfake\25_3.h5')

    while is_detection_running:
        success, frame = camera.read()
        if not success:
            break
        
        resized_img = cv2.resize(frame, (224, 224))  # Adjust the resize dimensions to (224, 224)
        input_img = resized_img / 255.0  # Normalize input images to [0, 1]
        input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension
            
        predictions1 = model1.predict(input_img)

            
        pred = "FAKE" if predictions1 >= 0.7 else "REAL"
        print(f'The predicted class of the video is {pred}')


        cv2.putText(frame, f"PREDICTION: {pred}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate(camera):
    global is_detection_running
    while is_detection_running:
        yield next(detect_barcodes(camera))




