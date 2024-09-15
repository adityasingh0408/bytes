# -- coding: utf-8 --


import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
import pandas as pd
model1 = load_model(r'C:\\projects\\deep fake\\25_3.h5')
# model2 = load_model(r'C:\\projects\\deep fake\\25_3.h5')


count = 0

ss = input("enter the file location: ")

img = cv2.imread(ss)
  # Skip if image not found
    
resized_img = cv2.resize(img, (224, 224))  # Adjust the resize dimensions to (224, 224)
input_img = resized_img / 255.0  # Normalize input images to [0, 1]
input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension
    
predictions1 = model1.predict(input_img)

    
pred = "FAKE" if predictions1 >= 0.3 else "REAL"
print(f'The predicted class of the video is {pred}')
    

cv2.destroyAllWindows()