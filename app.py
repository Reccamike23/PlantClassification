##This program is created for Streamlit using writefile app.py to make it run as streamlit application

import os
import cv2
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np

modelpath = 'best.pt'
st.title('Insert your image for Plant Classification Prediction')
image = st.file_uploader('upload image', type=['png', 'jpg', 'jpeg', 'gif'])
if image:
   image = Image.open(image)
   st.image(image=image)
   model = YOLO(modelpath)
   result = model(image)
   names = result[0].names
   probability = result[0].probs.data.numpy()
   prediction = np.argmax(probability)

   st.write(prediction)
   if prediction == 0:
      st.write('This is Canada Buffaloberry')

   elif prediction == 1:
      st.write('This is Labrador Tea')

   elif prediction == 2:
      st.write('This is Prickly Rose')

   elif prediction == 3:
      st.write('This is Red osier Dogwood')

   elif prediction == 4:
      st.write('This is Velvet Leaved Blueberry')
   else:
        st.write('Not a recognized plant')

# Instruct user when no image is uploaded
else:
    st.write('Please upload an image file to predict.')

