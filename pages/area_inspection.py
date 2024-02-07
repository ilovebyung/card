# Build UI and embedding functions
# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models 
import streamlit as st  
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import cv2
import util

min_area = 750923
max_area = 893320

# st.write(""" Area Calculation """)

file = st.file_uploader("", type=["jpg"])

        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    # calculate image area
    numpy_array = np.array(image)

    # Apply Otsu's thresholding to segment the foreground
    ret, thresh = cv2.threshold(numpy_array, 80, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate the area (number of black pixels)
    area = np.sum(thresh == 255) 

    if area < min_area:
      st.write(f":red[Area of {area} is less than threshold ]")
    else:
      st.write(f":blue[Area of {area} is within range ]")

    st.write(f"Diagnosed Image")
    
    # show thresh
    st.image(thresh, use_column_width=True)

    # st.write(f"Diagnosed Image with different colormap")
    # diff_image_colormap = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    # st.image(diff_image_colormap, use_column_width=True)
