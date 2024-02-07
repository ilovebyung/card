# Build UI and embedding functions
# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models 
import streamlit as st  
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import cv2
import util

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Detect Detection Home",
    initial_sidebar_state = 'auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML


with st.sidebar:
        # st.title("Defect Detection")
        st.subheader(" Defect Detection helps an user to identify a defected part and spot detected area")


st.title(""" Defect Detection Home """)

