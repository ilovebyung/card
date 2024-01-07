# Build UI and embedding functions
# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models 
import streamlit as st  
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np
import cv2
import util

# hide deprication warnings which directly don't affect the working of the application
import warnings
# warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Detect Detection",
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

# def prediction_cls(prediction): # predict the class of the images based on the model results
#     for key, clss in class_names.items(): # create a dictionary of the output classes
#         if np.argmax(prediction)==clss: # check the class
            
#             return key

with st.sidebar:
        st.title("Defect Detection")
        st.subheader(" Defect detection helps an user to spot detected area.")

st.write("""
         # Defect Detection  
         """
         )

file = st.file_uploader("", type=["jpg", "png"])

# def import_and_predict(image_data, model):
#         size = (224,224)    
#         image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
#         img = np.asarray(image)
#         img_reshape = img[np.newaxis,...]
#         prediction = model.predict(img_reshape)
#         return prediction

        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    '''process comparision from tf'''
    # image = cv2.imread(file, 0)
    # plt.imshow(image, cmap='gray')

    # calculate image loss
    numpy_array = np.array(image)
    loss = util.image_loss(numpy_array)
    st.write(loss)

    # calculate decoded image
    decoded = util.decoded_image(numpy_array)

    # show difference
    difference = util.diff_image(numpy_array,decoded)
    st.image(difference, use_column_width=True)

    # predictions = import_and_predict(image, model)
    # x = random.randint(98,99)+ random.randint(0,99)*0.01
    # st.sidebar.error("Accuracy : " + str(x) + " %")

    # class_names = ['Anthracnose', 'Bacterial Canker','Cutting Weevil','Die Back','Gall Midge','Healthy','Powdery Mildew','Sooty Mould']

    # string = "Detected Disease : " + class_names[np.argmax(predictions)]
    # if class_names[np.argmax(predictions)] == 'Healthy':
    #     st.balloons()
    #     st.sidebar.success(string)

    # elif class_names[np.argmax(predictions)] == 'Anthracnose':
    #     st.sidebar.warning(string)
    #     st.markdown("## Remedy")
    #     st.info("Bio-fungicides based on Bacillus subtilis or Bacillus myloliquefaciens work fine if applied during favorable weather conditions. Hot water treatment of seeds or fruits (48°C for 20 minutes) can kill any fungal residue and prevent further spreading of the disease in the field or during transport.")

