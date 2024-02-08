# Build UI and embedding functions
# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models 
import streamlit as st  
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import datetime
import util
from sqlalchemy import create_engine 
from sqlalchemy.sql import text

def format_filename():
  """Returns the current date in YYYYMMDD_mmddss format."""
  now = datetime.datetime.now()
  formatted_filename = now.strftime("%Y%m%d") + '_' +  now.strftime("%H%M%S") + '.jpg'
  return formatted_filename

with st.sidebar:
  # st.title("Defect Detection")
  st.subheader(" Defect Detection helps an user to identify a defected part and spot detected area")

# Create the SQL connection  
connection = st.connection('log', type='sql')

# st.write(""" Defect Detection """)

file = st.file_uploader("", type=["jpg"])

        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    # calculate image loss
    numpy_array = np.array(image)
    loss = util.image_loss(numpy_array)
    loss = round(loss,6)

    if loss > util.threshold:
      st.write(f":red[Loss of {loss} is greater than threshold ]")

      # insert into log
      with connection.session as session:
        formatted_filename = format_filename()
        session.execute(text(f"insert into log (date_time, material, measurement, detected) values ('{formatted_filename}', '5101341', {loss}, 1) ;"))
        session.commit()

    else:
      st.write(f":blue[Loss of {loss} is within range ]")

    st.write(f"Diagnosed Image")
    # calculate decoded image
    decoded = util.decoded_image(numpy_array)

    # show difference
    gray = util.diff_image(numpy_array,decoded)
    st.image(gray, use_column_width=True)

