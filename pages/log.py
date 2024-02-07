import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pickle
from datetime import date
from sqlalchemy import create_engine 
from sqlalchemy.sql import text

with st.sidebar:
        # st.title("Defect Detection")
        st.subheader(" Detected defects by date")


with open('sample_losses.pkl', 'rb') as f:
    loss_data = pickle.load(f)


# input inspection date 2024-02-07
date = st.date_input("What was the inspection date?", date.today())
st.write("date", date)

# # extract date to select only the input date
# df['date'] = pd.to_datetime(df['date_time']).dt.date

# if date == pd.to_datetime(df['date_time']).dt.date: 
#         st.write("Defect found at ", df)



# Create the SQL connection to pets_db as specified in your secrets file.
connection = st.connection('log', type='sql')

with connection.session as session:
    session.execute(text('CREATE TABLE IF NOT EXISTS log (date_time TEXT, material INTEGER, measurement INTEGER, detected INTEGER);'))
    session.execute(text('CREATE TABLE IF NOT EXISTS material (material INTEGER, description TEXT, min INTEGER, max INTEGER);'))
    session.commit()

# # Select some data with conn.session.
# with connection.session as s:
#     rows = s.execute("select * from log")
#     df = pd.DataFrame(pet_owners)
#     st.write("Defect found at ", df)
    
    
