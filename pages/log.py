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

def format_date(date_str):
  """Returns the current date in YYYYMMDD format."""
  fmt = '%Y-%m-%d'
  new_format = '%Y%m%d'

  date_obj = datetime.datetime.strptime(date_str, fmt)
  formatted_date = date_obj.strftime(new_format)
  return formatted_date

# input date in yyyy-mm-dd format
date = st.date_input("What was the inspection date?", date.today())
formatted_date = format_date(str(date))
# st.write("date", formatted_date)

# Create the SQL connection  
connection = st.connection('log', type='sql')

# Select some data with connection.session.
with connection.session as session:
    query = f"select * from log where date_time like '{formatted_date}%';"
    log = session.execute(text(query))
    df = pd.DataFrame(log)
    st.write(f"Defect found on {formatted_date}", df)


# # extract date to select only the input date
# df['date'] = pd.to_datetime(df['date_time']).dt.date

# if date == pd.to_datetime(df['date_time']).dt.date: 
#         st.write("Defect found on ", df)

