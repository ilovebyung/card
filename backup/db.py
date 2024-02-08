import streamlit as st
from sqlalchemy import create_engine 
import pandas as pd

'''
# Create .streamlit/secrets.toml
[connections.pets]
url = "sqlite:///log.db"
'''

# Create the SQL connection to pets_db as specified in your secrets file.
conn = st.connection('log', type='sql')

# Insert some data with conn.session.
with conn.session as s:
    s.execute('CREATE TABLE IF NOT EXISTS log (date_time TEXT, material INTEGER, measurement INTEGER, detected INTEGER);')
    s.execute('CREATE TABLE IF NOT EXISTS material (material INTEGER, description TEXT, min INTEGER, max INTEGER);')
    pet_owners = {'jerry': 'fish', 'barbara': 'cat', 'alex': 'puppy'}
    for k in pet_owners:
        s.execute(
            'INSERT INTO pet_owners (person, pet) VALUES (:owner, :pet);',
            params=dict(owner=k, pet=pet_owners[k])
        )
    s.commit()

# Select some data with conn.session.
with conn.session as s:
    rows = s.execute("select * from log")
    df = pd.DataFrame(pet_owners)

