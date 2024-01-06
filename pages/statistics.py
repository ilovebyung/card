import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import datetime as dt
st.title('Sample Losses')
st.write('number of samples:19, mean:0.0058')
df = pd.read_csv('sample_losses.csv')

fig_mpl, ax_mpl = plt.subplots()
ax_mpl = plt.hist(df['loss'])
plt.xlabel('loss')
plt.ylabel('number of samples')
st.pyplot(fig_mpl)