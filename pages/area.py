import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Inspect the Size of ROI Area",
    page_icon=":eye:",
)

with st.sidebar:
        # st.title("Defect Detection")
        st.subheader(" Samples area shows distribution of areas")

df = pd.read_csv('sample_losses.csv')

fig_mpl, ax_mpl = plt.subplots()
ax_mpl = plt.hist(df['area'])
plt.xlabel('area')
plt.ylabel('number of samples')
st.pyplot(fig_mpl)

cnt = len(df.index)
st.write(f'number of samples: {cnt}')