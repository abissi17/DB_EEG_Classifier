"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import numpy as np

st.title('Predicting Alzheimer\'s Disease with Machine Learning')
st.text("By Decoded Brain")
st.sidebar.success("Alzheimer Predictor")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    #prediction = predict(uploaded_file)
    #st.subheader(prediction)
    st.image(uploaded_file,caption="Uploaded Image", use_container_width=True)

st.text("""
    Alzheimer's disease is a progressive neurodegenerative disorder that affects millions of people worldwide. Early diagnosis is crucial for managing the disease and improving the quality of life for patients and their families. In this project, we will explore how machine learning can be used to predict the likelihood of Alzheimer's disease based on various features such as age, cognitive test scores, and brain imaging data.

    We will use a dataset that contains information about patients, including their demographic details, medical history, and results from cognitive tests. Our goal is to build a machine learning model that can accurately classify patients as either having Alzheimer's disease or being healthy.

    Let's get started by loading the dataset and performing some exploratory data analysis!
    """)





