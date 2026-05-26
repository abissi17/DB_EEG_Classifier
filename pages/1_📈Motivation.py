import streamlit as st
    
# Inject custom CSS to increase the logo size
st.html("""
    <style>
        [alt=Logo] {
            height: 3.5rem; /* Adjust this value to make it larger or smaller */
        }
    </style>
""")

# st.logo("/Users/nabijade/Downloads/dashboard_visuals/db-trans.png")
st.logo("/Users/nabijade/Downloads/dashboard_visuals/db-white-cover-removebg-preview.png")
    
# Setting DB logo for the browser tab    
st.set_page_config(
    page_title="Predicting Alzheimer's Disease with Machine Learning",
    page_icon="/Users/nabijade/Downloads/dashboard_visuals/db-trans.png"
) 

st.title("Why Predict Alzheimer's Disease?")
st.text("""
    Alzheimer's disease is a progressive neurodegenerative disorder that affects approximately 57 million people worldwide. Early diagnosis is crucial for managing the disease and improving the quality of life for patients and their families. In this project, we will explore how machine learning can be used to predict the likelihood of Alzheimer's disease based on the correlation between patients' EEG signals and their diagnosis. By analyzing the EEG signals, we can identify patterns that may indicate the presence of Alzheimer's disease, allowing for earlier intervention and better treatment outcomes.

    This project uses a dataset containing 88 patients' gender, age, diagnosis, and MMSE score. Our goal is to build a machine learning model that can accurately classify patients as either having Alzheimer's disease or being healthy using each patient's EEG signals. 
    """)