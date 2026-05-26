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

#========================================================================================
st.title("Cleaning EEG Signals")
st.text("""    Before we can use EEG signals for machine learning, we need to clean the data to remove noise and artifacts. EEG signals can be contaminated by various sources of noise, such as muscle activity, eye movements, and electrical interference. In this section, we will apply some common preprocessing techniques to clean the EEG signals.

    We will start by loading the raw EEG data and applying a bandpass filter to remove frequencies that are outside the typical range of brain activity (0.5 - 50 Hz). Next, Artifact Subspace Reconstruction (ASR) was applied using a standard deviation of 17 to remove the noisiest artifacts from each channel signal. Subsequently, Independent Component Analysis (ICA) was applied to completely unmix the frequency signal and smooth out the frequency wave. Finally, we will normalize the cleaned signals to ensure that they are on the same scale for machine learning.

    Let's go through each of these steps in detail and see how they improve the quality of our EEG data for predicting Alzheimer's disease.
    """)

st.image(["/Users/nabijade/Downloads/dashboard_visuals/dirty.png", "/Users/nabijade/Downloads/dashboard_visuals/clean.png"], caption=["Raw EEG signal with noise and artifacts", "Cleaned EEG signal after preprocessing"], width=350)
