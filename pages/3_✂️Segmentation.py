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
st.title("Segmenting EEG Signals")
st.text("""    After cleaning the EEG signals, the next step is to segment the continuous EEG data into smaller epochs that can be used for feature extraction and machine learning. Segmenting the EEG signals allows us to analyze specific time windows of brain activity, which can be crucial for identifying patterns associated with Alzheimer's disease.

    We will segment the EEG signals into epochs of a fixed length (e.g., 1 second) and apply a sliding window approach to create overlapping segments. This will help us capture more information from the EEG data and improve the performance of our machine learning model.

    Let's see how we can segment the EEG signals and prepare them for feature extraction in the next step.
    """)
st.image("/Users/nabijade/Downloads/dashboard_visuals/oneChannel.png", caption="Segmented EEG signal ready for feature extraction")
