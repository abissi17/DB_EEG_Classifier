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
st.title("Feature Extraction from EEG Signals")
st.text("""    Once we have segmented the EEG signals, we need to extract meaningful features that can be used for machine learning. Feature extraction is the process of transforming raw EEG data into a set of numerical features that capture important characteristics of the brain activity. We will extract features such as frequency bins and time frequency bins from the segmented EEG signals. These features can help us identify patterns in the brain activity that are associated with Alzheimer's disease. We will use these features to train our machine learning model and evaluate its performance in predicting Alzheimer's disease.
    """)
st.image(["/Users/nabijade/Downloads/dashboard_visuals/ADSpect.png", "/Users/nabijade/Downloads/dashboard_visuals/HSpect.png"], caption=["Power Spectral Density feature extracted from EEG signals with AD", "Power Spectral Density feature extracted from EEG signals without AD"], width=350)
