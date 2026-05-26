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
st.title("Conclusion")
st.text("""    In this project, we have explored how machine learning can be used to predict Alzheimer's disease based on EEG signals. 
        We went through the entire process, starting from data cleaning and preprocessing, to feature extraction, data splitting, and model training and evaluation. 
        Our machine learning model was able to achieve a certain level of accuracy in predicting Alzheimer's disease for early diagnosis. 
        However, there is still room for improvement, and future work could involve exploring more advanced machine learning algorithms, incorporating additional features, and using larger datasets to further enhance the predictive performance of the model. 
        Overall, this project highlights the importance of machine learning in healthcare and its potential to improve the diagnosis and management of Alzheimer's disease.
    """)