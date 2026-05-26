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
st.title("Data Split: 70/12.5/12.5")
st.text("""    After extracting features from the EEG signals, we need to split our dataset into three parts: training, validation, and test sets. A common split ratio is 70% for training, 12.5% for validation, and 12.5% for testing. The training set is used to train our machine learning model, the validation set is used to tune hyperparameters and prevent overfitting, and the test set is used to evaluate the final performance of our model on unseen data. This split ensures that we have enough data for training while also allowing us to assess the generalization of our model to new data.
    """) 
left_co, cent_co, right_co = st.columns(3)
with left_co: 
    st.image("/Users/nabijade/Downloads/dashboard_visuals/datasplit.png", caption="Data split into training, validation, and test sets", width=700)
   