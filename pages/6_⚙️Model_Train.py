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
st.title("Model Training and Evaluation")
st.text("""    With our dataset prepared and split, we can now train our machine learning model to predict Alzheimer's disease. We will use a classification algorithm through a Convolutional Neural Network (CNN) to train our model on the training set. After training, we will evaluate the performance of our model on the validation set to tune hyperparameters and prevent overfitting. Finally, we will test our model on the unseen test set to assess its accuracy, precision, recall, and F1-score. This evaluation will help us understand how well our model can predict Alzheimer's disease based on the features extracted from the EEG signals.
    """)
st.image(["/Users/nabijade/Downloads/dashboard_visuals/trainVal.png", "/Users/nabijade/Downloads/dashboard_visuals/cm.png"], caption=["Model training and evaluation process for predicting Alzheimer's disease", "Confusion matrix showing the performance of the machine learning model in predicting Alzheimer's disease"], width=350)
