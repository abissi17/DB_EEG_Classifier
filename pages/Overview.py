import streamlit as st

st.title("Why Predict Alzheimer's Disease?")
st.text("""
    Alzheimer's disease is a progressive neurodegenerative disorder that affects millions of people worldwide. Early diagnosis is crucial for managing the disease and improving the quality of life for patients and their families. In this project, we will explore how machine learning can be used to predict the likelihood of Alzheimer's disease based on various features such as age, cognitive test scores, and brain imaging data.

    We will use a dataset that contains information about patients, including their demographic details, medical history, and results from cognitive tests. Our goal is to build a machine learning model that can accurately classify patients as either having Alzheimer's disease or being healthy.

    Let's get started by loading the dataset and performing some exploratory data analysis!
    """)

#========================================================================================
st.title("Cleaning EEG Signals")
st.text("""    Before we can use EEG signals for machine learning, we need to clean the data to remove noise and artifacts. EEG signals can be contaminated by various sources of noise, such as muscle activity, eye movements, and electrical interference. In this section, we will apply some common preprocessing techniques to clean the EEG signals.

    We will start by loading the raw EEG data and applying a bandpass filter to remove frequencies that are outside the typical range of brain activity (0.5 - 50 Hz). Next, we will use Independent Component Analysis (ICA) to identify and remove artifacts from the EEG signals. Finally, we will normalize the cleaned signals to ensure that they are on the same scale for machine learning.

    Let's go through each of these steps in detail and see how they improve the quality of our EEG data for predicting Alzheimer's disease.
    """)

st.image("/Users/nabijade/Downloads/1dashboard visuals/dirty.png", caption="Raw EEG signal with noise and artifacts")
st.image("/Users/nabijade/Downloads/1dashboard visuals/clean.png", caption="Cleaned EEG signal after preprocessing")

#========================================================================================
st.title("Segmenting EEG Signals")
st.text("""    After cleaning the EEG signals, the next step is to segment the continuous EEG data into smaller epochs that can be used for feature extraction and machine learning. Segmenting the EEG signals allows us to analyze specific time windows of brain activity, which can be crucial for identifying patterns associated with Alzheimer's disease.

    We will segment the EEG signals into epochs of a fixed length (e.g., 1 second) and apply a sliding window approach to create overlapping segments. This will help us capture more information from the EEG data and improve the performance of our machine learning model.

    Let's see how we can segment the EEG signals and prepare them for feature extraction in the next step.
    """)
st.image("/Users/nabijade/Downloads/1dashboard visuals/segment.png", caption="Segmented EEG signal ready for feature extraction")

#========================================================================================
st.title("Feature Extraction from EEG Signals")
st.text("""    Once we have segmented the EEG signals, we need to extract meaningful features that can be used for machine learning. Feature extraction is the process of transforming raw EEG data into a set of numerical features that capture important characteristics of the brain activity. We will extract features such as power spectral density, coherence, and entropy from the segmented EEG signals. These features can help us identify patterns in the brain activity that are associated with Alzheimer's disease. We will use these features to train our machine learning model and evaluate its performance in predicting Alzheimer's disease.
    """)
st.image("/Users/nabijade/Downloads/1dashboard visuals/ADSpect.png", caption="Power Spectral Density feature extracted from EEG signals with AD")
st.image("/Users/nabijade/Downloads/1dashboard visuals/HSpect.png", caption="Power Spectral Density feature extracted from EEG signals without AD")

#========================================================================================
st.title("Data Split: 70/12.5/12.5")
st.text("""    After extracting features from the EEG signals, we need to split our dataset into three parts: training, validation, and test sets. A common split ratio is 70% for training, 12.5% for validation, and 12.5% for testing. The training set is used to train our machine learning model, the validation set is used to tune hyperparameters and prevent overfitting, and the test set is used to evaluate the final performance of our model on unseen data. This split ensures that we have enough data for training while also allowing us to assess the generalization of our model to new data.
    """) 
st.image("/Users/nabijade/Downloads/1dashboard visuals/datasplit.png", caption="Data split into training, validation, and test sets")
   
#========================================================================================
st.title("Model Training and Evaluation")
st.text("""    With our dataset prepared and split, we can now train our machine learning model to predict Alzheimer's disease. We will use a classification algorithm, such as Random Forest or Support Vector Machine, to train our model on the training set. After training, we will evaluate the performance of our model on the validation set to tune hyperparameters and prevent overfitting. Finally, we will test our model on the unseen test set to assess its accuracy, precision, recall, and F1-score. This evaluation will help us understand how well our model can predict Alzheimer's disease based on the features extracted from the EEG signals.
    """)
st.image("/Users/nabijade/Downloads/1dashboard visuals/trainVal.png", caption="Model training and evaluation process for predicting Alzheimer's disease")
st.image("/Users/nabijade/Downloads/1dashboard visuals/cm.png", caption="Confusion matrix showing the performance of the machine learning model in predicting Alzheimer's disease")

#========================================================================================
st.title("Conclusion")
st.text("""    In this project, we have explored how machine learning can be used to predict Alzheimer's disease based on EEG signals. We went through the entire process, starting from data cleaning and preprocessing, to feature extraction, data splitting, and model training and evaluation. Our machine learning model was able to achieve a certain level of accuracy in predicting Alzheimer's disease, demonstrating the potential of using EEG signals from django.conf import settings for early diagnosis. However, there is still room for improvement, and future work could involve exploring more advanced machine learning algorithms, incorporating additional features, and using larger datasets to further enhance the predictive performance of the model. Overall, this project highlights the importance of machine learning in healthcare and its potential to improve the diagnosis and management of Alzheimer's disease.
    """)

