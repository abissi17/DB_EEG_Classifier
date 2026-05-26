"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import numpy as np
import torch
from cnn_classes import CSANet2, ShuffleAttention
from io import BytesIO

# Below must be included in every page file for the logo and website title to be the same across all pages
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

with st.sidebar:
    st.info("Please see **📈Motivation** for more!")
    
st.title('Predicting Alzheimer\'s Disease with Machine Learning')
st.text("By Decoded Brain")

st.page_link("/Users/nabijade/Desktop/Repositories/DB_EEG_Classifier/pages/1_📈Motivation.py", label="Go to Motivation Page", icon="ℹ️")

# Load the trained model
@st.cache_resource
def load_model(model_name):
    import io
    import pickle
    import __main__
    import torch.storage as _torch_storage

    __main__.CSANet2 = CSANet2
    __main__.ShuffleAttention = ShuffleAttention

    # jake_cnn_2.pkl was saved with raw pickle.dump (not torch.save), so
    # torch.load misreads the format. We load with plain pickle instead.
    # The CUDA storage bytes embedded in the pickle are loaded via
    # torch.storage._load_from_bytes, which calls torch.load internally
    # with no map_location. We patch it to force CPU before unpickling.
    _orig = _torch_storage._load_from_bytes
    _torch_storage._load_from_bytes = lambda b: torch.load(
        io.BytesIO(b), map_location='cpu'
    )
    try:
        with open(model_name, 'rb') as f:
            model = pickle.load(f)
    finally:
        _torch_storage._load_from_bytes = _orig

    model.eval()
    return model

model = load_model('jake_cnn_2.pkl')

# Prediction function
def predict(uploaded_file):
    try:
        # Load the .npy file from uploaded bytes
        bytes_data = uploaded_file.read()
        data = np.load(BytesIO(bytes_data))

        if data.ndim == 3:
            # Single epoch (19, 23, 118) — add batch dimension
            data = data[np.newaxis]
        elif data.ndim != 4:
            return f"Error: Expected 3D or 4D array, but got shape {data.shape}", None, None, None

        num_epochs, channels, freqs, times = data.shape
        if (channels, freqs, times) != (19, 23, 118):
            return f"Error: Expected shape (N, 19, 23, 118), but got {data.shape}", None, None, None

        # Convert all epochs to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Process all epochs and average predictions
        with torch.no_grad():
            predictions = model(data_tensor)
            probability = torch.sigmoid(predictions).mean().item()

        # Interpret result
        if probability >= 0.6:
            result = "Alzheimer's Disease Detected"
            confidence = probability * 100
        else:
            result = "Healthy (No Alzheimer's Disease)"
            confidence = (1 - probability) * 100

        return result, confidence, probability, num_epochs

    except Exception as e:
        return f"Error processing file: {str(e)}", None, None, None

uploaded_file = st.file_uploader("Upload a .npy file (EEG Data)", type=["npy"])
if uploaded_file is not None:
    result = predict(uploaded_file)

    if isinstance(result, tuple) and result[1] is not None:
        prediction, confidence, probability, num_epochs = result
        st.subheader("Prediction Result")
        st.write(f"**{prediction}**")
        st.write(f"Confidence: {confidence:.2f}%")
        st.write(f"Raw Probability: {probability:.4f}")
        st.write(f"Number of Epochs Processed: {num_epochs}")
    else:
        st.error(result[0])
