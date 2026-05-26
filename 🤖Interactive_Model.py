"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
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

# Define the CNN model class (required for unpickling)
# jerry's cnn class
class ShuffleAttention(nn.Module):
    def __init__(self, channels=128, groups=8):
        super().__init__()

        assert channels % (2 * groups) == 0, \
            "channels must be divisible by 2 * groups"

        self.channels = channels
        self.groups = groups

        branch_channels = channels // (2 * groups)

        # Channel attention branch
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_fc = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Spatial attention branch
        self.spatial_norm = nn.GroupNorm(
            num_groups=1,
            num_channels=branch_channels
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # PyTorch built-in ChannelShuffle
        self.channel_shuffle = nn.ChannelShuffle(groups)

    def forward(self, x):
        B, C, H, W = x.shape
        G = self.groups

        # split feature map into G groups
        x = x.reshape(B * G, C // G, H, W)

        # split each group into channel-attention branch and spatial-attention branch
        x_channel, x_spatial = torch.chunk(x, chunks=2, dim=1)

        # channel attention
        channel_weight = self.channel_fc(self.avg_pool(x_channel))
        x_channel = x_channel * channel_weight

        # spatial attention
        spatial_weight = self.spatial_conv(self.spatial_norm(x_spatial))
        x_spatial = x_spatial * spatial_weight

        # concat two branches
        out = torch.cat([x_channel, x_spatial], dim=1)

        # restore to original shape
        out = out.reshape(B, C, H, W)

        # shuffle channels
        out = self.channel_shuffle(out)

        return out
class CSANet2(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.6)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.6)
        )

        self.attention = ShuffleAttention(channels=128, groups=8)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # (B, 19, 23, 118) -> (B, 1, 19*23, 118)
        x = x.reshape(B, 1, C * H, W)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.attention(x)

        x = self.global_pool(x)
        x = x.flatten(1)

        x = self.fc(x)

        return x

# Load the trained model
@st.cache_resource
def load_model(model_name):
    model = CSANet2()
    state_dict = torch.load(model_name, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model('jerry_model.pkl')

# Prediction function
def predict(uploaded_file):
    try:
        # Load the .npy file from uploaded bytes
        bytes_data = uploaded_file.read()
        data = np.load(BytesIO(bytes_data))

        # Multiple epochs: (num_epochs, 19, 23, 118)
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
        if probability >= 0.5:
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




