import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import gdown
import os
import re
import logging
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors

# Suppress TensorFlow warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# -------------------
# Google Drive File IDs
# -------------------
EMBEDDINGS_URL = "https://drive.google.com/uc?id=11RCycijG4J-sHe8kTLL_D7mjHwISEiPV"
FILENAMES_URL = "https://drive.google.com/uc?id=1CyMcg6PmFDzQ1Mt9jmAgL4mj7jB73arI"

# Paths
EMBEDDINGS_PATH = "embeddings.pkl"
FILENAMES_PATH = "filenames.pkl"

# -------------------
# Download Data if Missing
# -------------------
def download_files():
    if not os.path.exists(EMBEDDINGS_PATH):
        gdown.download(EMBEDDINGS_URL, EMBEDDINGS_PATH, quiet=False)
    if not os.path.exists(FILENAMES_PATH):
        gdown.download(FILENAMES_URL, FILENAMES_PATH, quiet=False)

download_files()

# -------------------
# Cache model and data
# -------------------
@st.cache_resource
def load_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    return tf.keras.Sequential([base_model, GlobalMaxPooling2D()])

@st.cache_data
def load_features():
    feature_list = np.array(pickle.load(open(EMBEDDINGS_PATH, 'rb')))
    filenames = pickle.load(open(FILENAMES_PATH, 'rb'))
    return feature_list, filenames

model = load_model()
feature_list, filenames = load_features()

# -------------------
# Helper Functions
# -------------------
def save_uploaded_file(uploaded_file):
    filename_clean = re.sub(r'[^a-zA-Z0-9_.-]', '_', uploaded_file.name)
    try:
        with open(filename_clean, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return filename_clean
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    return result / np.linalg.norm(result)

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Fashion Finder AI", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #fafafa;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        font-size: 48px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 10px;
    }
    .rec-caption {
        font-size: 18px;
        color: #2874A6;
        text-align: center;
        margin-top: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">👗 Fashion Finder AI</div>', unsafe_allow_html=True)
st.write("Upload an image to find similar fashion items from our dataset!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_path = save_uploaded_file(uploaded_file)
    if img_path:
        display_image = Image.open(img_path)
        st.image(display_image, caption="Uploaded Image", use_container_width=True)

        # Extract features
        features = extract_features(img_path, model)
        indices = recommend(features, feature_list)

        st.subheader("Similar Items Found:")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            if i < len(indices[0]):
                with col:
                    st.image(filenames[indices[0][i]], use_container_width=True)
                    st.markdown(f'<div class="rec-caption">Style Match #{i+1}</div>', unsafe_allow_html=True)
