
import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import gdown

# ========================
# Google Drive File IDs
# ========================
FILENAMES_ID = "1CyMcg6PmFDzQ1Mt9jmAgL4mj7jB73arI"   # filenames.pkl
EMBEDDINGS_ID = "11RCycijG4J-sHe8kTLL_D7mjHwISEiPV" # embeddings.pkl

FILENAMES_PATH = "filenames.pkl"
EMBEDDINGS_PATH = "embeddings.pkl"

# ========================
# Download files if missing
# ========================
if not os.path.exists(FILENAMES_PATH):
    gdown.download(f"https://drive.google.com/uc?id={FILENAMES_ID}", FILENAMES_PATH, quiet=False)

if not os.path.exists(EMBEDDINGS_PATH):
    gdown.download(f"https://drive.google.com/uc?id={EMBEDDINGS_ID}", EMBEDDINGS_PATH, quiet=False)

# ========================
# Load data
# ========================
feature_list = np.array(pickle.load(open(EMBEDDINGS_PATH, 'rb')))
filenames = pickle.load(open(FILENAMES_PATH, 'rb'))

# ========================
# Load Model
# ========================
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# ========================
# Custom CSS
# ========================
st.markdown("""<style>
/* Vibrant background gradient */
body {
    background: linear-gradient(135deg, #FF9A8B 0%, #FF6B95 50%, #FF8E53 100%);
    background-attachment: fixed;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
/* Main container */
.main-container {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
    padding: 2rem;
    margin: 2rem auto;
    max-width: 1200px;
    border: 1px solid rgba(255, 255, 255, 0.3);
}
/* Title */
.big-title {
    text-align: center;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #FF416C, #FF4B2B);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: 1px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #ffffff;
    margin-bottom: 2rem;
    font-weight: 400;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
}
/* Upload section */
.upload-section {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    padding: 2rem;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    margin-bottom: 2rem;
    border: 2px dashed #FF416C;
    transition: all 0.3s ease;
}
.upload-section:hover {
    border-color: #FF4B2B;
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(255, 75, 43, 0.3);
}
/* Recommendation cards */
.rec-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    margin-bottom: 1rem;
    position: relative;
    border: 1px solid rgba(255, 255, 255, 0.5);
}
.rec-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 25px rgba(255, 75, 43, 0.2);
}
.rec-card:before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 5px;
    background: linear-gradient(90deg, #FF416C, #FF4B2B);
}
.rec-caption {
    padding: 1rem;
    text-align: center;
    font-size: 1rem;
    color: #333;
    font-weight: 600;
}
/* Badge */
.rec-badge {
    position: absolute;
    top: 10px;
    right: 10px;
    background: #FF416C;
    color: white;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    z-index: 1;
}
/* Footer */
.footer {
    margin-top: 3rem;
    text-align: center;
    color: rgba(255, 255, 255, 0.8);
    font-size: 0.9rem;
    padding: 1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.2);
}
</style>""", unsafe_allow_html=True)

# ========================
# Main Container
# ========================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <h1 class="big-title">Fashion Finder AI</h1>
        <p class="subtitle">Discover your perfect style match with our intelligent recommendation system</p>
    </div>
""", unsafe_allow_html=True)

# ========================
# Upload Section
# ========================
with st.container():
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("""
        <h3 style="text-align: center; color: #FF416C; margin-bottom: 1.5rem;">📤 Upload Your Fashion Item</h3>
        <p style="text-align: center; color: #666; margin-bottom: 1.5rem;">
            Upload an image of your favorite clothing item to discover similar styles that match your taste
        </p>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# Functions
# ========================
def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# ========================
# Display Results
# ========================
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        st.image(display_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        indices = recommend(features, feature_list)

        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                st.markdown(f'<div class="rec-badge">{i + 1}</div>', unsafe_allow_html=True)
                st.markdown('<div class="rec-card">', unsafe_allow_html=True)
                st.image(filenames[indices[0][i]], use_container_width=True)
                st.markdown(
                    f'<div style="padding:1rem; text-align:center; font-size:1rem; font-weight:600; background:linear-gradient(90deg, #6DD5FA, #FFFFFF); -webkit-background-clip:text; color:transparent;">Style Match <span style="color:#2196F3;">#{i + 1}</span></div>',
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("❌ Error uploading the file. Please try again.")

# ========================
# Footer
# ========================
st.markdown("""
    <div class="footer">
        <p>Developed with ❤ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)

# Close container
st.markdown('</div>', unsafe_allow_html=True)
