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


# -------- Google Drive से pickle download --------
def download_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

# तुम्हारे Google Drive file IDs
EMBEDDINGS_ID = "11RCycijG4J-sHe8kTLL_D7mjHwISEiPV"  # embeddings.pkl
FILENAMES_ID = "1CyMcg6PmFDzQ1Mt9jmAgL4mj7jB73arI"    # filenames.pkl

# Download अगर local में files नहीं हैं
download_from_drive(EMBEDDINGS_ID, "embeddings.pkl")
download_from_drive(FILENAMES_ID, "filenames.pkl")

# Load pickle files
feature_list = pickle.load(open("embeddings.pkl", "rb"))
filenames = pickle.load(open("filenames.pkl", "rb"))

# Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Custom CSS for Styling
st.markdown("""
    <style>
        /* Vibrant background gradient */
        body {
            background: linear-gradient(135deg, #FF9A8B 0%, #FF6B95 50%, #FF8E53 100%);
            background-attachment: fixed;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Main container with glass morphism effect */
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

        /* Title styling with vibrant gradient */
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

        /* Upload section styling */
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

        /* Card styling for recommendations */
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

        /* Badge for recommendation number */
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

        /* Section headers */
        .section-header {
            font-size: 1.8rem;
            color: white;
            margin: 1.5rem 0;
            font-weight: 700;
            position: relative;
            display: inline-block;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
        }

        .section-header:after {
            content: "";
            position: absolute;
            bottom: -8px;
            left: 0;
            width: 50px;
            height: 4px;
            background: linear-gradient(90deg, #FF416C, #FF4B2B);
            border-radius: 2px;
        }

        /* Footer styling */
        .footer {
            margin-top: 3rem;
            text-align: center;
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9rem;
            padding: 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.2);
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
            .big-title {
                font-size: 2.5rem;
            }
            .subtitle {
                font-size: 1rem;
            }
        }

        /* File uploader customization */
        .stFileUploader > div > div {
            border: 2px dashed #FF416C !important;
            background: rgba(255, 255, 255, 0.9) !important;
            border-radius: 15px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header with animated gradient text
st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <h1 class="big-title">Fashion Finder AI</h1>
        <p class="subtitle">Discover your perfect style match with our intelligent recommendation system</p>
    </div>
""", unsafe_allow_html=True)

# Upload section with improved styling
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


# Functions
def save_uploaded_file(uploaded_file):
    try:
        os.makedirs('uploads', exist_ok=True)  # Agar folder nahi hai to banado
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"Error saving file: {e}")
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


# Display results
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display uploaded image with styling
        display_image = Image.open(uploaded_file)
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">Your Style Inspiration</h2>', unsafe_allow_html=True)
        st.image(display_image,
                 use_container_width=True,
                 caption="")
        st.markdown('</div>', unsafe_allow_html=True)

        # Extract features
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # Get recommendations
        indices = recommend(features, feature_list)

        # Display recommendations with improved layout
        st.markdown('<h2 class="section-header">✨ Recommended For You</h2>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color: rgba(255,255,255,0.9); margin-bottom: 1.5rem; text-align: center;">We found these items that match your style perfectly</p>',
            unsafe_allow_html=True)

        # Create columns for recommendations
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

        # Add a call-to-action button
        st.markdown("""
            <div style="text-align: center; margin-top: 2rem;">
                <p style="color: rgba(255,255,255,0.9); margin-bottom: 1rem;">Love what you see?</p>
            </div>
        """, unsafe_allow_html=True)

    else:
        st.error("❌ Error uploading the file. Please try again.")
else:
    # Show placeholder content when no image is uploaded
    st.markdown("""
        <div style="text-align: center; margin: 3rem 0; padding: 2rem; background: rgba(255, 255, 255, 0.9); border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
            <div style="font-size: 5rem; margin-bottom: 1rem; color: #FF416C;">👗</div>
            <h3 style="color: #FF416C;">Ready to Discover Your Style?</h3>
            <p style="color: #666;">Upload an image of your favorite fashion item to get personalized recommendations</p>
        </div>
    """, unsafe_allow_html=True)

# Footer with social links
st.markdown("""
    <div class="footer">
        <p>Developed with ❤ using Streamlit</p>
        <div style="margin-top: 0.5rem;">
            <a href="#" style="margin: 0 10px; color: rgba(255,255,255,0.8); text-decoration: none;">Instagram</a>
            <a href="#" style="margin: 0 10px; color: rgba(255,255,255,0.8); text-decoration: none;">Twitter</a>
            <a href="#" style="margin: 0 10px; color: rgba(255,255,255,0.8); text-decoration: none;">GitHub</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# Close main container
st.markdown('</div>', unsafe_allow_html=True)



