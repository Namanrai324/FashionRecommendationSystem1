# Fashion Recommendation System1
👗 Fashion Finder AI

Fashion Finder AI is an AI-powered recommendation system that uses deep learning and computer vision to suggest visually similar fashion items based on an uploaded image.

🚀 Features

Image-based search – Upload an image and get 5 visually similar fashion recommendations.
Deep Learning powered – Uses ResNet50 for feature extraction.
Content-based filtering – Finds similar styles using Nearest Neighbors algorithm.
Interactive UI – Built with Streamlit for a modern and responsive interface.
Cloud storage integration – Embeddings stored and retrieved via Google Drive.

🛠 Tech Stack

Python – Core language
TensorFlow / Keras – Deep learning & feature extraction
scikit-learn – Similarity search with Nearest Neighbors
Streamlit – Web app framework
OpenCV – Image processing

# Project Structure
.
├── app.py                # Streamlit web app
├── feature_extraction.py # Script to generate image embeddings
├── embeddings.pkl        # Precomputed image embeddings
├── filenames.pkl         # List of image file paths
├── images/               # Dataset images
└── uploads/              # Uploaded images (runtime)

⚙️ Installation 

git clone https://github.com/yourusername/fashion-finder-ai.git
cd fashion-finder-ai
pip install -r requirements.txt
streamlit run app.py

📸 Usage

Open the web app.
Upload a clothing/fashion item image.
Get 5 similar style recommendations instantly.

📊 Model Details

Model: ResNet50 (pre-trained on ImageNet)
Feature Extraction: Global Max Pooling applied to convolutional layers
Similarity Metric: Euclidean distance via Nearest Neighbors

#❤️ Acknowledgements

TensorFlow
Streamlit
scikit-learn
