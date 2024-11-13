import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Model
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import os
from IPython.display import Image
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Fashion Recommendation System",
    page_icon="👗",
    layout="wide"
)

# Custom CSS
# Update the .recommendation-title class in the CSS section to have white text color:

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stHeader {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    div.css-1r6slb0.e1tzin5v2 {
        background-color: #ffffff;
        border: 1px solid #eeeeee;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin: 0.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div.css-1r6slb0.e1tzin5v2:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .recommendation-title {
        color: #ffffff;  /* Changed to white */
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f2f6;
        background-color: #1f1f1f;  /* Added dark background for contrast */
        padding: 10px;  /* Added padding */
        border-radius: 8px;  /* Added rounded corners */
    }
    .recommendation-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
    }
    .recommendation-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .recommendation-header h2 {
        margin-bottom: 0;
        color:#fff
    }
    .match-number {
        background-color: #4CAF50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .similarity-score {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .image-container {
        position: relative;
        overflow: hidden;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .upload-placeholder {
        background-color: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    .stImage {
    max-width: 100%;
    height: auto;
    object-fit: contain;
    }
    </style>
    """, unsafe_allow_html=True)

# Header section
st.markdown('<div class="stHeader">', unsafe_allow_html=True)
st.title('🎭 Fashion Recommendation System')
st.markdown("""
    Upload an image of a fashion item, and we'll recommend similar items from our database!
""")
st.markdown('</div>', unsafe_allow_html=True)

# Load the saved features and filenames
Image_features = pkl.load(open('Images_features.pkl','rb'))
filenames = pkl.load(open('filenames.pkl','rb'))

def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result/norm(result)
    return norm_result

# Initialize the model
@st.cache_resource
def load_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])
    return model

model = load_model()

# Initialize NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# Upload section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
upload_file = st.file_uploader("Choose a fashion item image...", type=['jpg', 'jpeg', 'png'])

if upload_file is not None:
    try:
        # Create upload directory if it doesn't exist
        os.makedirs('upload', exist_ok=True)
        
        # Save and display uploaded image
        with open(os.path.join('upload', upload_file.name), 'wb') as f:
            f.write(upload_file.getbuffer())
        
        # Display uploaded image with title
        st.markdown('<div class="recommendation-title">📸 Your Uploaded Item</div>', unsafe_allow_html=True)
        st.image(upload_file)
        
        # Process image and get recommendations
        with st.spinner('Finding similar items...'):
            input_img_features = extract_features_from_images(os.path.join('upload', upload_file.name), model)
            distances, indices = neighbors.kneighbors([input_img_features])
            
            # Display recommendations section
            st.markdown('<div class="recommendation-container">', unsafe_allow_html=True)
            st.markdown('<div class="recommendation-title">🔍 Similar Fashion Items</div>', unsafe_allow_html=True)
            
            # Create columns for recommendations
            cols = st.columns(5)
            for idx, col in enumerate(cols):
                with col:
                    st.markdown(f'<div class="match-number">Match #{idx+1}</div>', unsafe_allow_html=True)
                    # Changed use_column_width=True to use_column_width=False to maintain original size
                    st.image(filenames[indices[0][idx+1]], use_container_width=False)
                    similarity = (1 - distances[0][idx+1]) * 100
                    st.markdown(f'<div class="similarity-score">Similarity: {similarity:.1f}%</div>', 
                            unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.markdown("Please try uploading a different image.")
else:
    # Display placeholder content
    st.markdown('<div class="upload-placeholder">', unsafe_allow_html=True)
    st.markdown("### 👆 Upload an image to get started!")
    st.markdown("""
    #### How it works:
    1. Upload a fashion item image
    2. Our AI analyzes the style and features
    3. Get 5 similar fashion recommendations
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
