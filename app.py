import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 1. Labels must be in Alphabetical Order to match your Notebook's training
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@st.cache_resource
def load_model():
    model_path = 'emotiondetector.h5' 
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error(f"Model file '{model_path}' not found. Please ensure it is uploaded via Git LFS.")
        return None

model = load_model()

st.set_page_config(page_title="Real-time Emotion AI", page_icon="🎭")
st.title("🎭 Face Emotion Recognition")

# 2. Input Options: Upload or Live Camera
option = st.radio("Select Input Source:", ("Live Camera", "Upload Image"))

if option == "Live Camera":
    source = st.camera_input("Smile for the camera!")
else:
    source = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])

# 3. Processing the Image
if source is not None and model is not None:
    # --- PREPROCESSING (The "91% Accuracy" Recipe) ---
    # Convert image to Grayscale
    img = Image.open(source).convert('L') 
    
    # Resize to 48x48
    img_resized = img.resize((48, 48))
    
    # Normalize (0 to 1) and Reshape for the CNN (1, 48, 48, 1)
    img_array = np.array(img_resized) / 255.0
    img_reshaped = img_array.reshape(1, 48, 48, 1)

    # --- INFERENCE ---
    with st.spinner('Analyzing facial expressions...'):
        predictions = model.predict(img_reshaped)
        
        # Get ONLY the highest confidence result
        max_idx = np.argmax(predictions[0])
        final_emotion = CLASS_NAMES[max_idx]
        confidence = np.max(predictions[0]) * 100

    # --- DISPLAY HIGHEST CONFIDENCE ONLY ---
    st.divider()
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(img, caption="Face Capture", width=150)
        
    with col2:
        st.write("### Prediction")
        # Highlighting only the winner
        st.success(f"**Emotion:** {final_emotion}")
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

    # Add a small note for recruiters
    st.info("Technical Info: This model is a deep CNN trained on 48x48 grayscale images with 91.21% accuracy.")