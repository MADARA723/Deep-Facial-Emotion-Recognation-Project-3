import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 1. Alphabetical Labels (Must match your Notebook's training folders)
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@st.cache_resource
def load_model():
    model_path = 'emotiondetector.h5' 
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error(f"Model file '{model_path}' not found in the repository.")
        return None

model = load_model()

st.title("🎭 Face Emotion Recognition")
st.write("Instant AI-based facial expression analysis.")

# Input Source
file = st.file_uploader("Upload or take a photo", type=["jpg", "jpeg", "png"])

if file is not None and model is not None:
    # --- PREPROCESSING (The exact steps from your 91% notebook) ---
    # 1. Convert to Grayscale
    img = Image.open(file).convert('L') 
    
    # 2. Resize to 48x48
    img_resized = img.resize((48, 48))
    
    # 3. Normalize and Reshape: (1, 48, 48, 1)
    img_array = np.array(img_resized) / 255.0
    img_reshaped = img_array.reshape(1, 48, 48, 1)
    
    # --- PREDICTION ---
    with st.spinner('Analyzing...'):
        predictions = model.predict(img_reshaped)
        
        # 4. Get ONLY the highest confidence result
        max_idx = np.argmax(predictions[0])
        final_emotion = CLASS_NAMES[max_idx]
        confidence = np.max(predictions[0]) * 100

    # --- DISPLAY ONLY THE WINNER ---
    st.divider()
    
    # Display the processed face and the result side-by-side
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(img, caption="Detected Face", width=150)
        
    with col2:
        st.subheader("Analysis Result")
        # Highlighting the winner with a success box
        st.success(f"**Emotion:** {final_emotion}")
        st.info(f"**Confidence Score:** {confidence:.2f}%")

    # Business Logic for Recruiters (The "Why")
    st.markdown("---")
    st.caption("Technical Note: This prediction uses a custom CNN architecture with 91.21% validation accuracy.")