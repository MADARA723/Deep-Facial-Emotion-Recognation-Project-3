import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps
import os

# 1. EXACT Labels from your Notebook (Lowercase & Alphabetical)
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@st.cache_resource
def load_emotion_model():
    model_path = 'emotiondetector.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_emotion_model()

st.set_page_config(page_title="DeepFER Emotion AI", page_icon="🎭")
st.title("🎭 Deep Facial Emotion Recognition")

# User Input
option = st.radio("Choose Input Method:", ("Live Camera", "Upload Photo"))
if option == "Live Camera":
    source = st.camera_input("Take a snapshot")
else:
    source = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if source and model:
    # --- PREPROCESSING (The "91% Recipe") ---
    # Step 1: Grayscale (Notebook requirement)
    img = Image.open(source).convert('L')
    
    # Step 2: Auto-contrast (Helps the model see features in low light)
    img = ImageOps.autocontrast(img)
    
    # Step 3: Resize to 48x48
    img_48 = img.resize((48, 48))
    
    # Step 4: Normalize and Reshape (1, 48, 48, 1)
    img_array = np.array(img_48) / 255.0
    img_reshaped = img_array.reshape(1, 48, 48, 1)

    # --- INFERENCE ---
    with st.spinner("Analyzing..."):
        predictions = model.predict(img_reshaped)
        max_idx = np.argmax(predictions[0])
        final_emotion = CLASS_NAMES[max_idx]
        confidence = np.max(predictions[0]) * 100

    # --- RESULTS ---
    st.divider()
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Show what the model actually sees (48x48 Grayscale)
        st.image(img_48, caption="Model's View", width=120)
        
    with col2:
        # Display the prediction in Bold/Capitalized for UI but using correct index
        st.subheader(f"Result: {final_emotion.capitalize()}")
        st.metric("Confidence", f"{confidence:.2f}%")
        
        # UI Feedback based on confidence
        if confidence < 50:
            st.warning("Low confidence. Make sure your face is centered and well-lit.")
        else:
            st.success("High confidence prediction!")

st.markdown("---")
st.caption("Training Accuracy: 91.21% | Dataset: FER2013 | Framework: TensorFlow/Keras")