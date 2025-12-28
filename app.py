import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps
import os

# 1. Exact Labels from your Notebook
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load Face Detector (Built into OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@st.cache_resource
def load_emotion_model():
    model_path = 'emotiondetector2.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_emotion_model()

st.title("🎭 Live Emotion AI (High Accuracy)")

source = st.camera_input("Look at the camera and take a snapshot")

if source and model:
    # --- STEP 1: Convert to OpenCV format ---
    file_bytes = np.asarray(bytearray(source.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- STEP 2: Detect the Face ---
    # This removes the background and shoulders
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Crop to the face only
            roi_gray = gray_frame[y:y+h, x:x+w]
            # Resize to 48x48
            roi_resized = cv2.resize(roi_gray, (48, 48))
            # Normalize
            roi_normalized = roi_resized / 255.0
            # Reshape for model
            final_input = np.reshape(roi_normalized, (1, 48, 48, 1))

            # --- STEP 3: Prediction ---
            predictions = model.predict(final_input)
            max_idx = np.argmax(predictions[0])
            label = CLASS_NAMES[max_idx]
            conf = np.max(predictions[0]) * 100

            # --- STEP 4: UI Results ---
            st.divider()
            col1, col2 = st.columns([1, 2])
            with col1:
                # Show exactly what the model "sees"
                st.image(roi_resized, caption="Cropped Face", width=120)
            with col2:
                st.header(f"Detected: {label.capitalize()}")
                st.progress(int(conf))
                st.write(f"Confidence: {conf:.2f}%")
    else:
        st.warning("No face detected! Please move closer or check your lighting.")

st.info("Tip: The model works best when your face is centered and you aren't wearing glasses.")