import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# 1. Exact labels from your notebook (Lowercase)
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@st.cache_resource
def load_emotion_model():
    model_path = 'emotiondetector.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_emotion_model()

st.set_page_config(page_title="DeepFER Emotion Analyzer", layout="wide")
st.title("🎭 Deep Facial Emotion Recognition")

# 2. Input Options
option = st.sidebar.radio("Input Method", ("Live Camera", "Upload Image"))

if option == "Live Camera":
    source = st.camera_input("Take a snapshot")
else:
    source = st.file_uploader("Upload an image (Dataset images supported)", type=["jpg", "png", "jpeg"])

# 3. Processing Function
def process_and_predict(input_source, is_upload=False):
    # Convert to OpenCV format
    file_bytes = np.asarray(bytearray(input_source.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- FACE DETECTION LOGIC ---
    if is_upload:
        # Relaxed detection for small dataset images
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))
    else:
        # Stricter detection for live camera to ignore background
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Process ROI
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
    else:
        if is_upload:
            # If upload and no face found, it's likely already a cropped face (like in datasets)
            st.info("No face bounding box detected. Processing the full image as a face.")
            roi_gray = gray
        else:
            return None, None, "No face detected in camera. Please get closer and ensure good lighting."
    
    # Prepare for model
    roi_resized = cv2.resize(roi_gray, (48, 48))
    roi_normalized = roi_resized / 255.0
    final_input = np.reshape(roi_normalized, (1, 48, 48, 1))
    
    # Prediction
    preds = model.predict(final_input)[0]
    return roi_resized, preds, None

# 4. Display Logic
if source and model:
    is_upload_mode = (option == "Upload Image")
    cropped_face, predictions, error = process_and_predict(source, is_upload=is_upload_mode)
    
    if error:
        st.error(error)
    else:
        st.divider()
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(cropped_face, caption="Model Input (48x48)", width=200)
            
        with col2:
            st.subheader("Analysis Results")
            
            # Find the max confidence
            max_conf = np.max(predictions)
            winners = [i for i, val in enumerate(predictions) if val == max_conf]
            
            # Display Top Prediction(s)
            st.write("### 🏆 Top Prediction")
            for idx in winners:
                st.success(f"**{CLASS_NAMES[idx].upper()}**: {max_conf*100:.2f}%")
            
            st.divider()
            
            # Display All Label Confidences
            st.write("### 📊 All Confidences")
            for i, label in enumerate(CLASS_NAMES):
                label_text = f"**{label.capitalize()}**" if i in winners else f"{label.capitalize()}"
                confidence_val = predictions[i] * 100
                st.write(f"{label_text}: {confidence_val:.2f}%")
                st.progress(min(int(confidence_val), 100))

st.markdown("---")
st.caption("Training Validation Accuracy: 91.21% | Dataset: FER2013 | Pre-processing: 48x48 Grayscale")