import streamlit as st
import os

# 1. Dependency Safety Check
try:
    import tensorflow as tf
    import cv2
    import numpy as np
    from PIL import Image
except ImportError as e:
    st.error(f"Missing library: {e}. Check your requirements.txt!")
    st.stop()

# 2. Page Configuration
st.set_page_config(
    page_title="Emotion AI Detector",
    page_icon="🎭",
    layout="centered"
)

# 3. Model Loading Logic (Optimized for Streamlit Cloud)
@st.cache_resource
def load_emotion_model():
    # Make sure this filename matches exactly what is in your GitHub Repo
    model_path = 'emotiondetector2.h5' 
    
    if not os.path.exists(model_path):
        st.error(f"🚨 Model file '{model_path}' not found!")
        st.info("Tip: If you used Git LFS, ensure the file was pushed correctly and shows an 'LFS' tag on GitHub.")
        return None
    
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_emotion_model()

# Emotion Dictionary (Matching DeepFER-Final.ipynb)
emotion_labels = {
    0: 'Angry', 
    1: 'Disgust', 
    2: 'Fear', 
    3: 'Happy', 
    4: 'Neutral', 
    5: 'Sad', 
    6: 'Surprise'
}

# 4. UI Header
st.title("🎭 Deep Facial Emotion Recognition")
st.markdown("""
Predicting human emotions using a **Convolutional Neural Network (CNN)**.
This model was trained on the FER2013 dataset.
""")

# 5. Input Selection
input_mode = st.radio("Choose your source:", ("Upload Image", "Use Live Camera"))

if input_mode == "Upload Image":
    source = st.file_uploader("Upload a face photo", type=["jpg", "png", "jpeg"])
else:
    source = st.camera_input("Smile for the camera!")

# 6. Inference Pipeline
if source is not None and model is not None:
    # --- PREPROCESSING ---
    # Convert image to Grayscale (as required by DeepFER)
    raw_img = Image.open(source).convert('L')
    
    # Resize to 48x48 (Matching notebook input shape)
    img_resized = raw_img.resize((48, 48))
    img_array = np.array(img_resized)
    
    # Reshape for CNN: (Batch, Height, Width, Channels)
    # Normalized by 255.0 to match training scale
    processed_img = img_array.reshape(1, 48, 48, 1) / 255.0

    # --- PREDICTION ---
    with st.spinner("Analyzing micro-expressions..."):
        prediction = model.predict(processed_img)
        predicted_idx = np.argmax(prediction[0])
        label = emotion_labels[predicted_idx]
        confidence = np.max(prediction[0]) * 100

    # --- DISPLAY RESULTS ---
    st.divider()
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(raw_img, caption="Original Image", use_container_width=True)
    
    with col2:
        st.subheader("Result")
        st.markdown(f"**Detected Emotion:** `{label}`")
        st.progress(int(confidence))
        st.write(f"**Confidence:** {confidence:.2f}%")

    # Show Probability Chart
    st.write("### Emotion Probability Distribution")
    chart_data = {emotion_labels[i]: float(prediction[0][i]) for i in range(len(emotion_labels))}
    st.bar_chart(chart_data)