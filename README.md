# 🎭 Deep Facial Emotion Recognition (FER)
**Live Demo:** [[Live Demo]](https://deep-facial-emotion-recognation-project-3-hxrgzfm5wulxuxhs3xsh.streamlit.app/)  
**Validation Accuracy:** 91.21%  
**Developer:** Asit Atmaram More

## 📌 Project Overview
This project is an AI-powered application that detects human facial expressions in real-time. Using a **Deep Convolutional Neural Network (CNN)**, the system analyzes facial features to classify emotions into seven distinct categories. 

The application is optimized for production, featuring a dual-input system that handles both static image uploads and live webcam snapshots with high precision.



---

## 🚀 Key Features
* **Dual Input Mode:** Toggle between uploading local photos or using a live webcam capture.
* **Smart Face Detection:** Integrated **OpenCV Haar Cascades** to automatically crop and center the face, removing background noise for a 91%+ accuracy rate.
* **Full Confidence Breakdown:** Displays a real-time probability distribution for all 7 emotion labels.
* **Production-Ready Deployment:** Hosted on Streamlit Cloud with **Git LFS** integration for seamless large-model handling.

---

## 📊 Model & Dataset Details
* **Dataset:** Trained on the **FER2013** dataset ($48 \times 48$ grayscale images).
* **Architecture:** Custom CNN with Dropout and Batch Normalization layers to prevent overfitting.
* **Labels:** - `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

---

## 🛠️ Technical Stack
* **Language:** Python 3.x
* **Frameworks:** TensorFlow, Keras
* **Computer Vision:** OpenCV (cv2)
* **Web UI:** Streamlit
* **Libraries:** NumPy, Pillow (PIL), Scikit-Learn
* **Version Control:** Git LFS (Large File Storage)



---

## 📂 Project Structure
```text
├── app.py                # Main Streamlit application script
├── emotiondetector.h5    # Pre-trained CNN model (Managed via Git LFS)
├── requirements.txt      # Production dependencies
├── .gitattributes        # LFS configuration file
└── DeepFER-Final.ipynb   # Model training and research notebook



⚙️ Installation & Local Setup
Clone the Repo:

Bash

git clone [https://github.com/MADARA723/Deep-Facial-Emotion-Recognation-Project-3.git](https://github.com/MADARA723/Deep-Facial-Emotion-Recognation-Project-3.git)
Install Requirements:

Bash

pip install -r requirements.txt
Run App:

Bash

streamlit run app.py
📄 License
This project is open-source. Feel free to use the code for educational and research purposes.


---

### **How to add this to GitHub:**
1.  On your GitHub repository page, click **"Add file"** > **"Create new file"**.
2.  Name the file exactly **`README.md`**.
3.  Paste the code above into the editor.
4.  **Important:** Replace `[Insert Your Streamlit Link Here]` with your actual live URL.
5.  Click **"Commit changes"**.

**I'm ready when you are! Which project should we write the LinkedIn post for first?**
