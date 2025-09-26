import streamlit as st
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import pickle

import joblib
classifier = joblib.load("classifier.pkl")

# --- Load Face Detector & FaceNet ---
mtcnn = MTCNN()
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# --- Load Classifier (مثلاً SVM) ---

# --- دالة استخراج embedding ---
def get_embedding(img):
    face = mtcnn(img)
    if face is None:
        return None
    face = face.unsqueeze(0)  # batch dimension
    embedding = facenet(face).detach().numpy()
    return embedding.reshape(1, -1)  # تأكيد الشكل (1,512)

# --- Streamlit UI ---
st.title("Face Recognition App")

choice = st.radio("Choose input method:", ("Upload Image", "Use Camera"))

image = None  # define upfront

if choice == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

elif choice == "Use Camera":
    img_file = st.camera_input("Take a picture")
    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Captured Image", use_column_width=True)

# --- Face Recognition ---
if image is not None:
    embedding = get_embedding(image)
    if embedding is not None:
        predicted_person = classifier.predict(embedding)
        st.success(f"✅ Person recognized: {predicted_person[0]}")
    else:
        st.warning("❌ No face detected in the image")
