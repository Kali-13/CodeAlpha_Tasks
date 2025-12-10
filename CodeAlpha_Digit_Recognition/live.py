import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2 

# ----------------------------
# Load your trained ANN model
# ----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# ----------------------------
# Preprocessing function
# ----------------------------
def preprocess(imgg):
    img = np.array(imgg)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    resized_img=cv2.resize(gray,(28,28))
    resized_img=tf.keras.utils.normalize(resized_img,axis=1)
    resized_img=np.array(resized_img).reshape(-1,28,28,1)
   

    return resized_img


st.title("🔢 Handwritten Digit Classifier")
st.write("Upload an image of a handwritten digit and the model will predict it.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Show uploaded image
    
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=200)

    # Preprocess
    input_img = preprocess(img)

    # Predict
    prediction = model.predict(input_img)
    predicted_class = np.argmax(prediction)

    st.subheader("🎯 Predicted Digit:")
    st.success(f"**{predicted_class}**")

    st.write("Raw model output:", prediction)

