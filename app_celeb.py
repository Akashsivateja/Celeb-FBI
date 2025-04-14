import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Title of the App
st.title("👤 Age, Gender, Height, Weight Predictor")
st.markdown("Upload an image to predict **Age**, **Gender**, **Height**, and **Weight**")

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("celeb_fbi_model.h5")
    return model

model = load_model()

# Prediction function
def predict(image):
    img = image.resize((128, 128))  # Resize to match model input
    img_array = np.array(img)

    if img_array.shape[-1] == 4:  # If PNG with alpha channel
        img_array = img_array[:, :, :3]

    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    gender, height, weight, age = model.predict(img_array)

    # Process results
    gender_label = "Male" if gender[0][0] < 0.5 else "Female"
    height_cm = round(height[0][0], 2)
    weight_kg = round(weight[0][0], 2)
    age_yrs = round(age[0][0], 2)

    return gender_label, height_cm, weight_kg, age_yrs

# Upload UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        gender, height, weight, age = predict(image)
        st.success(f"**Gender:** {gender}")
        st.success(f"**Height:** {height} cm")
        st.success(f"**Weight:** {weight} kg")
        st.success(f"**Age:** {age} years")
