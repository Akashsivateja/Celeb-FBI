import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Title of the App
st.title("👤 Age, Gender, Height, Weight Predictor")
st.markdown("Upload an image to predict **Age**, **Gender**, **Height**, and **Weight**")

# Load the trained model
@st.cache_resource  # This will cache the model so that it doesn't reload every time
def load_model():
    return tf.keras.models.load_model("celeb_fbi_model.h5")  # Load the saved model

model = load_model()  # Load the model once at the start

# Image Preprocessing
def predict(image):
    img = image.resize((128, 128))  # Resize to 128x128
    img_array = np.array(img)  # Convert image to numpy array
    if img_array.shape[-1] == 4:  # Handle PNG images with an alpha channel
        img_array = img_array[:, :, :3]  # Remove the alpha channel, keep RGB
    img_array = img_array / 255.0  # Normalize pixel values to the range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (e.g., shape becomes (1, 128, 128, 3))

    # Debugging line: check the shape of the image being passed to the model
    st.write(f"Image shape after processing: {img_array.shape}")

    # Make prediction
    gender, height, weight, age = model.predict(img_array)

    # Debugging: print predictions
    st.write(f"Predicted Gender: {gender}")
    st.write(f"Predicted Height: {height}")
    st.write(f"Predicted Weight: {weight}")
    st.write(f"Predicted Age: {age}")

    gender_label = "Male" if gender[0][0] < 0.5 else "Female"
    height_cm = round(height[0][0], 2)
    weight_kg = round(weight[0][0], 2)
    age_yrs = round(age[0][0], 2)

    return gender_label, height_cm, weight_kg, age_yrs


# Upload UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        gender, height, weight, age = predict(image)
        st.success(f"**Gender:** {gender}")
        st.success(f"**Height:** {height} cm")
        st.success(f"**Weight:** {weight} kg")
        st.success(f"**Age:** {age} years")
