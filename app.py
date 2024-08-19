# Import required packages
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


# Project introduction
st.write("# AI-Powered Chest X-ray App")
st.write("###### This tool was developed for a final project in Artificial Intelligence at Plymouth University, using an advanced AI model with 97% accuracy to predict chest conditions.")
st.write("### Date: August 2024")  # At the top, part of the introduction

# Create a file uploader
uploaded_file = st.file_uploader("Upload your chest X-ray image here...", type=["jpg", "jpeg", "png"])

# Check if the image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    st.write("")

    # Preprocess the image
    img = np.array(image)

    # Ensure image has 3 dimensions (height, width, channels)
    if img.ndim == 2:  # Grayscale image (height, width)
        img = np.stack((img,)*3, axis=-1)  # Convert to RGB by stacking
    elif img.ndim == 4:  # Exceeds expected dimensions (batch, height, width, channels)
        img = img[0]  # Use the first image in the batch

    # Resize the image using TensorFlow
    img = tf.image.resize(tf.convert_to_tensor(img), (64, 64))
    # Normalize the image
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Change the batch dimension to (1, 64, 64, 3)

    # Load the trained model
    model_path = r"Chest_Xray_Model_Tf_2_15.h5"

    try:
        model = load_model(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading the model: {e}")

    # Make predictions
    prediction = model.predict(img)

    # Determine the label based on the highest probability
    class_index = np.argmax(prediction)
    if class_index == 0:
        label = "Covid-19"
    elif class_index == 1:
        label = "Normal"
    elif class_index == 2:
        label = "Viral Pneumonia"
    else:
        label = "Tuberculosis"

    # Display the prediction
    st.write(f"## Predicted Chest Condition: {label}")

    # Provide advice based on the prediction
    if label == "Covid-19":
        st.write("### Advice:")
        st.write("- **Seek medical help immediately.**")
        st.write("- Isolate yourself to prevent spreading the virus to others.")
        st.write("- Follow local health guidelines and inform your close contacts.")
    elif label == "Normal":
        st.write("### Advice:")
        st.write("- **Your chest X-ray appears normal.**")
        st.write("- Continue to maintain a healthy lifestyle and follow any preventive measures to stay healthy.")
        st.write("- If you have symptoms, consult a healthcare provider for further advice.")
    elif label == "Viral Pneumonia":
        st.write("### Advice:")
        st.write("- **Consult with a healthcare provider promptly.**")
        st.write("- Follow the treatment plan prescribed by your doctor.")
        st.write("- Rest and stay hydrated to support your recovery.")
    elif label == "Tuberculosis":
        st.write("### Advice:")
        st.write("- **Seek immediate medical attention.**")
        st.write("- Follow the treatment regimen strictly as prescribed by your healthcare provider.")
        st.write("- Inform your close contacts as tuberculosis can be contagious.")


