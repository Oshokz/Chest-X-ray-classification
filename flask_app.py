# Import required packages
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, render_template, request, url_for

app = Flask(__name__)

# Project introduction
st.write("# AI-Powered Chest X-ray App")
st.write("###### This tool was developed for a final project in Artificial Intelligence at Plymouth University, using an advanced AI model with 97% accuracy to predict chest conditions.")


# load the model 
model_path = r"Chest_Xray_Model_Tf_2_15.h5"
try:
    model = load_model(model_path)
       st.success("Processing your image for predictions...")
    except Exception as e:
        st.error(f"Error loading the image: {e}")

# Create a file uploader
uploaded_file = st.file_uploader("Upload your chest X-ray image here...", type=["jpg", "jpeg", "png"])

def predict_image(image):
    image =image.convert('RGB')
    #Preprocess the image for prediction
    img = np.array(image)
    #resize the image
    img = tf.image.resize(img, (64, 64)) # since the size used to build the model is 64
    #normalize the image
    img = img / 255.0
    img = np.expand_dims (img, axis = 0)
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



@app.route("/predict/", methods =  ["GET", "POST"])
def predict():
    image = None
    label = None
    
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        
        #Check if image is uploaded
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            image = image.convert("RGB")
            image.save("static/uploaded_image.jpg")
            label = predict_image(image)

    return render_template("index_html", image = url_for("static", filename = "uploaded_image.jpg"), label = label)

if __name__ == "__main__":
    app.run(debug = True)

# create a static folder (empty) and templates folder containing index.html and result.html

# Add a small "October 2024" in the bottom right corner
st.markdown(
    """
    <div style="position: fixed;
                bottom: 10px;
                right: 10px;
                font-size: 20px;
                color: gray;">
        October 2024
    </div>
    """,
    unsafe_allow_html=True
)
