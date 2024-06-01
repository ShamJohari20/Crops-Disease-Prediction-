import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import json
import io

# Load the pre-trained model
model = load_model('model_VGG16.h5')

# Load class indices from JSON file
with open('VGG_CPD_class_indices.json', 'r') as f:
    class_indices = json.load(f)


def predict_disease(image):
    # Preprocess the image
    image = image.resize((224, 224))  #  VGG16 input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Perform prediction
    prediction = model.predict(image)

    # Get the predicted class
    predicted_class = class_indices[str(np.argmax(prediction))]

    return predicted_class


# Streamlit UI
st.title('Crops Disease Prediction')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(io.BytesIO(uploaded_file.read()))

    # Resize the image to a smaller size for display
    resized_image = image.resize((300, 200))

    st.image(resized_image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        try:
            prediction = predict_disease(image)
            st.success(f"Result: {prediction}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
