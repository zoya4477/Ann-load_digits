# integrated_app.py

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Function to train and save the model
def train_and_save_model():
    # Load dataset
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Preprocessing
    X = StandardScaler().fit_transform(X)
    y = to_categorical(y, num_classes=10)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build ANN model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(64,)),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # Save the model
    model.save('ann_digits_model.h5')

# Function to create Streamlit app
def create_streamlit_app():
    # Load pre-trained model
    model = load_model('ann_digits_model.h5')

    # Streamlit UI
    st.title("Digit Recognition Using ANN")
    st.write("Upload an 8x8 image of a digit (0-9)")

    uploaded_file = st.file_uploader("Choose an image...", type="png")

    if uploaded_file is not None:
        # Process the uploaded image
        image = Image.open(uploaded_file).convert('L')
        image = image.resize((8, 8), Image.ANTIALIAS)
        image_array = np.array(image)
        image_array = image_array.reshape(1, -1)  # Flatten the image
        image_array = image_array / 255.0  # Normalize the image

        # Predict using the model
        prediction = model.predict(image_array)
        predicted_digit = np.argmax(prediction)

        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write(f"Predicted Digit: {predicted_digit}")

# Main entry point
if __name__ == "__main__":
    if not os.path.exists('ann_digits_model.h5'):
        st.write("Model not found. Training the model...")
        train_and_save_model()
        st.write("Model trained and saved as 'ann_digits_model.h5'.")
    
    create_streamlit_app()

