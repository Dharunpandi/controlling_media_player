import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

def load_pretrained_model(weights_path):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(120, 120, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),  # Updated Dense layer with 256 units
        tf.keras.layers.Dense(7, activation='softmax')  # Output layer with 7 classes
    ])
    
    model.load_weights(weights_path)
    return model

def preprocess_image(image):
    image = cv2.resize(image, (120, 120))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    image = image.reshape(120, 120, 1)
    image = image / 255.0 
    return np.expand_dims(image, axis=0)

model = load_pretrained_model('gesture-model.h5')

st.title("Hand Gesture Recognition from Video Feed")
run = st.checkbox('Run Video Feed')

frame_placeholder = st.empty()

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture image from camera.")
        break


    processed_image = preprocess_image(frame)

    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    frame_placeholder.image(frame, channels="RGB", caption=f"Predicted Gesture Class: {predicted_class[0]}", use_column_width=True)

    time.sleep(0.1) 

cap.release()
