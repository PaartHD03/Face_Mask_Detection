#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('mask_detection_model.h5')

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use the default camera (index 0)

# Define the labels
labels = ['without_mask', 'with_mask']

# Function to preprocess the frame
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (100, 100))  # Resize frame to match input size of the model
    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    return normalized_frame.reshape(-1, 100, 100, 3)  # Reshape for model input

# Main loop to capture and process frames
while True:
    ret, frame = cap.read()  # Capture frame from the camera

    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Predict mask using the model
    prediction = model.predict(processed_frame)
    label = labels[np.argmax(prediction)]

    # Add label text to the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Mask Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:




