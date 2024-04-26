# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ObK6R-4UXt2TvhI7uPikIlo1drKkvHQn
"""

pip install streamlit

#importing libraries
import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf

from google.colab import drive
drive.mount('/content/drive')

#load the trained model
from keras.models import load_model
import os
dirname = os.path.dirname('/content/drive/MyDrive/HSAR/model')
model = load_model(os.path.join('/content/drive/MyDrive/HSAR/model', 'hsar.h5'))
#/content/drive/MyDrive/HSAR/Another copy of Human Activity Recognition Using LRCN Model.ipynb

import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Define constants
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["noFights", "fights"]
model_file_path = "/content/drive/MyDrive/HSAR/model/hsar.h5"
convlstm_model = load_model("/content/drive/MyDrive/HSAR/model/hsar.h5")

def perform_action_recognition_on_frame(frame, frames_queue):
    resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
    normalized_frame = resized_frame / 255
    frames_queue.append(normalized_frame)

    if len(frames_queue) == SEQUENCE_LENGTH:
        # Perform action recognition (replace this with your actual model prediction)
        # Dummy prediction for demonstration
        predicted_labels_probabilities = convlstm_model.predict(np.expand_dims(frames_queue, axis=0))[0]
        predicted_label = np.random.randint(len(CLASSES_LIST))
        predicted_class_name = CLASSES_LIST[predicted_label]

        # Draw predicted class name on frame
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Draw predicted class name on frame with black background box
        text_size = cv2.getTextSize(predicted_class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x, text_y = 10, 30  # Position of the text
        padding = 5  # Padding around the text
        box_coords = ((text_x, text_y + padding), (text_x + text_size[0] + padding * 2, text_y - text_size[1] - padding))

        # Draw the black background box
        cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 0), -1)

        # Draw the predicted class name on the frame
        cv2.putText(frame, predicted_class_name, (text_x + padding, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def main():
    st.title("Live Camera Detection")
    st.write("Click the button below to start the camera and perform activity detection.")

    if st.button("Predict from Camera"):
        cap = cv2.VideoCapture(0)  # Open the default camera (0) on your laptop
        frames_queue = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = perform_action_recognition_on_frame(frame, frames_queue)

            cv2.imshow('Human Activity Prediction', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

!streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py

! pip install streamlit-webrtc

!pip install streamlit-webrtc>=0.51.0

!streamlit run app.py & npx localtunnel --port 8501
