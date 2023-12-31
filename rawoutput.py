import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('shape_recognition_model.h5')

# Load and preprocess the image
img = cv2.imread('circle.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (64, 64))
img_normalized = img_resized / 255.0

# Predict using the model
predictions = model.predict(np.array([img_normalized]))
predicted_probabilities = predictions[0]

print(predicted_probabilities)
