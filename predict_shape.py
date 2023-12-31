
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('shape_recognition_model.h5')

def predict_shape(image_path, confidence_threshold=0.6):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not open or find the image at {image_path}")

    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    predictions = model.predict(np.array([img]))
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]

    shapes = ['circle', 'square', 'triangle']

    # Check if prediction confidence meets the threshold
    if confidence < confidence_threshold:
        return "Uncertain"
    return shapes[predicted_index]

# Test the function
if __name__ == "__main__":
    image_path = input("Enter the path to the image: ")  # Make the image path configurable
    print("Predicted shape:", predict_shape(image_path))

