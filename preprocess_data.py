
import cv2
import numpy as np
import os

BASE_PATH = 'data'  # Base path where the shape directories are located

def load_data(base_path=BASE_PATH):
    shapes = ['circle', 'square', 'triangle']
    data = []
    labels = []

    for idx, shape in enumerate(shapes):
        shape_dir = os.path.join(base_path, shape)
        if not os.path.exists(shape_dir):
            print(f"Directory {shape_dir} not found!")
            continue

        for filename in os.listdir(shape_dir):
            img_path = os.path.join(shape_dir, filename)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Error reading image: {img_path}")
                    continue

                if img.shape[:2] != (64, 64):
                    img = cv2.resize(img, (64, 64))
                img = img / 255.0
                data.append(img)
                labels.append(idx)
            except Exception as e:
                print(f"Error processing image {img_path}. Error: {e}")

    return np.array(data), np.array(labels)

data, labels = load_data()
np.save('data.npy', data)
np.save('labels.npy', labels)
