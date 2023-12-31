
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Load data
data = np.load('data.npy')
labels = np.load('labels.npy')
labels = tf.keras.utils.to_categorical(labels, 3)

# Split data into train, validation, and test sets
train_images, test_images, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Model architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Model compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Model training with data augmentation
model.fit(datagen.flow(train_images, train_labels, batch_size=32),
          epochs=50,  # Increased epochs as early stopping will monitor and stop if necessary
          validation_data=(val_images, val_labels),
          callbacks=[early_stopping])

# Save the trained model
model.save('shape_recognition_model.h5')
