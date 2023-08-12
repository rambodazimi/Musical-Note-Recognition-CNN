"""
Autor: Rambod Azimi

This python script will create a CNN neural network model for classification of musical notes

There are five different music symbols considered for the classification. All categories have 1000 numbers of data. 800 For training set and 200 for validation set.

1. Whole Note
2. Half Note
3. Quarter Note
4. Eight Note
5. Sixteenth Note

I have used this dataset from kaggle:
https://www.kaggle.com/datasets/kishanj/music-notes-datasets?resource=download

This project has been exclusively written for the lab of professor Eunice Jun
"""

import tensorflow as tf
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

"""
I decided to split the dataset into 2 parts:
1. Training set (80%)
2. Validation set (20%)

Each training example is a jpg image (64x64 pixel)
"""

# Construct the new path by concatenating the script directory and the relative path
train_data_dir = os.path.join(script_directory, 'Dataset')
validation_data_dir = os.path.join(script_directory, 'Validation')

new_images_dir = os.path.join(script_directory, 'Test')

# Data augmentation and preprocessing
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255
)

# Load and augment training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Load validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Define your CNN architecture
model = tf.keras.Sequential([
    # Add convolutional layers, pooling layers, and dense layers
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # Replace num_classes with the number of classes in your dataset
])

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Evaluate the model
validation_loss, validation_accuracy = model.evaluate(validation_generator, steps=len(validation_generator))
print("Validation accuracy:", validation_accuracy)

# Make predictions on new images
new_images_generator = validation_datagen.flow_from_directory(
    new_images_dir,
    target_size=(64, 64),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Class names
class_names = ['Eight Note', 'Half Note', 'Quarter Note', 'Sixteenth Note', 'Whole Note']

def load_and_predict_image():
    file_path = filedialog.askopenfilename()
    
    if file_path:
        image = Image.open(file_path).resize((64, 64))
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        prediction = model.predict(image)
        predicted_class = class_names[np.argmax(prediction)]
        
        result_label.config(text=f"Predicted Class: {predicted_class}")

        if(predicted_class == 'Whole Note'):
            description_label.config(text=f"A whole note is a basic musical note that represents a sound lasting for a specific duration. It has an open oval shape and is not filled. The whole note is often used to indicate a sustained tone or a long beat.")

        if(predicted_class == 'Half Note'):
            description_label.config(text=f"A half note is a musical note with an oval shape that is filled. It signifies a sound that lasts for half the duration of a whole note. In musical notation, it is often used to represent a note with a medium duration.")

        if(predicted_class == 'Quarter Note'):
            description_label.config(text=f"A quarter note is a fundamental musical note with a solid oval shape. It signifies a sound lasting for a quarter of the duration of a whole note. In sheet music, it is frequently used to convey rhythmic patterns.")

        if(predicted_class == 'Eight Note'):
            description_label.config(text=f"An eighth note is a musical note with a solid oval shape and a single flag attached to the stem. It indicates a sound lasting for an eighth of the duration of a whole note. Eighth notes are used to convey faster rhythms.")

        if(predicted_class == 'Sixteenth Note'):
            description_label.config(text=f"A sixteenth note is a musical note with a solid oval shape and two flags attached to the stem. It represents a sound lasting for a sixteenth of the duration of a whole note. Sixteenth notes are used to convey rapid and intricate rhythms.")

        img = Image.open(file_path)
        img.thumbnail((250, 250))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

# Create the main window
root = tk.Tk()
root.title("Music Note Classifier")

# Create widgets
browse_button = tk.Button(root, text="Browse Image", command=load_and_predict_image)
result_label = tk.Label(root, text="", font=("Helvetica", 16))
description_label = tk.Label(root, text="", font=("Helvetica", 12))

image_label = tk.Label(root)

# Place widgets on the window
browse_button.pack(pady=10)
result_label.pack(pady=10)
description_label.pack(pady=10)
image_label.pack()

# Run the GUI event loop
root.mainloop()

