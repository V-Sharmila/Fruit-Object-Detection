# Fruit Image Classification with TensorFlow
This repository contains code for classifying fruit images using a Convolutional Neural Network (CNN) with TensorFlow and Keras. The model is built on the ResNet50 architecture and trained to classify images into four categories: background, orange, apple, and banana.

Project Overview
Objective: To classify fruit images into predefined categories using deep learning.
Architecture: ResNet50 pre-trained model with a custom classifier on top.
Dataset: The dataset includes images and corresponding XML files with bounding boxes and labels.
Prerequisites
Make sure you have Python installed and the necessary packages. You can install the required packages using pip:


pip install tensorflow opencv-python-headless matplotlib

# Directory Structure

train_root: Directory containing training images and XML annotations.
val_root: Directory containing validation images and XML annotations.
# Code Description
Imports: The code imports necessary libraries, including TensorFlow, Keras, OpenCV, and others.
Data Preprocessing: Functions to preprocess images and load the dataset.
Model Definition: Defines a CNN model using ResNet50 as the base and custom dense layers on top.
Training: Compiles and trains the model on the dataset.

# Usage
Prepare Dataset: Ensure your dataset is organized as described in the train_root and val_root directories.
Run Training Script: Execute the provided script to start training the model. The script will handle preprocessing, model creation, and training.
python

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import os
from xml.etree import ElementTree as et
from sklearn.model_selection import train_test_split

# Define your root directories
train_root = '/kaggle/input/fruit-images-for-object-detection/train_zip/train'
val_root = '/kaggle/input/fruit-images-for-object-detection/test_zip/test'

# Define your labels
labels = ['background', 'orange', 'apple', 'banana']
label2targets = {l: t for t, l in enumerate(labels)}
num_classes = len(labels)

def preprocess_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))  # Resize the image to a fixed size
    img = np.array(img)
    return img

def load_dataset(root):
    images = []
    targets = []
    xml_paths = sorted([os.path.join(root, file) for file in os.listdir(root) if file.endswith('.xml')])
    for xml_path in xml_paths:
        img_path = xml_path.replace('.xml', '.jpg')
        img = preprocess_img(img_path)
        tree = et.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            label = obj.find('name').text
            label_id = label2targets.get(label, 0)  # Assign 0 (background) if label not found
            box = obj.find('bndbox')
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            targets.append([xmin, ymin, xmax, ymax, label_id])  # Store targets as a list
            images.append(img)
    return np.array(images), np.array(targets)

# Load the dataset
X_train, y_train = load_dataset(train_root)
X_val, y_val = load_dataset(val_root)

def get_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes)(x)
    model = keras.Model(inputs=base_model.input, outputs=outputs)
    return model

# Compile your model
model = get_model(num_classes)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train your model
model.fit(X_train, y_train[:, -1], epochs=30, batch_size=32, validation_data=(X_val, y_val[:, -1]))
Results
The model will train for 30 epochs and output accuracy and loss metrics for both training and validation data.
