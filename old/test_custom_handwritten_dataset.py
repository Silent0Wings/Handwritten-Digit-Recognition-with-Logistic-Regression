# test_custom_handwritten_dataset.py

# pip install opencv-python pandas scikit-learn seaborn matplotlib


# this code simply allows me to test a different dataset as a test 
# dataset and another unrelated dataset as training in this case a 
# self generated data set made of my own handwritting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plot_utils

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# === Step 1: Train on load_digits ===
digits = load_digits()
X = digits.data / 16.0  # Normalize to 0–1
y = digits.target

model = LogisticRegression(max_iter=10000)
model.fit(X, y)

# === Step 2: Load your custom handwritten dataset (CSV) ===
custom_data = pd.read_csv('dataset.csv')

# Separate labels and pixel data
labels = custom_data['label'].values
pixel_data = custom_data.drop('label', axis=1).values

# === Step 3: Normalize pixel data (your dataset uses 0 or 16) ===
pixel_data = pixel_data / 16.0

# === Step 4: Predict on your handwritten dataset ===
predictions = model.predict(pixel_data)

# === Step 5: Evaluate results ===
print("Custom Dataset Accuracy:", accuracy_score(labels, predictions))
print(classification_report(labels, predictions))
print(confusion_matrix(labels, predictions))

# === Optional: Visual plots ===
plot_utils.plot_confusion_matrix(labels, predictions, labels=digits.target_names)
plot_utils.plot_classification_report(labels, predictions)
