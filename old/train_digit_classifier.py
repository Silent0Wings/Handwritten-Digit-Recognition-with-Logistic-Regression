import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import joblib

# Get image_number from command line argument
if len(sys.argv) < 2:
    print("Usage: python script.py <image_number>")
    sys.exit(1)

image_number = int(sys.argv[1])

# Load dataset
df = pd.read_csv('dataset.csv')

# Prepare feature matrix X and labels Y
X = df.drop(['label'], axis=1)
Y = df['label']

# Retrieve and display one image
fig, axes = plt.subplots(1, 1, figsize=(3, 3))
pixels = np.array(X.iloc[image_number]).reshape(28, 28)
axes.imshow(pixels, cmap='gray')
axes.set_title(f"Label: {Y.iloc[image_number]}")
axes.axis('off')

# Split dataset into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)

# Train SVM classifier
classifier = SVC(kernel="linear", random_state=6)
classifier.fit(train_x, train_y)

# Save trained model
os.makedirs('model', exist_ok=True)
joblib.dump(classifier, "model/digit_recognizer")

# Predict and calculate accuracy
prediction = classifier.predict(test_x)
print("Accuracy= ", metrics.accuracy_score(prediction, test_y))

# Show the selected image plot
plt.show()
