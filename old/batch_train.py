import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import joblib

# old code that just runs indidividual images to train each image at a time (DEPRIATED UNRELATED TO CURRENT PROJECT)
# Load dataset
df = pd.read_csv('dataset.csv')

# Count total images (rows)
print(f"Total images: {len(df)}")

image_size =len(df)

df = pd.read_csv('dataset.csv')

for i in range(image_size):
    print(f"Processing image index: {i}")
    # Get image_number from command line argument

    image_number = i

    # Load dataset

    # Prepare feature matrix X and labels Y
    X = df.drop(['label'], axis=1)
    Y = df['label']

    # Retrieve and display one image
    fig, axes = plt.subplots(1, 1, figsize=(3, 3))
    pixels = np.array(X.iloc[image_number]).reshape(28, 28)
    axes.imshow(pixels, cmap='gray')
    axes.set_title(f"Label: {Y.iloc[image_number]}")
    axes.axis('off')

    # Close the figure to free memory
    plt.close() # It closes each figure after creating it, avoiding memory buildup and removing the warning.

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
    # plt.show()