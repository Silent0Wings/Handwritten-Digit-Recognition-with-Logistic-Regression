import pandas as pd
import numpy as np
import os
import cv2

# this code will simply naviguate a csv file and extract the images and put them in a organized folder

# Load dataset
df = pd.read_csv('digits_dataset.csv')

# Create main output folder
output_dir = 'processed_images'
os.makedirs(output_dir, exist_ok=True)

# Iterate over all rows
for idx, row in df.iterrows():
    label = row['label']
    pixels = np.array(row[1:]).reshape(8, 8).astype(np.uint8)

    # Scale binary (0/1) to 0–255 grayscale
    pixels = pixels * 255

    # Create subfolder for label if not exists
    label_folder = os.path.join(output_dir, str(label))
    os.makedirs(label_folder, exist_ok=True)

    # Build image filename
    filename = os.path.join(label_folder, f'image_{idx}.png')

    # Save image using OpenCV
    cv2.imwrite(filename, pixels)

print("All images processed and saved to folders!")
