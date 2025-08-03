# generate_dataset_binarize_then_resize.py

import cv2
import csv
import glob

header = ["label"]
for i in range(64):  # 8×8 = 64 pixels
    header.append("pixel" + str(i))

with open('dataset.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)

for label in range(10):
    dirList = glob.glob("captured_images/" + str(label) + "/*.png")

    for img_path in dirList:
        im = cv2.imread(img_path)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_blurred = cv2.GaussianBlur(im_gray, (1, 1), 0)


        # Binarize BEFORE resizing
        binarized = (im_blurred > 10).astype('uint8') * 255

        # Resize to 8×8 after binarization
        roi = cv2.resize(binarized, (8, 8), interpolation=cv2.INTER_AREA)

        data = [label]
        rows, cols = roi.shape

        for i in range(rows):
            for j in range(cols):
                k = roi[i, j]
                k = 1 if k > 10 else 0  # binarize again after resize (if needed)
                k = k * 16  # scale to match load_digits range
                data.append(k)

        with open('dataset.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)