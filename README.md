# Handwritten-Digit-Recognition-with-Logistic-Regression

## Project Overview

This project does handwritten digit classification using logistic regression. It compare how good the model works on two datasets:
- `load_digits` from scikit-learn (8x8 grayscale digits)
- `MNIST` from Kaggle (28x28 grayscale digits)

The images are normalized and exported to csv files. Then they are re-converted into images and saved in folders by number (0 to 9) for easy viewing and understanding.

---

## Objective

To see how well logistic regression works on simple image recognition tasks. It look at:
- How accurate the model is
- What effects tuning parameters have
- How dataset size change performance
- Visualizing the results clearly

---

## Methodology

- Normalized pixel data
- Split the data (80% for training, 20% for testing)
- Train logistic regression with good hyperparameters
- Evaluate the results using:
  - Accuracy score
  - Precision, recall and F1
  - Confusion matrix
  - Sample digit images and plots

---

## Parameters Tuned

Some parameters we changed to see the effect:
- `max_iter`: number of iterations for training (128 was good for scikit-learn, 1000 for mnist)
- `solver`: solver type like `lbfgs`
- `C`: regularization (smaller = stronger regularization)
- `class_weight`: used or not used depending on balance
- Pixel normalization: like dividing by 9 or 16

---

## Visualizations

- Sample images from each digit class (0–9)
- Images before and after normalization
- Bar graph showing how many of each digit
- Line plots for accuracy vs:
  - test size
  - normalization value
  - max iterations
  - regularization value
- Heatmaps of classification report and confusion matrix

---

## Key Findings

- On `load_digits` dataset we got 99.2% accuracy
- On MNIST we got around 92.2%
- Changing normalization and number of iterations helped
- More data makes it harder for the model unless tuned properly

---


## License

This project is under MIT license.
