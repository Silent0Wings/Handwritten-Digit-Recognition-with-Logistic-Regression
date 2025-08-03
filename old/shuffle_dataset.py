import pandas as pd
from sklearn.utils import shuffle


# think of it as shufling cards

# Load the completed dataset
data = pd.read_csv('dataset.csv')

# Shuffle the rows randomly
data = shuffle(data)

# Optionally save the shuffled dataset back
data.to_csv('dataset.csv', index=False)

print("Shuffling complete. Saved as 'shuffled_dataset.csv'")

# Mixes the rows randomly
# Stops pattern problems (like all 0’s grouped)
# Helps the model learn better
# Makes sure training is fair