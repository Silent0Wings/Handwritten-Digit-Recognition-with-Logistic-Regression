import pandas as pd  # import pandas library for data handling


#this code simply check wether there is a null row a type of corruption commun in csv files


# Load dataset from CSV file into a DataFrame (table-like structure)
df = pd.read_csv('dataset.csv')

# Check rows if ANY column has a missing (NaN) value
# df.isnull() -> DataFrame of True/False per cell (True if NaN)
# .any(axis=1) -> True if any cell in the row is True (missing)
# df[...] -> filters and keeps only rows where condition is True
corrupted_rows = df[df.isnull().any(axis=1)]

# Print how many corrupted rows were found
print(f"Number of corrupted rows: {len(corrupted_rows)}")

# If there are any corrupted rows, print their full content
if not corrupted_rows.empty:
    print(corrupted_rows)
