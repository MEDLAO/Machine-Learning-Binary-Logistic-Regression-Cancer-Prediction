import pandas as pd
import seaborn as sns

# Loading the data
data = pd.read_csv("cancer.csv")

# Viewing the data
print(data.head()) # retrieving the first five rows of the dataframe
print(data.info()) # succinct summary
print(data.describe())

# Cleaning the data
