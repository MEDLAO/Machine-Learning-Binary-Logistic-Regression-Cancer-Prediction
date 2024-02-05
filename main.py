import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Loading the data
data = pd.read_csv("cancer.csv")

# Viewing the data
print(data.head()) # retrieving the first five rows of the dataframe
print(data.info()) # succinct summary
print(data.describe())

# Cleaning the data
sns.heatmap(data.isnull()) # visualizing missing data
# plt.show()
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
print(data.head())

# convert 'M' and 'B' to 1 and 0 respectively
data.diagnosis = [1 if value == "M" else 0 for value in data.diagnosis]
print(data.head())
