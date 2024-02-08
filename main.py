import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Loading the data
data = pd.read_csv("cancer.csv")

# Viewing the data
# print(data.head()) # retrieving the first five rows of the dataframe
# print(data.info()) # succinct summary
# print(data.describe())

# Cleaning the data
sns.heatmap(data.isnull()) # visualizing missing data
# plt.show()
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
# print(data.head())

# convert 'M' and 'B' to 1 and 0 respectively
data.diagnosis = [1 if value == "M" else 0 for value in data.diagnosis]
# print(data.head())
# to convert integers into a category
data['diagnosis'] = data['diagnosis'].astype("category", copy=False)
data['diagnosis'].value_counts().plot(kind="bar")
# plt.show()

# dividing the data into features and target variables
y = data["diagnosis"] # our target variables
X = data.drop(["diagnosis"], axis=1)
# print(y)
# print(X)

# Scaling/Normalization
# create a scaler object
scaler = StandardScaler()

# fit the scale to the data and transform the data
X_scaled = scaler.fit_transform(X)
# print(X)
# print(X_scaled)
