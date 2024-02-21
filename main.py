import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Loading the data
data = pd.read_csv("cancer.csv")

# Viewing the data
# print(data.head()) # retrieving the first five rows of the dataframe
# print(data.info()) # succinct summary
print(data.describe())

# Cleaning the data
# sns.heatmap(data.isnull()) # visualizing missing data
# plt.show()
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
# print(data.head())

# convert 'M' and 'B' to 1 and 0 respectively
data.diagnosis = [1 if value == "M" else 0 for value in data.diagnosis]
# print(data.head())
# to convert integers into a category
data['diagnosis'] = data['diagnosis'].astype("category", copy=False)
data['diagnosis'].value_counts()
# sns.countplot(x='diagnosis', data=data, palette='hls')
# plt.show()

# Dividing the data into features and target variables
y = data["diagnosis"] # our target variables
X = data.drop(["diagnosis"], axis=1)
# print(y)
# print(X)

#plot logistic regression curve
sns.regplot(x=X["radius_mean"], y=y, data=data, logistic=True, ci=None, scatter_kws={'color': 'purple'}, line_kws={'color': 'black'})
plt.show()


# Scaling
scaler = StandardScaler() # create a scaler object

# fit the scale to the data and transform the data
X_scaled = scaler.fit_transform(X)
# print(X)
# print(X_scaled)

# Splitting the data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Training the data
# creating the logistic regression model
lr = LogisticRegression()

# training the model on the training data
lr.fit(X_train, y_train)

# predicting the target variables on test data
y_pred = lr.predict(X_test)
print(y_pred)
print(y_test)

# Evaluation of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .2f}")
print(classification_report(y_test, y_pred))

matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
print('Confusion matrix : \n', matrix)
