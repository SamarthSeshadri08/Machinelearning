# -*- coding: utf-8 -*-
"""MLlab4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yItM1u0CXyyfc0Y-HecCV5dlj5bEBq_O
"""

import pandas as pd

# initialize list of lists
data = [['<=30', 'high', 'no', 'fair', 'no'], ['<=30', 'high', 'no', 'excellent', 'no'],
        ['31..40', 'high', 'no', 'fair', 'yes'], ['>40', 'medium', 'no', 'fair', 'yes'],
        ['>40', 'low', 'yes', 'fair', 'yes'], ['>40', 'low', 'yes', 'excellent', 'no'],
        ['31..40', 'low', 'yes', 'excellent', 'yes'], ['<=30', 'medium', 'no', 'fair', 'no'],
        ['<=30', 'low', 'yes', 'fair', 'yes'], ['>40', 'medium', 'yes', 'fair', 'yes'],
        ['<=30', 'medium', 'yes', 'excellent', 'yes'], ['31..40', 'medium', 'no', 'excellent', 'yes'],
        ['31..40', 'high', 'yes', 'fair', 'yes'], ['>40', 'medium', 'no', 'excellent', 'no']]

# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['age', 'income','student','credit_rating','buys_computer'])

# print dataframe.
df

df['buys_computer']

df.shape

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Fit the label encoder to your data
#df['age_encoded'] = label_encoder.fit_transform(df['age'])
#label_encoder.fit(df)

# Transform the data
#encoded_data = label_encoder.transform(df)

# Print the encoded data
for x in df.columns:
  df[x]=label_encoder.fit_transform(df[x])
print(df)

import math
from collections import Counter

# Function to calculate entropy
def entropy(data):
    n = len(data)
    label_counts = Counter(data)
    entropy = 0
    for label in label_counts:
        prob = label_counts[label] / n
        entropy -= prob * math.log2(prob)
    return entropy

# Calculate entropy of target variable (buys_computer)
target_entropy = entropy(df['buys_computer'])

# Define a function to calculate conditional entropy
def conditional_entropy(feature, target):
    feature_value_counts = Counter(feature)
    conditional_entropy = 0
    for value, count in feature_value_counts.items():
        prob = count / len(feature)
        subset_target = [target[i] for i in range(len(target)) if feature[i] == value]
        conditional_entropy += prob * entropy(subset_target)
    return conditional_entropy

# Calculate information gain for each feature
age_gain = target_entropy - conditional_entropy(df['age'], df['buys_computer'])
income_gain = target_entropy - conditional_entropy(df['income'], df['buys_computer'])
student_gain = target_entropy - conditional_entropy(df['student'], df['buys_computer'])
credit_rating_gain = target_entropy - conditional_entropy(df['credit_rating'], df['buys_computer'])

# Print information gains
print(f'Age Information Gain: {age_gain}')
print(f'Income Information Gain: {income_gain}')
print(f'Student Information Gain: {student_gain}')
print(f'Credit Rating Information Gain: {credit_rating_gain}')

#the attribute with the highest information gain will be considered as the first attribute for decision tree
#Age
X=df.iloc[:,:]
X=X.drop(['buys_computer'],axis=1)
print(X.shape)
y=df['buys_computer']
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier

# Assuming Tr_X and Tr_y are your training features and labels
# You need to convert categorical features to numerical using label encoding

# Create an instance of the DecisionTreeClassifier
model = DecisionTreeClassifier()

# Fit the model on the training data
model.fit(X_train, y_train)

# Calculate training set accuracy
train_accuracy = model.score(X_train, y_train)
print(f"Training Set Accuracy: {train_accuracy}")

# Get the depth of the tree
tree_depth = model.get_depth()
print(f"Tree Depth: {tree_depth}")

import matplotlib.pyplot as plt
from sklearn import tree
plt.figure(figsize=(70,20))
tree.plot_tree(model, filled=True)
plt.show()