#Author: Cameron O'Brien
#Date Started: 09/26/2018

#Importing packages
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#loading dataset from csv file
data = pd.read_csv('creditcard.csv')

#going through dataset
print(data.columns)
print(data.shape)
print(data.describe())

# plot histogram of parameter
data.hist(figsize = (20,20))
plt.show()

#Determine number of fraud transactions in dataset
fraud = data[data['Class']==1]
valid = data[data['Class']==0]
outlier_fraction = len(fraud) / float(len(valid))
print(outlier_fraction)
print('Fraud Cases: {}'.format(len(fraud)))
print('Valid Cases: {}'.format(len(valid)))

#Create a correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

#get all columns of data frame
columns = data.columns.tolist()

#filter columns to remove data not wanted
columns = [c for c in columns if c not in ["Class"]]

#store the variable we'll be predicting on
target = "Class"
x = data[columns]
y = data[target]

#print the shapes of x and y
print(x.shape)
print(y.shape)

# define a random state
state = 1

#define the outlier detection methods
classifiers = {
    "IsolationForest": IsolationForest(max_samples = len(x),
        contamination = outlier_fraction, random_state = state),
     "Local Outlier Factor": LocalOutlierFactor(n_neighbors = 20, contamination = outlier_fraction)
}

#fit the model
n_outliers = len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    # fit the data and log outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(x)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(x)
        scores_pred = clf.decision_function(x)
        y_pred = clf.predict(x)

    # Reshape the prediction values to 0 for Valid, 1 for Fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    n_errors = (y_pred != y).sum()

    #Run classification matrics
    print('{}: {}:'.format(clf_name, n_errors))
    print(accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))
