#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:17:04 2022

@author: lorenzomeloncelli
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_excel('Unified Survey.xlsx')
X = dataset.iloc[:, [20,21,23,24,25,26,29,42,45]].values
X = pd.DataFrame(X)
def gnumeric_func (data):
  data = data.apply(lambda x: pd.factorize(x)[0])
  return data
X_factor = gnumeric_func(X)
y = dataset.iloc[:, 7].values
y = list(y)
independent = []
for i in y:
    if i == 4:
        independent.append(1)
    else:
        independent.append(0)
 
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_factor, independent, test_size = 0.25, random_state = 0)

# Feature Scaling
# The algorithm should not be biased towards variables with higher magnitude. To overcome this problem, we can bring down all the variables to the same scale.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

