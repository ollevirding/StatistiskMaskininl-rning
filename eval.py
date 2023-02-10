#!/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb

from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from kfold import kfold



# Logistic regression
print("Logistic", kfold(skl_lm.LogisticRegression, solver='liblinear')[1])

# LDA
print("LDA", kfold(skl_da.LinearDiscriminantAnalysis)[1])



# QDA 
print("QDA", kfold(skl_da.QuadraticDiscriminantAnalysis)[1])

# kNN
print("KNN", kfold(skl_nb.KNeighborsClassifier,n_neighbors=30)[1])


# Tree based method
print("Tree", kfold(tree.DecisionTreeClassifier,max_depth = 7)[1])

# Bagging
print("Bagging", kfold(BaggingClassifier)[1])

# RandomForestClassifier
print("Random Forest", kfold(RandomForestClassifier)[1])

# AdaBoost
print("AdaBoost", kfold(AdaBoostClassifier)[1])

# GradientBoosting
print("Gradient Boosting", kfold(GradientBoostingClassifier)[1])
