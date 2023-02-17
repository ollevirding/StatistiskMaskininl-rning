import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

np.random.seed(1)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.drop(columns=['Lead'])
y = train['Lead']
X_train,X_val,y_train,y_val = skl_ms.train_test_split(X,y,test_size=0.3)

parameters = [
    [5, 10, 20, 50, 100, 200, 500], #n_estimators (The number of trees in the forrest)
    ["gini", "entrophy", "log_loss"], #criterion (The function to measure the quality of a split)
    [5, 10, 50, 100, 200, None], #Max_depth (The maximum depth of the tree)
    [2, 4, 8, 16, 32, 64, 128], #min_samples_split (The minimum number of samples required to split an internal node)
    [1, 2, 4, 8, 16, 32, 64, 128], #min_samples_leaf (The minimum number of samples required to be at a leaf node)
    [0.0, 0.01, 0.1, 1.0, 10.0, 100.0], #min_weight_fraction_leaf (The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.)
    ["sqrt", "log2", None], #max_features (The number of features to consider when looking for the best split:)
    [10, 50, 100, 200, 500, 1000, None], #max_leaf_nodes (Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.)
    [0.0, 0.001, 0.01, 0.1, 0.5], #min_impurity_decrease (A node will be split if this split induces a decrease of the impurity greater than or equal to this value.)
    [True, False], #bootstrap (Whether bootstrap samples are used when building trees)
    [-1, None] #n_job (Process in use)
]


