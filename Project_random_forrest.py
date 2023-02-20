import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

np.random.seed(1)


train = pd.read_csv('train.csv')

X = train.drop(columns=['Lead'])


y = train['Lead']
X_train, X_val, y_train, y_val = skl_ms.train_test_split(X, y, test_size=0.3)

parameters = [
    [350],  # n_estimators (The number of trees in the forrest)
    ["log_loss"],  # criterion (The function to measure the quality of a split)
    [60],  # Max_depth (The maximum depth of the tree)
    # min_samples_split (The minimum number of samples required to split an internal node)
    [7],
    # min_samples_leaf (The minimum number of samples required to be at a leaf node)
    [1],
    # min_weight_fraction_leaf (The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.)
    [0.0],
    # max_features (The number of features to consider when looking for the best split:)
    [None],
    [420],  # max_leaf_nodes (Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.)
    # min_impurity_decrease (A node will be split if this split induces a decrease of the impurity greater than or equal to this value.)
    [0.0],
    # bootstrap (Whether bootstrap samples are used when building trees)
    [True],
]

"""parameters = [
    range(280,400,20), #n_estimators (The number of trees in the forrest)
    ["gini", "entropy", "log_loss"], #criterion (The function to measure the quality of a split)
    [5, 10, 50, 100, 200, None], #Max_depth (The maximum depth of the tree)
    [2, 4, 8, 16, 32, 64, 128], #min_samples_split (The minimum number of samples required to split an internal node)
    [1, 2, 4, 8, 16, 32, 64, 128], #min_samples_leaf (The minimum number of samples required to be at a leaf node)
    [0.0, 0.01, 0.1, 1.0, 10.0, 100.0], #min_weight_fraction_leaf (The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.)
    ["sqrt", "log2", None], #max_features (The number of features to consider when looking for the best split:)
    [10, 50, 100, 200, 500, 1000, None], #max_leaf_nodes (Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.)
    [0.0, 0.001, 0.01, 0.1, 0.5], #min_impurity_decrease (A node will be split if this split induces a decrease of the impurity greater than or equal to this value.)
    [True, False], #bootstrap (Whether bootstrap samples are used when building trees)
    [-1, None] #n_job (Process in use)
 ]"""

"""
#Define Model
model = RandomForestClassifier()

#Define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=0)

#Define space
space = dict()

space['n_estimators'] = parameters[0]
space['criterion'] = parameters[1]
space['max_depth'] = parameters[2]
space['min_samples_split'] = parameters[3]
space['min_samples_leaf'] = parameters[4]
space['min_weight_fraction_leaf'] = parameters[5]
space['max_features'] = parameters[6]
space['max_leaf_nodes'] = parameters[7]
space['min_impurity_decrease'] = parameters[8]
space['bootstrap'] = parameters[9]


search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)

result = search.fit(X, y)


summarize result
print('Best Score: %s\n' % result.best_score_)
print('Best Hyperparameters:')
for item, value in result.best_params_.items():
    print(f"{item}: \t\t{value}")

 """

model = RandomForestClassifier(
    n_estimators                =   parameters[0][0],
    criterion                   =   parameters[1][0],
    max_depth                   =   parameters[2][0],
    min_samples_split           =   parameters[3][0],
    min_samples_leaf            =   parameters[4][0],
    min_weight_fraction_leaf    =   parameters[5][0],
    max_leaf_nodes              =   parameters[7][0],
    bootstrap                   =   parameters[9][0]
)



n_fold = 10
models = []
models.append(model)

missclassification = np.zeros((n_fold,len(models)))
cv = skl_ms.KFold(n_splits=n_fold,random_state=1,shuffle=True)

for i,(train_index,val_index) in enumerate(cv.split(X)):
    
    X_train,X_val = X.iloc[train_index],X.iloc[val_index]
    y_train,y_val = y.iloc[train_index],y.iloc[val_index]
    for m in range(np.shape(models)[0]):
        model = models[m]
        model.fit(X_train,y_train)
        prediction = model.predict(X_val)
        missclassification[i,m] = np.mean(prediction==y_val)

print('Confusion Matrix for Gradient Boosting:\n')
print(pd.crosstab(prediction,y_val),'\n')

plt.boxplot(missclassification)
plt.title('Cross validation acc for different models')
plt.xticks(np.arange(1)+1,['Random Forest Classifier'])
plt.show()
