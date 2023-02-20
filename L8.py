import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

import graphviz

np.random.seed(1)

OJ = pd.read_csv('oj.csv')

trainI = np.random.choice(OJ.shape[0],size=800,replace=False)
trainIndex = OJ.index.isin(trainI)
train = OJ.iloc[trainIndex]
test = OJ.iloc[~trainIndex]

X_train = train.drop(columns=['Purchase'])
y_train = train['Purchase']
X_train = pd.get_dummies(X_train, columns=['Store7'])
model = tree.DecisionTreeClassifier(splitter='random')
model.fit(X_train,y_train)

dot_data = tree.export_graphviz(model,out_file=None,feature_names = X_train.columns, class_names = model.classes_, filled = True, rounded = True, leaves_parallel=True,proportion=True)
graph = graphviz.Source(dot_data)
graph

X_test = test.drop(columns=['Purchase'])
y_test = test['Purchase']
X_test = pd.get_dummies(X_test, columns = ['Store7'])

prediction = model.predict(X_test)
acc = np.mean(prediction == y_test)
print(f'Accuracy is {acc}')
pd.crosstab(prediction,y_test)