import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms

from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.drop(columns=['Lead'])
y = train['Lead']

# kNN
n_runs = 10
K = np.arange(1,100)

missclassification = np.zeros((n_runs,len(K)))
for i in range(n_runs):
    X_train,X_val,y_train,y_val = skl_ms.train_test_split(X,y,test_size=0.3)

    for j,k in enumerate(K):
        model = skl_nb.KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train,y_train)
        prediction = model.predict(X_val)
        missclassification[i,j]= np.mean(prediction!=y_val)

average_mis = np.mean(missclassification,axis=0)
plt.plot(K,average_mis)
plt.show()  

# Tree based method
n_runs = 20
K = np.arange(1,20)

missclassification = np.zeros((n_runs,len(K)))
for i in range(n_runs):
    X_train,X_val,y_train,y_val = skl_ms.train_test_split(X,y,test_size=0.3)

    for j,k in enumerate(K):
        model = tree.DecisionTreeClassifier(max_depth=k)
        model.fit(X_train,y_train)
        prediction = model.predict(X_val)
        missclassification[i,j]= np.mean(prediction!=y_val)

average_mis = np.mean(missclassification,axis=0)
plt.plot(K,average_mis)
plt.show()  