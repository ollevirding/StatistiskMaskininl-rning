import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms

from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier,GradientBoostingClassifier
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.drop(columns=['Lead'])
y = train['Lead']

trainI = np.random.choice(train.shape[0],size=800,replace=False)
trainIndex = train.index.isin(trainI)
traindata = train.iloc[trainIndex]
valdata = train.iloc[~trainIndex]

X_train = traindata.drop(columns=['Lead'])
y_train = traindata['Lead']
X_val = valdata.drop(columns=['Lead'])
y_val = valdata['Lead']

n = np.arange(50,400,30)
n_fold = 10
cv = skl_ms.KFold(n_splits=n_fold,random_state=2,shuffle=True)


missclassification = np.zeros(len(n))
for train_index,val_index in cv.split(X):
    X_train,X_val = X.iloc[train_index],X.iloc[val_index]
    y_train,y_val = y.iloc[train_index],y.iloc[val_index]
    for j,k in enumerate(n):
        model = GradientBoostingClassifier(n_estimators=k)
        model.fit(X_train,y_train)
        prediction = model.predict(X_val)
        missclassification[j] += np.mean(prediction == y_val) 

missclassification /= n_fold

plt.plot(n,missclassification)
plt.show()