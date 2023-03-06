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
train.info()
trainI = np.random.choice(train.shape[0],size=800,replace=False)
trainIndex = train.index.isin(trainI)
traindata = train.iloc[trainIndex]
valdata = train.iloc[~trainIndex]

X_train = traindata.drop(columns=['Lead'])
y_train = traindata['Lead']
X_val = valdata.drop(columns=['Lead'])
y_val = valdata['Lead']

models = []
models.append(skl_da.QuadraticDiscriminantAnalysis())
models.append(skl_da.QuadraticDiscriminantAnalysis())
#models.append(GradientBoostingClassifier(learning_rate = 0.1,n_estimators=210))


n_fold = 10
n_runs = 5
acc = np.zeros((n_fold,n_runs))
for runs in range(n_runs):

    for n in range(n_fold):
  
        trainI = np.random.choice(train.shape[0],size=800,replace=False)
        trainIndex = train.index.isin(trainI)
        traindata = train.iloc[trainIndex]
        valdata = train.iloc[~trainIndex]

        X_train = traindata.drop(columns=['Lead','Gross','Total words'])
        y_train = traindata['Lead']
        X_val = valdata.drop(columns=['Lead','Gross','Total words'])
        y_val = valdata['Lead'] 

        for m in range(np.shape(models)[0]):
            model = models[m]
            model.fit(X_train,y_train)
            prediction = model.predict(X_train)
            for i,j in enumerate(prediction):
                if j != y_train.iloc[i]:
                    for k in range(5):
                        traindata.loc[np.shape(traindata)[0]+1]=(traindata.loc[y_train.index[i]])
    
        X_train = traindata.drop(columns=['Lead','Gross','Total words'])
        y_train = traindata['Lead']

        model = GradientBoostingClassifier(learning_rate = 0.1,n_estimators=190)
        model.fit(X_train,y_train)
        prediction = model.predict(X_val)
        acc[n,runs] = np.mean(prediction == y_val)

plt.boxplot(acc)
plt.title('Accuracy for LoveBoosting')
plt.show()
