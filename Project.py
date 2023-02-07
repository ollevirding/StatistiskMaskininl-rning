import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms

from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


np.random.seed(1)

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

# Logistic regression
model = skl_lm.LogisticRegression(solver='liblinear')
model.fit(X_train,y_train)
prediction = model.predict(X_val)
acc = np.mean(prediction == y_val)
print(f'Accuracy for Logistic Regression: {acc}')

# LDA
model = skl_da.LinearDiscriminantAnalysis()
model.fit(X_train,y_train)
prediction = model.predict(X_val)
acc = np.mean(prediction == y_val)
print(f'Accuracy for LDA: {acc}')

# QDA 
model = skl_da.QuadraticDiscriminantAnalysis()
model.fit(X_train,y_train)
prediction = model.predict(X_val)
acc = np.mean(prediction == y_val)
print(f'Accuracy for QDA: {acc}')

# kNN
model = skl_nb.KNeighborsClassifier(n_neighbors=4)
model.fit(X_train,y_train)
prediction = model.predict(X_val)
acc = np.mean(prediction == y_val)
print(f'Accuracy for kNN: {acc}')

# Tree based method
model = tree.DecisionTreeClassifier(max_depth = 7)
model.fit(X_train,y_train)
prediction = model.predict(X_val)
acc = np.mean(prediction == y_val)
print(f'Accuracy for tree based method: {acc}')

# Bagging
model = BaggingClassifier()
model.fit(X_train,y_train)
prediction = model.predict(X_val)
acc = np.mean(prediction == y_val)
print(f'Accuracy for bagging: {acc}')

# RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,y_train)
prediction = model.predict(X_val)
acc = np.mean(prediction == y_val)
print(f'Accuracy for randomforestclassifier: {acc}')

# AdaBoost
model = AdaBoostClassifier()
model.fit(X_train,y_train)
prediction = model.predict(X_val)
acc = np.mean(prediction == y_val)
print(f'Accuracy for AdaBoosting: {acc}')

# GradientBoosting
model = GradientBoostingClassifier()
model.fit(X_train,y_train)
prediction = model.predict(X_val)
acc = np.mean(prediction == y_val)
print(f'Accuracy for GradientBoosting: {acc}')

# Love boosting v2
models = []
models.append(skl_lm.LogisticRegression(solver='newton-cholesky'))
models.append(skl_da.QuadraticDiscriminantAnalysis())

for m in range(np.shape(models)[0]):
    model = models[m]
    model.fit(X_train,y_train)
    prediction = model.predict(X_val)
    for i,j in enumerate(prediction):
        if j != y_train.iloc[i]:
            traindata.loc[np.shape(traindata)[0]+1]=(traindata.loc[y_train.index[i]])
            X_train = traindata.drop(columns=['Lead'])
            y_train = traindata['Lead']

model = GradientBoostingClassifier()
model.fit(X_train,y_train)
prediction = model.predict(X_val)
acc = np.mean(prediction == y_val)
print(f'Love boosting v2 accuracy: {acc}')
    
n_fold = 10
models = []
models.append(skl_lm.LogisticRegression(solver='newton-cholesky'))
models.append(skl_da.LinearDiscriminantAnalysis())
models.append(skl_da.QuadraticDiscriminantAnalysis())
models.append(skl_nb.KNeighborsClassifier(n_neighbors=4))
models.append(tree.DecisionTreeClassifier(max_depth = 7))
models.append(BaggingClassifier())
models.append(RandomForestClassifier())
models.append(AdaBoostClassifier())
models.append(GradientBoostingClassifier())

missclassification = np.zeros((n_fold,len(models)))
cv = skl_ms.KFold(n_splits=n_fold,random_state=1,shuffle=True)

for i,(train_index,val_index) in enumerate(cv.split(X)):
    X_train,X_val = X.iloc[train_index],X.iloc[val_index]
    y_train,y_val = y.iloc[train_index],y.iloc[val_index]
    for m in range(np.shape(models)[0]):
        model = models[m]
        model.fit(X_train,y_train)
        prediction = model.predict(X_val)
        missclassification[i,m] = np.mean(prediction == y_val)

plt.boxplot(missclassification)
plt.title('Accuracy for different models')
plt.xticks(np.arange(9)+1,('Logistic Regression','LDA','QDA','kNN','Tree Based','Bagging', 'Random Forest', 'AdaBoost', 'GradientBoost'))
plt.show()
#def mftobinary():
