import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,RepeatedStratifiedKFold
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier,GradientBoostingClassifier
from scipy.stats import loguniform

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.drop(columns=['Lead','Total words','Gross'])
y = train['Lead']
X_train,X_val,y_train,y_val = skl_ms.train_test_split(X,y,test_size=0.3)

#model = GradientBoostingClassifier(learning_rate=0.3,n_estimators=250,min_samples_split = 12,max_features='sqrt',subsample=0.9)

n_fold = 10
n_runs = 2
estimators = [180,190,200,220,250]
accuracy = np.zeros((n_fold,n_runs+1))
models = []
n_estimators = 210
models.append(GradientBoostingClassifier(learning_rate = 0.1,n_estimators=n_estimators))
models.append(GradientBoostingClassifier())
model = GradientBoostingClassifier(learning_rate = 0.1, n_estimators = n_estimators)
acc = np.zeros(n_fold)
for n in range(n_fold):
    X_train,X_val,y_train,y_val = skl_ms.train_test_split(X,y,test_size=0.3)

    model.fit(X_train,y_train)
    prediction = model.predict(X_val)
    acc[n] = np.mean(prediction==y_val)

print(f'Accuracy of tuned GradiantBoosting {np.mean(acc)}')

for n in range(n_runs):
    cv = skl_ms.KFold(n_splits=n_fold,random_state=0,shuffle=True)
    model = models[n]

    for i,(train_index,val_index) in enumerate(cv.split(X)):
    
        X_train,X_val = X.iloc[train_index],X.iloc[val_index]
        y_train,y_val = y.iloc[train_index],y.iloc[val_index]
        model.fit(X_train,y_train)
        prediction = model.predict(X_val)
        accuracy[i,n] = np.mean(prediction==y_val)
    X = train.drop(columns=['Lead'])
    y = train['Lead']
cv = skl_ms.KFold(n_splits=n_fold,random_state=1,shuffle=True)                                                                                  

MaleGuesser = []

for j in range(len(y_val)):
    MaleGuesser.append('Male')

for i,(train_index,val_index) in enumerate(cv.split(X)):
    accuracy[i,2] = np.mean(MaleGuesser==y_val)
plt.boxplot(accuracy)
plt.title(f'K-Fold validation acc for GradientBoosting for n_estimators  = {n_estimators}')
plt.xticks(np.arange(n_runs+1)+1,('Tuned Gradiant Boosting','Untuned Gradient Boosting','Classifier Only Guessing Male'))
plt.show()

print('Confusion Matrix for tuned Gradient Boosting:\n')
print(pd.crosstab(prediction,y_val),'\n')


