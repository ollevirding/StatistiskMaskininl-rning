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

model = GradientBoostingClassifier(learning_rate=0.15,n_estimators=160,subsample=0.9,min_samples_split = 80, max_features = 'sqrt',max_depth = 50)

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
plt.xticks(np.arange(1)+1,['Gradientboosting'])
plt.show()
