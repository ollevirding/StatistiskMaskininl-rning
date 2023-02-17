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

# define model
model = GradientBoostingClassifier()

# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define space
space = dict()
#space['loss'] = ['log_loss','exponential'] # log_loss best
space['learning_rate'] = np.linspace(0.01,1,10) # 0.45 best
space['n_estimators'] = np.linspace(100,200,10) # 150 best
#space['subsample'] = np.linspace(0.01,1,10) # 0.78 best
#space['criterion'] = ['friedman_mse','squared_error'] # friedman_use best
#space['min_samples_split'] = np.arange(2,100,10) # 12 best
#space['min_samples_leaf'] = np.arange(1,100,10)
#space['min_weight_fraction_leaf'] = np.linspace(0,0.5,10)
#space['max_depth'] = np.arange(1,100,10)
#space['min_impurity_decrease'] = np.linspace(0,100,10)
#space['init'] = ['zero',None]
#space['max_features'] = [1, 'sqrt', 'log2']
#space['max_leaf_nodes'] = [np.arange(2,1000,10),None]
#space['validation_fraction'] = np.linspace(0.01,0.99,10)


# define search
search = GridSearchCV(model,space,scoring='accuracy',n_jobs=-1,cv=cv)

# execute search
result = search.fit(X,y)

# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)



