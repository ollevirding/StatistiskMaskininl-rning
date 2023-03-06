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
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

# define space
space = dict()
space['loss'] = ['log_loss','exponential'] # log_loss best - default setting
space['learning_rate'] = [0.1,0.4] # 0.2 best
space['n_estimators'] = [160,140] # 150 best
space['subsample'] = [0.8,0.5] # 0.78 best
space['criterion'] = ['friedman_mse','squared_error'] # friedman_use best - default setting 
space['min_samples_split'] = [10,80] # 92 best 
space['min_samples_leaf'] = [1,20] # 1 best
#space['min_weight_fraction_leaf'] = np.linspace(0,0.5,10) # 0.05 best default performs best
space['max_depth'] = [10,50] # 1 best - default setting
#space['min_impurity_decrease'] = np.linspace(0,100,10) # 0 best - default setting
#space['init'] = ['zero',None] # best None - default setting 
space['max_features'] = ['sqrt', 'log2'] # log2 best
#space['max_leaf_nodes'] = np.arange(2,1000,5)
#space['validation_fraction'] = [0.001,0.005] # best 0.01, basically no change


# define search
search = GridSearchCV(model,space,scoring='accuracy',n_jobs=-1,cv=cv)

# execute search
result = search.fit(X,y)

# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)



