#!/bin/python3
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.drop(columns=['Lead'])
y = train['Lead']

trainI = np.random.choice(train.shape[0],size=800,replace=False)
trainIndex = train.index.isin(trainI)
traindata = train.iloc[trainIndex]
valdata = train.iloc[~trainIndex]

X_train = traindata.drop(columns=['Lead'])

print(X_train)
print("-"*100)
for col in X_train:
    X_train[col] = X_train[col]/10000

print(X_train["Year"])