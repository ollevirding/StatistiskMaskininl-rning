#!/bin/python3
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from kfold import kfold

def inputNormalization(x,xval, change = False):
    if not change: 
        x=x.copy()
        xval = xval.copy()

    maxs = [max(x[col]) for col in x]
    mins = [min(x[col]) for col in x]

    for i,col in enumerate(x):
        #for val in x[col]
        x[col] = x[col].apply(lambda val: (val-mins[i])/(maxs[i]-mins[i]))
        xval[col] = xval[col].apply(lambda val: (val-mins[i])/(maxs[i]-mins[i]))
        #x[col] = (x[col]-mins[i])/(maxs[i]-mins[i])
    return [x,xval]
#kfold
#normalize


train = pd.read_csv('train.csv')
x = train.drop(columns=['Lead'])
y = train['Lead']


#x_n, y_n = inputNormalization(x, y)
model, error = kfold(MLPClassifier, norm=True, solver="lbfgs", alpha = 1e-5, hidden_layer_sizes=(100,2), random_state=1)
print(error)