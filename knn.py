#!/bin/python3
import pandas as pd
from kfold import kfold as kf
from kfoldnew import kfold as kfn
import sklearn.neighbors as skl_nb
import numpy as np
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

def weights(x, col, w, change = False):
    if not change: x = x.copy()
    x[col] = x[col].apply(lambda val: val*w)
    return x


def squareInput(x, col, change = False): 
    if not change: x = x.copy()
    x[col] = x[col].apply(lambda val: val**2)
    return x
def expInput(x, col, change = False): 
    if not change: x = x.copy()
    x[col] = x[col].apply(lambda val: np.exp(val))
    return x

data = pd.read_csv('train.csv') # kanske egentligen borde vara parameter
x = data.drop(columns=["Lead"])
y = data["Lead"]



#ändra inputsen
#model, error = kf(15,skl_nb.KNeighborsClassifier, True, n_neighbors = 8)

#x = inputNormalization(x,x)[0]
x = squareInput(x,"Gross")

#x = x.drop(columns=["Year"])
#x = x.drop(columns=["Gross"])
#
xn = inputNormalization(x,x)[0]
#xn = expInput(xn,"Gross")
#xn = weights(xn,"Number words male", 1.1)
#xn = weights(xn,"Number words female", 1.1)
#xn = weights(xn,"Number of words lead", 0.5)






model, error1 = kfn(x,y,15,skl_nb.KNeighborsClassifier, True, n_neighbors = 8)
print(error1)

model, error2 = kfn(xn,y,15,skl_nb.KNeighborsClassifier, False, n_neighbors = 8)
print(error2)

