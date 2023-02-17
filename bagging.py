#!/bin/python3

from sklearn.ensemble import BaggingClassifier
import pandas as pd
import numpy as np
import sklearn.discriminant_analysis as skl_da

import sklearn.neighbors as skl_nb


data = pd.read_csv('train.csv') # kanske egentligen borde vara parameter
x = data.drop(columns=["Lead"])
y = data["Lead"]

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

#model = BaggingClassifier(estimator=skl_da.QuadraticDiscriminantAnalysis(), n_estimators=400, oob_score = True)
#model.fit(x, y)

#knn låg neighboors -> mycket komplexitet, liten bias men stor varians -> bagging bra
#knn

model = BaggingClassifier(estimator=skl_nb.KNeighborsClassifier(n_neighbors=3), n_estimators=200, oob_score=True)
model.fit(x, y)


#knn med norm
'''
model = BaggingClassifier(estimator=skl_nb.KNeighborsClassifier(n_neighbors=6), n_estimators=200, oob_score=True)
x_norm = inputNormalization(x,x)[0]
model.fit(x_norm, y)
'''

'''
Öka trainingdata med liknande data(bootstrapping), räkna ut fit och error för varje del i uppdelat set
Fit blir average(el. majority vote) av allas svar
Reducerar varians - därför bra för modeller med hög varians, ex knn där n<<, andra komplexa modeller

E_new ~ average av alla fel
Varje del-lösning bra bias men dålig varians(låg n) -> average påverkar ej bias
men reducerar varians
OOB: varje "bag" använder ~63% av den originella datan -> det som blir kvar kan 
användas som hold out och användas för felberäkning, likt kfold
'''
print(model.oob_score_)
