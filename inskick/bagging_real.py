#!/bin/python3

from sklearn.ensemble import BaggingClassifier
import pandas as pd
import numpy as np
import sklearn.discriminant_analysis as skl_da
import kfoldnew as kfn
'''
Uses Bagging with base mode QDA
Kfold is only implemented to retrive the confusion matrix for "unseen data"
The error estimation oob_score and the kfold estimation of E_new should be essentially equivalent
'''

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




x = x.drop(columns=["Year"])
x = x.drop(columns=["Gross"])



numwordratiodf = (x["Number words female"]-x["Number words male"])/x["Total words"]
x=x.assign(numwordratio = numwordratiodf)
x=x.drop(columns=["Number words male"])


diffagedf = x["Mean Age Male"]-x["Mean Age Female"]
x=x.assign(diffage = diffagedf)
x=x.drop(columns=["Mean Age Male"]) #bättre score med en med mer rimligt me båda?
x=x.drop(columns=["Mean Age Female"])


diffage2df = x["Age Lead"]-x["Age Co-Lead"]
x=x.assign(diffage2 = diffage2df)
x=x.drop(columns=["Age Lead"])
x=x.drop(columns=["Age Co-Lead"])



def weights(x, col, w, change = False):
    if not change: x = x.copy()
    x[col] = x[col].apply(lambda val: val*w)
    return x

'''
xn = weights(xn, "diffage", 10)
xn = weights(xn, "diffage2", 10)
xn = weights(xn, "numwordratio", 10)
'''
'''
err = []
for i in range(100):
    model = BaggingClassifier(estimator=skl_da.QuadraticDiscriminantAnalysis(reg_param=0), n_estimators=400, oob_score = True)
    model.fit(x, y)
    err.append(model.oob_score_)

plt.boxplot(err)
plt.title("Bagging accuracy for 100 runs")
plt.show()
'''


model, Ekfld, conf, bal = kfn.kfold(x,y,15,BaggingClassifier, False, estimator=skl_da.QuadraticDiscriminantAnalysis(), n_estimators=400, oob_score = True, max_samples=150)
pred = model.predict(x)
etrain = np.mean(pred == y)

print(f"Acc from kfold: {Ekfld}")
print(f"E_new from kfold: {1-Ekfld}")
print(f"oob score: {model.oob_score_}")
print(f"E_train: {1-etrain}")
print(f"E_gap: {Ekfld-etrain}")
print(f"Balanced accuracy: {bal}")
print("Confusion matrix")
print(conf)

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
