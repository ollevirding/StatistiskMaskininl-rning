#!/bin/python3

'''
Uses Bagging with base mode QDA
Kfold is only implemented to retrive the confusion matrix for "unseen data"
The error estimation oob_score and the kfold estimation of E_new should be essentially equivalent
'''


from sklearn.ensemble import BaggingClassifier
import pandas as pd
import numpy as np
import sklearn.discriminant_analysis as skl_da
import kfoldnew as kfn


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



#x = inputNormalization(x,x)[0]



x = x.drop(columns=["Year"])
#x["Gross"] = np.square(x['Gross'])
x = x.drop(columns=["Gross"])



numwordratiodf = (x["Number words female"]-x["Number words male"])/x["Total words"]
x=x.assign(numwordratio = numwordratiodf)
#x=x.drop(columns=["Number words female"]) ##borta
x=x.drop(columns=["Number words male"])


diffagedf = x["Mean Age Male"]-x["Mean Age Female"] #bäst error med översta men rimligare med båda
x=x.assign(diffage = diffagedf)
x=x.drop(columns=["Mean Age Male"])
x=x.drop(columns=["Mean Age Female"])


#diffnumdf = x["Number of male actors"]-x["Number of female actors"]
#x=x.assign(diffnum = diffnumdf)
#x=x.drop(columns=["Number of female actors"])
#x=x.drop(columns=["Number of male actors"])


diffage2df = x["Age Lead"]-x["Age Co-Lead"]
x=x.assign(diffage2 = diffage2df)
x=x.drop(columns=["Age Lead"])
x=x.drop(columns=["Age Co-Lead"])



def weights(x, col, w, change = False):
    if not change: x = x.copy()
    x[col] = x[col].apply(lambda val: val*w)
    return x


#x = inputNormalization(x,x)[0]

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


#print(pd.crosstab(predic,yval))
#model = BaggingClassifier(estimator=skl_da.QuadraticDiscriminantAnalysis(), n_estimators=400, oob_score = True, max_samples=150)
#model, Ekfld, conf = kfn.kfold(x,y,15,model)

model, Ekfld, conf = kfn.kfold(x,y,15,BaggingClassifier, False, estimator=skl_da.QuadraticDiscriminantAnalysis(), n_estimators=400, oob_score = True, max_samples=150)

print(f"E_new from kfold {Ekfld}")
print(f"oob score {model.oob_score_}")
pred = model.predict(x)
etrain = np.mean(pred == y)
print(f"E_train {etrain}")
print(f"E_gap {Ekfld-etrain}")

print(f"Confusion matrix {conf}")


#E_train = np.mean(pred == y)
#print(pd.crosstab(model.predict(x), y))
#knn låg neighboors -> mycket komplexitet, liten bias men stor varians -> bagging bra
#knn

#model = BaggingClassifier(estimator=skl_nb.KNeighborsClassifier(n_neighbors=4), n_estimators=200, oob_score=True)
#model.fit(xn, y)


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
#print(model.oob_score_)
#print("E_new", 1-model.oob_score_)
#print("Training error", 1-E_train)
#print("Generalization gap", E_train-model.oob_score_)

#print("Balanced accuracy:",met.balanced_accuracy_score(y, pred))

#print('Confusion Matrix for Gradient Boosting:\n')
#print(pd.crosstab(pred,y),'\n')