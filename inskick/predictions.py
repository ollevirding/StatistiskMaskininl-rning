import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skl_m
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier


def InputSelection(x):
    x = x.drop(columns=["Year"])
    x = x.drop(columns=["Gross"])

    numwordratiodf = (x["Number words female"] -
                    x["Number words male"])/x["Total words"]
    x = x.assign(numwordratio=numwordratiodf)
    x = x.drop(columns=["Number words male"])

    diffagedf = x["Mean Age Male"]-x["Mean Age Female"]
    x = x.assign(diffage=diffagedf)
    x = x.drop(columns=["Mean Age Male"])
    x = x.drop(columns=["Mean Age Female"])

    diffage2df = x["Age Lead"]-x["Age Co-Lead"]
    x = x.assign(diffage2=diffage2df)
    x = x.drop(columns=["Age Lead"])
    x = x.drop(columns=["Age Co-Lead"])
    return x


model = BaggingClassifier(estimator=skl_da.QuadraticDiscriminantAnalysis(), n_estimators=400)

data = pd.read_csv('train.csv') 
x_test = pd.read_csv('test.csv') 

x_train = data.drop(columns=["Lead"])
y_train = data["Lead"]

x_train = InputSelection(x_train)
x_test = InputSelection(x_test)

model.fit(x_train,y_train)

prediction = model.predict(x_test)

for i,gender in enumerate(prediction):
    if gender == 'Male':
        prediction[i] = int(0)
    else:
        prediction[i] = int(1)


np.savetxt('predictions.csv', np.reshape(prediction, [1, -1]), delimiter=',', fmt='%d')

