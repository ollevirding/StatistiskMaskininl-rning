#!/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
#from xgboost import XGBClassifier


data = pd.read_csv('train.csv') # kanske egentligen borde vara parameter
x = data.drop(columns=["Lead"])
y = data["Lead"]




p = 13
q = int(np.sqrt(p))





x = x.drop(columns=["Year"])
x["Gross"] = np.square(x['Gross'])
#x = x.drop(columns=["Gross"])



numwordratiodf = (x["Number words female"]-x["Number words male"])/x["Total words"]
x=x.assign(numwordratio = numwordratiodf)
x=x.drop(columns=["Number words female"]) ##borta
#x=x.drop(columns=["Number words male"])


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




model = RandomForestClassifier(oob_score=True, n_estimators=500)#, max_depth=7)

''' Tror att RandomForest = Bagging med Tree om max_features = None (tar med alla inputs som i bagging)
model = RandomForestClassifier(oob_score=True, n_estimators=500, max_features=None)#, max_depth=7)
model = BaggingClassifier(estimator=tree.DecisionTreeClassifier(), n_estimators=500, oob_score=True)
''' #Verkar va bättre än Randomforest med q = sqrt(p) (tar bara vissa inputs i varje runda)
#se 7.2



model.fit(x,y)
print(model.oob_score_)