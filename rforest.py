#!/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
#from xgboost import XGBClassifier


traincomplete = pd.read_csv('train.csv') # kanske egentligen borde vara parameter
traincompletex = traincomplete.drop(columns=["Lead"])
traincompletey = traincomplete["Lead"]




p = 13
q = int(np.sqrt(p))


model = RandomForestClassifier(oob_score=True, n_estimators=500)#, max_depth=7)

''' Tror att RandomForest = Bagging med Tree om max_features = None (tar med alla inputs som i bagging)
model = RandomForestClassifier(oob_score=True, n_estimators=500, max_features=None)#, max_depth=7)
model = BaggingClassifier(estimator=tree.DecisionTreeClassifier(), n_estimators=500, oob_score=True)
''' #Verkar va bättre än Randomforest med q = sqrt(p) (tar bara vissa inputs i varje runda)
#se 7.2



model.fit(traincompletex,traincompletey)
print(model.oob_score_)