#!/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms

from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
#from xgboost import XGBClassifier

np.random.seed(1)

traincomplete = pd.read_csv('train.csv')

traincompletex = traincomplete.drop(columns=["Lead"])
traincompletey = traincomplete["Lead"]


#test = pd.read_csv('test.csv')

#x = train.drop(columns=['Lead'])
#y = train['Lead']

print("Number of inputs", traincomplete.shape[0])


k = 15
klen = int(traincomplete.shape[0]/15)


splitInd = [i*klen for i in range(int(traincomplete.shape[0] / klen)+1)]
#print(splitInd)

if splitInd[-1] != traincomplete.shape[0]-1:
    splitInd.pop()
    #splitInd.append(x.shape[0]-1)

#print(splitInd)
traincompleteshuffle = traincomplete.sample(frac=1) #make sure the training data is not ordered
#print(xshuffle.index)
#print(x.index)
''' behöver nog inte göra såhär
xsplit = []
for i,ind in enumerate(splitInd):
        if i == len(splitInd)-1:
            xsplit.append(x.iloc[ind:])
        else:
            xsplit.append(x.iloc[ ind : splitInd[i+1]])
for hold in xsplit:
    train = xshuffle.iloc[~hold]
'''
#print([len(x) for x in xsplit])

#print(sum([len(x) for x in xsplit]))
Ehold = []
for i,ind in enumerate(splitInd):
        if i == len(splitInd)-1:
            hold = traincompleteshuffle.iloc[ind:]
            train = traincompleteshuffle.iloc[:ind]#+1??
            #print(hold.shape, train.shape)
            #print(hold.shape[0] + train.shape[0])

        else:#alt använd reindex på xshuffle och kör vanliga ~
            r=[i for i in range(ind, splitInd[i+1])] #~inverterar 
            rnot = [i for i in range(ind)] + [i for i in range(splitInd[i+1], traincomplete.shape[0])]            
            hold = traincompleteshuffle.iloc[r]#[ind:splitInd[i+1]]
            train = traincompleteshuffle.iloc[rnot]

            trainx = train.drop(columns=["Lead"])
            trainy = train["Lead"]
            holdx = hold.drop(columns=["Lead"])
            holdy = hold["Lead"]
            #rind = xshuffle.index.isin(r) # denna kommer nog ta bort shuffle :( sorterar kring index av x ine xshuffle
            #hold = xshuffle.iloc[rind]
            #train = xshuffle.iloc[~rind]
            #print(hold.shape, train.shape)
            #print(hold.shape[0] + train.shape[0])
            #print(hold.index)

            model = skl_da.QuadraticDiscriminantAnalysis()
            model.fit(trainx, trainy)
            prediction = model.predict(holdx)
            acc = np.mean(prediction == holdy)
            Ehold.append(acc)

Eholdavg = np.average(Ehold)
print("E_new approx", Eholdavg)

goodmodel = skl_da.QuadraticDiscriminantAnalysis()
goodmodel.fit(traincompletex, traincompletey)

