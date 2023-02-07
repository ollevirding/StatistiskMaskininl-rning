#!/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb

data = pd.read_csv("train.csv", na_values='?', dtype={'ID': str}).dropna().reset_index()

x = data.iloc[:,1:-1]
y=data.iloc[:,-1]









#print(x)
#print(y)

#plt.scatter(data["Year"], data["Lead"])

#males = [i for i,d in enumerate(data["Lead"]) if d == "Male"]

#print(data.get(data.
#print(data.isin(["male"]))

#plt.show()

##väljs ut höjd/2 st siffror från lista upp till 0
#indexing_ = np.random.choice(biopsy.shape[0], int(biopsy.shape[0]/2), replace = False)

#indexing = biopsy.index.isin(indexing_) #gör inte så stor skillnad?

#test = biopsy.iloc[indexing]
#train = biopsy.iloc[~indexing]

#testX = test[["V3","V4", "V5"]]
#testY = test["class"]

#trainX = train[["V3","V4", "V5"]]
#trainY = train["class"]





















#import matplotlib.pyplot as plt
#import numpy as np
#
#fileName = "train.csv"
#with open(fileName, "r") as f:
#    r = f.read().split(",", )
#    #f.read
#    #inf = [f.readline().split() for i in range(100000) if f.r]
#
#    #inf = [l for l in f.read().split()]
#    #inf = f.read().split()
#
##ar = np.reshape(np.array(r), (-1,14))
##print(ar[0,:])
#print(r)


