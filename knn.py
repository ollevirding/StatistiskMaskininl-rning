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

def logInput(x, col, change = False): 
    if not change: x = x.copy()
    x[col] = x[col].apply(lambda val: np.log(val) if val>0 else 0)
    return x


def squareInput(x, col, change = False): 
    if not change: x = x.copy()
    x[col] = x[col].apply(lambda val: val**2)
    return x
def expInput(x, col, change = False): 
    if not change: x = x.copy()
    x[col] = x[col].apply(lambda val: np.exp(val))
    return x

def expminusInput(x, col, change = False): 
    if not change: x = x.copy()
    x[col] = x[col].apply(lambda val: np.exp(-val))
    return x


data = pd.read_csv('train.csv') # kanske egentligen borde vara parameter
x = data.drop(columns=["Lead"])
y = data["Lead"]



'''
obs test.csv måste normaliseras genom samma som träningsdatan går genom
'''

#ändra inputsen
#model, error = kf(15,skl_nb.KNeighborsClassifier, True, n_neighbors = 8)

#x = inputNormalization(x,x)[0]

x = x.drop(columns=["Year"])
x["Gross"] = np.square(x['Gross'])
#x = x.drop(columns=["Gross"])



numwordratiodf = (x["Number words female"]-x["Number words male"])/x["Total words"]
x=x.assign(numwordratio = numwordratiodf)
x=x.drop(columns=["Number words female"])
#x=x.drop(columns=["Number words male"])


diffagedf = x["Mean Age Male"]-x["Mean Age Female"] #bäst error med översta men rimligare med båda
x=x.assign(diffage = diffagedf)
x=x.drop(columns=["Mean Age Male"])
#x=x.drop(columns=["Mean Age Female"])

diffnumdf = x["Number of male actors"]-x["Number of female actors"]
x=x.assign(diffnum = diffnumdf)
#x=x.drop(columns=["Number of female actors"])
#x=x.drop(columns=["Number of male actors"])

diffage2df = x["Age Lead"]-x["Age Co-Lead"]
x=x.assign(diffage2 = diffage2df)
x=x.drop(columns=["Age Lead"])
x=x.drop(columns=["Age Co-Lead"])


'''
print(x["Mean Age Female"])
x["Age Lead"] = x["Age Lead"]-x["Age Co-Lead"]
x=x.drop(columns=["Age Co-Lead"])
'''


for col in x:
    print(col)



#x = x.drop(columns=["Mean Age Male"])

#x = weights(x, "Age Lead", 1.5)
#x = weights(x, "Age Co-Lead", 1.5)

#Mean age male/Female lite
#Age lead
#age colead
#

xn = inputNormalization(x,x)[0]

'''
är det verkligen rätt att kfold på normaliserad data?
borde inte kfold normalisera varje lilla
'''




#xn["Gross"] = np.square(xn['Gross'])

#xn = logInput(xn,"Age Lead")
#xn = squareInput(xn, "Age Lead")
#xn = weights(xn, "Age Lead", 1.5)
#xn = weights(xn, "Age Co-Lead", 1.1)

#xn = expInput(xn,"Gross")
#xn = weights(xn,"Number words male", 1.1)
#xn = weights(xn,"Number words female", 1.1)
#xn = weights(xn,"Number of words lead", 0.5)






#model, error1 = kfn(x,y,15,skl_nb.KNeighborsClassifier, True, n_neighbors = 8)
#print(error1)

model, E_new = kfn(xn,y,15,skl_nb.KNeighborsClassifier, False, n_neighbors = 8)
print(E_new)
E_new_real = 1-E_new

pred_train = model.predict(xn)
E_train = np.mean(pred_train == y)
E_train_real = 1-E_train
print("Estimated E_new:", E_new_real)
print("E_train:", E_train_real)
print("Generalization gap:", E_new_real-E_train_real)

#test måste få samma normalisation