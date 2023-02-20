#!/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
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


data = pd.read_csv("train.csv", na_values='?', dtype={'ID': str}).dropna().reset_index()

x = data.iloc[:,1:-1]
y=data.iloc[:,-1]

indexMale = [i for i,gender in enumerate(data["Lead"]) if gender == "Male"]
indexFemale = [i for i,gender in enumerate(data["Lead"]) if gender == "Female"] #alt alla index - male index

#plt.figure(1)
#plt.scatter(data["Year"], data["Lead"])
#plt.show()

#plt.figure(2)
#plt.scatter(data["Gross"], data["Lead"])
#plt.show()

def getMF(dataName):
    return [[x[dataName][i] for i in indexMale], [x[dataName][i] for i in indexFemale]]

def plotData(x,y, malecol = "blue", femcol = "red"):
    plt.title(f"Male {malecol} | Female {femcol}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.scatter(getMF(x)[0], getMF(y)[0], c=malecol)
    plt.scatter(getMF(x)[1], getMF(y)[1], c=femcol)
    plt.show()



plotData("Year", "Gross")

#new = x["Year"]*x["Gross"]
#x = x.assign(newvar = new)
#x["Gross"] = np.log(x['Gross'])
#plotData("Year", "Gross")

#x = inputNormalization(x,x)[0]
#x["Gross"] = np.square(x['Gross'])#np.log(x['Gross'])
#x["Year"] = np.square(x['Year'])#np.log(x['Gross'])

#plotData("Year","Gross")
plotData("Number words male","Gross")
plotData("Number words female","Gross")

grossm, grossf = getMF("Gross")

plt.hist(grossm)
plt.hist(grossf)
plt.show()
    
def gross():
    mf = getMF("Gross")

    plt.figure(1)
    plt.hist(mf[0], bins=20)
    plt.xlim(0,600)
    #plt.ylim(0,500)
    plt.xlabel("Pengar")
    plt.title("male")

    plt.figure(2)
    plt.hist(mf[1], bins = 20)
    plt.xlim(0,600)
 #   plt.ylim(0,500)
    plt.xlabel("Pengar")
    plt.title("female")
    plt.show()
    '''
    Vill amplify stora värden och hålla små isch samma
    '''
def totwrd():
    mf = getMF("Total words")

    plt.figure(1)
    plt.hist(mf[0], bins=20)
    plt.xlim(0,50_000)
    #plt.ylim(0,220)
    plt.xlabel("Total words")
    plt.title("male")

    plt.figure(2)
    plt.hist(mf[1], bins = 20)
    plt.xlim(0,50_000)
    #plt.ylim(0,220)
    plt.xlabel("Total words")
    plt.title("female")
    plt.show()
    '''
    inte superstor påverkan?
    '''
def femwrd():
    mf = getMF("Number words female")

    plt.figure(1)
    plt.hist(mf[0], bins=20)
    plt.xlim(0,20_000)
    #plt.ylim(0,220)
    plt.xlabel("female words")
    plt.title("male")

    plt.figure(2)
    plt.hist(mf[1], bins = 20)
    plt.xlim(0,20_000)
    #plt.ylim(0,220)
    plt.xlabel("female words")
    plt.title("female")
    plt.show()
    '''
    inte superstor påverkan?
    '''
def malwrd():
    mf = getMF("Number words male")

    plt.figure(1)
    plt.hist(mf[0], bins=20)
    plt.xlim(0,20_000)
    #plt.ylim(0,220)
    plt.xlabel("male words")
    plt.title("male")

    plt.figure(2)
    plt.hist(mf[1], bins = 20)
    plt.xlim(0,20_000)
    #plt.ylim(0,220)
    plt.xlabel("male words")
    plt.title("female")
    plt.show()
    '''
    inte superstor påverkan?
    '''
def diffwrd():
    mf = getMF("Difference in words lead and co-lead")

    plt.figure(1)
    plt.hist(mf[0], bins=20)
    plt.xlim(0,15_000)
    #plt.ylim(0,220)
    plt.xlabel("Difference in words lead and co-lead")
    plt.title("male")

    plt.figure(2)
    plt.hist(mf[1], bins = 20)
    plt.xlim(0,15_000)
    #plt.ylim(0,220)
    plt.xlabel("Difference in words lead and co-lead")
    plt.title("female")
    plt.show()
    '''
    inte superstor påverkan?
    '''
def leadwrd():
    mf = getMF("Number of words lead")

    plt.figure(1)
    plt.hist(mf[0], bins=20)
    plt.xlim(0,15_000)
    #plt.ylim(0,220)
    plt.xlabel("Number of words lead")
    plt.title("male")

    plt.figure(2)
    plt.hist(mf[1], bins = 20)
    plt.xlim(0,15_000)
    #plt.ylim(0,220)
    plt.xlabel("Number of words lead")
    plt.title("female")
    plt.show()
    '''
    inte superstor påverkan?
    '''
def nummale():
    mf = getMF("Number of male actors")

    plt.figure(1)
    plt.hist(mf[0], bins=20)
    plt.xlim(0,25)
    #plt.ylim(0,220)
    plt.xlabel("Number of male actors")
    plt.title("male")

    plt.figure(2)
    plt.hist(mf[1], bins = 20)
    plt.xlim(0,25)
    #plt.ylim(0,220)
    plt.xlabel("Number of male actors")
    plt.title("female")
    plt.show()
    '''
    inte superstor påverkan?
    '''
def numfem():
    mf = getMF("Number of female actors")

    plt.figure(1)
    plt.hist(mf[0], bins=20)
    plt.xlim(0,25)
    #plt.ylim(0,220)
    plt.xlabel("Number of female actors")
    plt.title("male")

    plt.figure(2)
    plt.hist(mf[1], bins = 20)
    plt.xlim(0,25)
    #plt.ylim(0,220)
    plt.xlabel("Number of female actors")
    plt.title("female")
    plt.show()
    '''
    inte superstor påverkan?
    '''
def leadage():
    mf = getMF("Age Lead")

    plt.figure(1)
    plt.hist(mf[0], bins=20)
    plt.xlim(0,100)
    #plt.ylim(0,220)
    plt.xlabel("Age lead")
    plt.title("male")

    plt.figure(2)
    plt.hist(mf[1], bins = 20)
    plt.xlim(0,100)
    #plt.ylim(0,220)
    plt.xlabel("Age lead")
    plt.title("female")
    plt.show()
    '''
    inte superstor påverkan?
    '''
def coleadage():
    mf = getMF("Age Co-Lead")

    plt.figure(1)
    plt.hist(mf[0], bins=20)
    plt.xlim(0,100)
    #plt.ylim(0,220)
    plt.xlabel("Age colead")
    plt.title("male")

    plt.figure(2)
    plt.hist(mf[1], bins = 20)
    plt.xlim(0,100)
    #plt.ylim(0,220)
    plt.xlabel("Age colead")
    plt.title("female")
    plt.show()
    '''
    inte superstor påverkan?
    '''
def yr():
    mf = getMF("Year")

    plt.figure(1)
    plt.hist(mf[0], bins=20)
    plt.xlim(1900,2020)
    #plt.ylim(0,220)
    plt.xlabel("Year")
    plt.title("male")

    plt.figure(2)
    plt.hist(mf[1], bins = 20)
    plt.xlim(1900,2020)
    #plt.ylim(0,220)
    plt.xlabel("Year")
    plt.title("female")
    plt.show()
    '''
    inte superstor påverkan?
    '''

#gross()
#print(data.get("Difference in words lead and co-lead").corr(data.get("Total words")))



'''
year
gross
words?
'''




#kolla om t.ex year har mest påverkan i början eller slut osv -> log
#softmax


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


