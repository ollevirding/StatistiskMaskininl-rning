#!/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms


train = pd.read_csv('train.csv')
a = pd.get_dummies(train,columns=['Lead'])

Word_female = train['Number words female']
Word_male = train['Number words male']

for index, row in train.iterrows():
    if row['Lead'] == 'Female':
        Word_female.at[index] += row['Number of words lead']
    else:
        Word_male.at[index] += row['Number of words lead']


plt.scatter(train['Year'] ,[Word_male - Word_female], label = "Data points")
m1 = skl_lm.LinearRegression()

x = train["Year"]
y = Word_male-Word_female

y = y.values.reshape(-1,1)
x = x.values.reshape(-1,1)

m1.fit(x, y)
ycalc = m1.predict(x)
#print(ycalc)


plt.plot(x, ycalc,'r', label = f"Fitted line y = {round(m1.coef_[0,0])}x + {round(m1.intercept_[0])}")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Male words - Female words")
plt.title("Difference of words split across gender and year")#number
plt.show()