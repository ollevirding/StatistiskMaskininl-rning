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



if False:
    plt.boxplot([Word_male, Word_female])
    plt.xticks([1,2],['Male','Female'])
    plt.title("Number of words split across gender")
    plt.show()


plt.scatter(train['Year'] ,[Word_male - Word_female])
m1 = skl_lm.LinearRegression()
m1.fit(train['Year'], Word_male-Word_female)
ycalc = m1.predict(train["Year"])
print(ycalc)
plt.plot(train['Year'], ycalc)
plt.title("Difference of words split across gender and year")#number
plt.show()

if False:
    plt.scatter([Word_male - Word_female], train['Gross'])
    plt.title("Income based on the fraction of words spoken by Male divided by Female")
    plt.show()

    plt.scatter(train['Total words'],train['Lead'])
    plt.show()