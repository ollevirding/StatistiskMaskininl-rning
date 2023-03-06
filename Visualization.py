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

number_male = []
number_female = []
for role in train['Lead']:
    if role == 'Male':
        number_male.append(1)
    elif role == 'Female':
        number_female.append(1)

print(f'Number of lead male actors = {len(number_male)}')
print(f'Number of lead female actors = {len(number_female)}')

plt.boxplot([Word_male, Word_female])
plt.xticks([1,2],['Male','Female'])
plt.title("Number of words split across gender")
plt.show()


plt.scatter(train['Year'] ,[Word_male - Word_female])
plt.title("Difference in words between the genders across the years")
plt.show()

plt.scatter([Word_male - Word_female], train['Gross'])
plt.title("Movie Gross based on the difference of words spoken by Male and Female")
plt.show()
