import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

f = plt.figure(figsize=(5, 7))
cormat = train.corr()
round(cormat,2)
sns.heatmap(cormat, annot = True)
plt.show()