import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

np.random.seed(1)



data = pd.read_csv('train.csv') # kanske egentligen borde vara parameter
x = data.drop(columns=["Lead"])
y = data["Lead"]

x = x.drop(columns=["Year"])
x = x.drop(columns=["Gross"])

numwordratiodf = (x["Number words female"]-x["Number words male"])/x["Total words"]
x=x.assign(numwordratio = numwordratiodf)
x=x.drop(columns=["Number words male"])

diffagedf = x["Mean Age Male"]-x["Mean Age Female"] 
x=x.assign(diffage = diffagedf)
x=x.drop(columns=["Mean Age Male"])
x=x.drop(columns=["Mean Age Female"])

diffage2df = x["Age Lead"]-x["Age Co-Lead"]
x=x.assign(diffage2 = diffage2df)
x=x.drop(columns=["Age Lead"])
x=x.drop(columns=["Age Co-Lead"])



def weights(x, col, w, change = False):
    if not change: x = x.copy()
    x[col] = x[col].apply(lambda val: val*w)
    return x

parameters = [
    [600],  # n_estimators (The number of trees in the forrest)
    ["entropy"],  # criterion (The function to measure the quality of a split)
    [60],  # Max_depth (The maximum depth of the tree)
    # min_samples_split (The minimum number of samples required to split an internal node)
    [7],
    # min_samples_leaf (The minimum number of samples required to be at a leaf node)
    [3],
    # min_weight_fraction_leaf (The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.)
    [0.0],
    # max_features (The number of features to consider when looking for the best split:)
    [3],
    [420],  # max_leaf_nodes (Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.)
    # min_impurity_decrease (A node will be split if this split induces a decrease of the impurity greater than or equal to this value.)
    [0.0],
    # bootstrap (Whether bootstrap samples are used when building trees)
    [True],
]

model = RandomForestClassifier(
    n_estimators                =   parameters[0][0],
    criterion                   =   parameters[1][0],
    max_depth                   =   parameters[2][0],
    min_samples_split           =   parameters[3][0],
    min_samples_leaf            =   parameters[4][0],
    min_weight_fraction_leaf    =   parameters[5][0],
    max_leaf_nodes              =   parameters[7][0],
    bootstrap                   =   parameters[9][0]
)



n_fold = 100
models = []
models.append(model)

missclassification = np.zeros((n_fold,len(models)))
cv = skl_ms.KFold(n_splits=n_fold,random_state=1,shuffle=True)
for i,(train_index,val_index) in enumerate(cv.split(x)):
    
    X_train,X_val = x.iloc[train_index],x.iloc[val_index]
    y_train,y_val = y.iloc[train_index],y.iloc[val_index]
    for m in range(np.shape(models)[0]):
        model = models[m]
        model.fit(X_train,y_train)
        prediction = model.predict(X_val)
        missclassification[i,m] = np.mean(prediction==y_val)
        print(pd.crosstab(prediction,y_val),'\n')

print('Confusion Matrix for Gradient Boosting:\n')
print(pd.crosstab(prediction,y_val),'\n')

plt.boxplot(missclassification)
plt.title('Cross validation accuracy for Random Forrest Classifier')
plt.xticks(np.arange(1)+1,['Random Forest Classifier'])
plt.show()
