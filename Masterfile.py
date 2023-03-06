import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skl_m
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier

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

def InputSelection(x):
    x = x.drop(columns=["Year"])
    x = x.drop(columns=["Gross"])

    numwordratiodf = (x["Number words female"] -
                    x["Number words male"])/x["Total words"]
    x = x.assign(numwordratio=numwordratiodf)
    x = x.drop(columns=["Number words male"])

    diffagedf = x["Mean Age Male"]-x["Mean Age Female"]
    x = x.assign(diffage=diffagedf)
    x = x.drop(columns=["Mean Age Male"])
    x = x.drop(columns=["Mean Age Female"])

    diffage2df = x["Age Lead"]-x["Age Co-Lead"]
    x = x.assign(diffage2=diffage2df)
    x = x.drop(columns=["Age Lead"])
    x = x.drop(columns=["Age Co-Lead"])
    return x

#====================================================#
#   INPUT MANIPULATION                               #
#====================================================#


data = pd.read_csv('train.csv')
x = data.drop(columns=["Lead"])
y = data["Lead"]

x = InputSelection(x)


#===============================================================#
#   K-FOLD FOR RANDOM FORREST, BAGGING AND GRADIENT BOOSTING    #
#===============================================================#



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
models = []
model = RandomForestClassifier(
    n_estimators=parameters[0][0],
    criterion=parameters[1][0],
    max_depth=parameters[2][0],
    min_samples_split=parameters[3][0],
    min_samples_leaf=parameters[4][0],
    min_weight_fraction_leaf=parameters[5][0],
    max_leaf_nodes=parameters[7][0],
    bootstrap=parameters[9][0]
)

models.append(model)

model = GradientBoostingClassifier(
    learning_rate=0.1,
    n_estimators=210
)
models.append(model)

model = BaggingClassifier(estimator=skl_da.QuadraticDiscriminantAnalysis(), n_estimators=400)
models.append(model)

n_fold = 2
missclassification = np.zeros((n_fold, len(models)+1))
recall = np.zeros((n_fold, len(models)+1))
precision = np.zeros((n_fold, len(models)+1))
f1 = np.zeros((n_fold, len(models)+1))
for m,model in enumerate(models):
    
    cv = skl_ms.KFold(n_splits=n_fold, random_state=1, shuffle=True)

    for i, (train_index, val_index) in enumerate(cv.split(x)):

        X_train, X_val = x.iloc[train_index], x.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        model.fit(X_train, y_train)
        prediction = model.predict(X_val)
        missclassification[i,m] = np.mean(prediction == y_val)
        f1[i,m] = skl_m.f1_score(prediction,y_val,pos_label='Female')
        precision[i,m] = skl_m.precision_score(prediction,y_val,pos_label='Female')
        recall[i,m] = skl_m.recall_score(prediction,y_val,pos_label='Female')


#====================================================#
#   K-FOLD FOR K-NEAREST NEIGHBOUR                   #
#====================================================#


data = pd.read_csv('train.csv') # kanske egentligen borde vara parameter
x = data.drop(columns=["Lead"])
y = data["Lead"]

x = x.drop(columns=["Year"])
x["Gross"] = np.square(x['Gross'])

numwordratiodf = (x["Number words female"]-x["Number words male"])/x["Total words"]
x=x.assign(numwordratio = numwordratiodf)
x=x.drop(columns=["Number words female"])

diffagedf = x["Mean Age Male"]-x["Mean Age Female"] 
x=x.assign(diffage = diffagedf)
x=x.drop(columns=["Mean Age Male"])

diffnumdf = x["Number of male actors"]-x["Number of female actors"]
x=x.assign(diffnum = diffnumdf)

diffage2df = x["Age Lead"]-x["Age Co-Lead"]
x=x.assign(diffage2 = diffage2df)
x=x.drop(columns=["Age Lead"])
x=x.drop(columns=["Age Co-Lead"])

xn = inputNormalization(x,x)[0]

model=skl_nb.KNeighborsClassifier(n_neighbors = 8)
cv = skl_ms.KFold(n_splits=n_fold, random_state=1, shuffle=True)

for i, (train_index, val_index) in enumerate(cv.split(x)):

    X_train, X_val = xn.iloc[train_index], xn.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    model.fit(X_train, y_train)
    prediction = model.predict(X_val)
    missclassification[i,3] = np.mean(prediction == y_val)
    f1[i,3] = skl_m.f1_score(prediction,y_val,pos_label='Female')
    precision[i,3] = skl_m.precision_score(prediction,y_val,pos_label='Female')
    recall[i,3] = skl_m.recall_score(prediction,y_val,pos_label='Female')


#====================================================#
#   PLOTTING RESULTS                                 #
#====================================================#

plt.boxplot(missclassification)
plt.title('Cross validation accuracy for different Methods')
plt.xticks(np.arange(len(models)+1)+1, ['Random Forest Classifier', 'Gradient Boosting','Bagging','knn'])
plt.show()
plt.boxplot(f1)
plt.title('F1 score for different Methods')
plt.xticks(np.arange(len(models)+1)+1, ['Random Forest Classifier', 'Gradient Boosting','Bagging','knn'])
plt.show()
plt.boxplot(recall)
plt.title('Recall score for different Methods')
plt.xticks(np.arange(len(models)+1)+1, ['Random Forest Classifier', 'Gradient Boosting','Bagging','knn'])
plt.show()
plt.boxplot(precision)
plt.title('Precision for different Methods')
plt.xticks(np.arange(len(models)+1)+1, ['Random Forest Classifier', 'Gradient Boosting','Bagging','knn'])
plt.show()

#====================================================#
#   OUR PREDICTIONS FOR TEST DATA                    #
#====================================================#

model = BaggingClassifier(estimator=skl_da.QuadraticDiscriminantAnalysis(), n_estimators=400)

data = pd.read_csv('train.csv') 
x_test = pd.read_csv('test.csv') 

x_train = data.drop(columns=["Lead"])
y_train = data["Lead"]

x_train = InputSelection(x_train)
x_test = InputSelection(x_test)

model.fit(x_train,y)

prediction = model.predict(x_test)
prediction.to_csv('predictions.csv', index=False)






