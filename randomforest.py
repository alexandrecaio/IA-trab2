import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree



df_wine = pd.read_csv('dataset1000.csv', header=0)
df_wine.head()


def makeAcuracy(tree,x_test,y_test):
    predictions = clf.predict(x_test)
    erro = 0.0
    for x in range(len(predictions)):
        if predictions[x] != y_test[x]:
            erro += 1.
    acuracy = (1-(erro/len(predictions)))
    return acuracy


    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = 
        train_test_split(X, y,test_size=0.3, random_state=2)
#Carregamos a classe LabelEncoder
enc = LabelEncoder()

#Agora associamos isso a coluna correspondente
for val in range(0,75):
    X[:,val] = enc.fit_transform(X[:,val])

# print(X)
clf = RandomForestClassifier(n_estimators=10000,
                                criterion='entropy',
                                max_features='sqrt',
                                random_state=0,
                                n_jobs=-1)

clf.fit(X_train, y_train)
# clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=76,
#                                    min_samples_leaf=5)
# clf = clf.fit(X_train,y_train)

# makeAcuracy(clf,X_test,y_test)


feat_labels = df_wine.columns[1:]

importances = clf.feature_importances_

indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        color='lightblue', 
        align='center')

plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()