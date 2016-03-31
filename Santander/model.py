import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn import preprocessing
from sklearn import cross_validation as cv
from sklearn.cross_validation import train_test_split
from itertools import combinations

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import linear_model as lm
import matplotlib
matplotlib.use("Agg")

# Separating variables and classification vars
train = pd.read_csv('train.csv', header=0)
train_vars = train.drop(["ID", "TARGET"], axis=1)
train_class = train["TARGET"]
test = pd.read_csv('test.csv', header=0)
test_ID = test["ID"]
test_vars = test.drop(["ID"], axis=1)
print train_class.value_counts()

# Scaling
scale = preprocessing.StandardScaler().fit(train_vars)

train_scaled = scale.transform(train_vars)
test_scaled = scale.transform(test_vars)

train_dataset = train_vars

# Finding nulls in the dataset
nulls_train = (train_dataset.isnull().sum() == 1).sum()
print "There are {} nulls in this dataset" .format(nulls_train)

# Identifying constant features


def identify_constant_features(dataframe):
    count_uniques = dataframe.apply(lambda x: len(x.unique()))
    constants = count_uniques[count_uniques == 1].index.tolist()
    return constants

# Feature Selection
constant_features_train = set(identify_constant_features(train_dataset))
print('There were {} constant features in TRAIN dataset.'.format(
        len(constant_features_train)))

train_dataset.drop(constant_features_train, inplace=True, axis=1)
test_vars.drop(constant_features_train, inplace=True, axis=1)

# Identifying equal features


def identify_equal_features(dataframe):
    features_to_compare = list(combinations(dataframe.columns.tolist(), 2))
    equal_features = []
    for compare in features_to_compare:
        is_equal = np.array_equal(dataframe[compare[0]], dataframe[compare[1]])
        if is_equal:
            equal_features.append(list(compare))
    return equal_features

equal_features_train = identify_equal_features(train_dataset)

print('There were {} pairs of equal features in TRAIN dataset.'.format(len(equal_features_train)))

features_to_drop = np.array(equal_features_train)[:, 1]
train_dataset.drop(features_to_drop, axis=1, inplace=True)
test_vars.drop(features_to_drop, axis=1, inplace=True)

skf = cv.StratifiedKFold(train_class, n_folds=3, shuffle=True)
score_metric = 'roc_auc'
scores = {}
#
#
def score_model(model):
    return cv.cross_val_score(model, train_vars, train_class, cv=skf, scoring=score_metric)
# #
# # Training Classifiers
# #
scores["RF"] = score_model(RandomForestClassifier())
print "RF done"
scores["Grad_Boost"] = score_model(GradientBoostingClassifier(n_estimators=100))
print "Grad Boost Done"
scores["Adaboost"] = score_model(AdaBoostClassifier(n_estimators=100))
print "Ada Boost Done"
# #
# #
model_scores = pd.DataFrame(scores).mean()
print model_scores

# gb = GradientBoostingClassifier()
# gb = gb.fit(train_vars, train_class)
#
# pred = gb.predict_proba(test_vars)[:, 1]
Preds = {}

clf_xgb = xgb.XGBClassifier(missing=np.nan, max_depth=3, min_child_weight=1, gamma=0, n_estimators=1400, learning_rate=0.05, nthread=4, subsample=0.90, colsample_bytree=0.85, scale_pos_weight=1, seed=4242)

X_train, X_test, y_train, y_test = train_test_split(train_vars, train_class, test_size=0.28, random_state=42)

clf_xgb.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc", eval_set=[(X_test, y_test)])

clf_1 = RandomForestClassifier(n_estimators=250)
clf_2 = GradientBoostingClassifier(n_estimators=250)
clf_3 = AdaBoostClassifier(n_estimators=250)


def predictions(xtrain, ytrain, xtest):
    clf = clf_1.fit(xtrain, ytrain)
    Preds["RF"] = clf.predict_proba(xtest)[:, 1]
    clf = clf_2.fit(xtrain, ytrain)
    Preds["G_Boost"] = clf.predict_proba(xtest)[:, 1]
    clf = clf_3.fit(xtrain, ytrain)
    Preds["Ada_Boost"] = clf.predict_proba(xtest)[:, 1]
    Preds["XG_Boost"] = clf_xgb.predict_proba(xtest)[:, 1]
    return pd.DataFrame(Preds)

Preds_1 = predictions(X_train, y_train, X_test)
print Preds_1.head()

print "RF = ", roc_auc_score(y_test, Preds_1["RF"])
print "G_Boost AUC = ", roc_auc_score(y_test, Preds_1["G_Boost"])
print "Ada_Boost = ", roc_auc_score(y_test, Preds_1["Ada_Boost"])

combi_mod = lm.LogisticRegression(C=1e11)
combi_mod = combi_mod.fit(Preds_1, y_test)
#
pred = combi_mod.predict_proba(Preds_1)[:, 1]
#
print "Combi_Model AUC = ", roc_auc_score(y_test, pred)

Preds_2 = predictions(train_vars, train_class, test_vars)
y_pred = combi_mod.predict_proba(Preds_2)[:, 1]

ada_pred = clf_xgb.predict_proba(test_vars)[:, 1]
submission = pd.DataFrame({
           "ID": test_ID,
           "TARGET": ada_pred
       })
#
submission.to_csv("sub_2.csv", index=False)
print "submission 1 written!"
#
submission = pd.DataFrame({
           "ID": test_ID,
           "TARGET": y_pred
       })
#
submission.to_csv("sub_3.csv", index=False)
print "submission 2 written!"
