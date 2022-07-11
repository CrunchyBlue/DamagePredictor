---
title: "Ensemble Implementation"
author: "David Gagliardi"
date: "07/11/2022"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Adjustments to Allow Multiple Processing
```{r}
library(reticulate)
use_python("C:/Users/dgags/Anaconda3/python.exe")
# update executable path in sys module
sys <- import("sys")
exe <- file.path(sys$exec_prefix, "pythonw.exe")
sys$executable <- exe
sys$`_base_executable` <- exe
# update executable path in multiprocessing module
multiprocessing <- import("multiprocessing")
multiprocessing$set_executable(exe)
```

## Main Imports
```{python}
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier
```

## Retrieve and Manipulate Initial Data
```{python}
X_train = pd.read_csv("train_values.csv")
X_test = pd.read_csv("test_values.csv")
y_train = pd.read_csv("train_labels.csv")
```

```{python}
X_train = X_train.drop("building_id", axis=1)
X_test = X_test.drop("building_id", axis=1)
y_train = y_train.drop("building_id", axis=1)
```

```{python}
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
```

## Determine and Remove Outliers
```{python}
from sklearn.ensemble import IsolationForest
```

```{python}
iso = IsolationForest(n_estimators=200, random_state=42, warm_start=True, contamination=0.01, n_jobs=-1)
iso.fit(X_train)
```

```{python}
y_preds_train = iso.predict(X_train)
```

```{python}
y_preds_train = y_preds_train.tolist()
```

```{python}
X_train = pd.DataFrame(data=X_train)
y_train = pd.DataFrame(data=y_train)
```

```{python}
X_train['Outlier'] = y_preds_train
y_train['Outlier'] = y_preds_train
```

```{python}
X_train = X_train[X_train.Outlier != -1]
y_train = y_train[y_train.Outlier != -1]
```

```{python}
X_train = X_train.drop("Outlier", axis=1)
y_train = y_train.drop("Outlier", axis=1)
```

## Normalize Data
```{python}
scaler = MinMaxScaler().fit(X_train)
  
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

## Train Test Split for Testing
```{python}
X = X_train
y = y_train
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

## Extreme Gradient Boosting
```{python}
xgb1 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.55, gamma=0.37,
              learning_rate=0.1, max_delta_step=0, max_depth=13,
              min_child_weight=0.1, missing=None, n_estimators=324, n_jobs=-1,
              nthread=8, num_classes=3, objective='multi:softprob',
              random_state=0, reg_alpha=3, reg_lambda=1, scale_pos_weight=1,
              seed=42, silent=None, subsample=0.9, verbosity=1)
```

```{python}
xgb1 = xgb1.fit(X_train, y_train.values.ravel())
```

```{python}
in_sample_preds = xgb1.predict(X_test)
```

```{python}
f1_score(y_test, in_sample_preds, average='micro')
```

## K Nearest Neighbors
```{python}
knn = KNeighborsClassifier(algorithm='auto', leaf_size=2, metric='minkowski',
                     metric_params=None, n_jobs=-1, n_neighbors=8, p=1,
                     weights='uniform')
```

```{python}
knn = knn.fit(X_train, y_train.values.ravel())
```

```{python}
in_sample_preds = knn.predict(X_test)
```

```{python}
f1_score(y_test, in_sample_preds, average='micro')
```

## Bagging Ensemble of KNN
```{python}
knn_bag = BaggingClassifier(base_estimator=knn,
                  bootstrap=False, bootstrap_features=False, max_features=0.5,
                  max_samples=1.0, n_estimators=15, n_jobs=-1, oob_score=False,
                  verbose=0, warm_start=False)
```

```{python}
knn_bag = knn_bag.fit(X_train, y_train.values.ravel())
```

```{python}
in_sample_preds = knn_bag.predict(X_test)
```

```{python}
f1_score(y_test, in_sample_preds, average='micro')
```

## XGB Hypertuning
# This finds the optimal number of trees through early stopping callbacks.
```{python}
eval_set = [(X_test, y_test)]
xgb1.fit(X_train, y_train.values.ravel(), early_stopping_rounds=50, eval_metric="mlogloss", eval_set=eval_set, verbose=True)
```

```{python}
features = xgb1.feature_importances_
```

# These are example parameters, I chose to break this apart into several runs to speed up the process.
```{python}
XGB_grid = {'max_depth': [7, 9, 11, 13, 15, 17, 19], 'min_child_weight': [0.1, 1, 10], 'colsample_bytree': [0.50, 0.55, 0.6], 'subsample':[0.85, 0.9, 0.95]}
```

```{python}
XGB_grid_search = GridSearchCV(estimator =  xgb1, param_grid=XGB_grid, n_jobs=-1, cv=3)
```

```{python}
XGB_grid_search.fit(X_train, y_train.values.ravel())
```

```{python}
XGB_grid_search.best_estimator_
```

## KNN Hypertuning
# These are example parameters, I chose to break this apart into several runs to speed up the process.
```{python}
knn_grid = {'leaf_size': [2, 4, 6, 8, 10], 'n_neighbors': [4, 6, 8, 10, 12]}
```

```{python}
knn_gridCV = GridSearchCV(estimator =  knn, param_grid=knn_grid, n_jobs=-1, cv=3)
```

```{python}
knn_gridCV = knn_gridCV.fit(X_train, y_train.values.ravel())
```

```{python}
knn_gridCV.best_estimator_
```

## Bagged KNN Hypertuning
# These are example parameters, I chose to break this apart into several runs to speed up the process.
```{python}
bag_grid = {'max_sample': [0.5, 0.8, 1], 'max_feature': [0.5, 0.8, 1], 'bootstrap': [False, True], 'bootstrap_features': [False, True]}
```

```{python}
bag_gridCV = GridSearchCV(estimator =  knn_bag, param_grid=bag_grid, n_jobs=-1, cv=3)
```

```{python}
bag_gridCV = bag_gridCV.fit(X_train, y_train.values.ravel())
```

```{python}
bag_gridCV.best_estimator_
```

## Creation of the Stacked Dataset (Be sure to prep data without train_test_split for this)
```{python}
xgb_probs = cross_val_predict(xgb1, X_train, y_train.values.ravel(), method='predict_proba', cv=5, n_jobs=-1)
```

```{python}
knn_probs = cross_val_predict(knn_bag, X_train, y_train.values.ravel(), method='predict_proba', cv=5, n_jobs=-1)
```

```{python}
xgb_df = pd.DataFrame(data=xgb_probs)
knn_df = pd.DataFrame(data=knn_probs)
```

```{python}
stacked_df = pd.concat([xgb_df, knn_df], axis=1)
```

```{python}
stacked_df.to_csv(r'C:\Users\dgags\Desktop\StackedTrainingData.csv', index = False, header=True)
```

## Creation of the Stacked Test Dataset (Be sure to prep data without train_test_split for this)
```{python}
xgb_preds = xgb1.predict_proba(X_test)
knn_preds = knn_bag.predict_proba(X_test)
```

```{python}
xgb_test_df = pd.DataFrame(data=xgb_preds)
knn_test_df = pd.DataFrame(data=knn_preds)
```

```{python}
stacked_test_df = pd.concat([xgb_test_df, knn_test_df], axis=1)
```

```{python}
stacked_test_df.to_csv(r'C:\Users\dgags\Desktop\StackedTestData.csv', index = False, header=True)
```

## Stacked Generalization Classifier
```{python}
S_train = pd.read_csv("StackedTrainingData.csv")
S_test = pd.read_csv("StackedTestData.csv")
```

# Split for use of Micro F1 Score (Don't use for final predictions)
```{python}
X = S_train
y = y_train
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

```{python}
ada = AdaBoostClassifier(algorithm='SAMME.R',
                   learning_rate=1.0, n_estimators=800)
```

```{python}
ada = ada.fit(X_train, y_train.values.ravel())
```

```{python}
in_sample_preds = ada.predict(X_test)
```

```{python}
f1_score(y_test, in_sample_preds, average='micro')
```

## Import Submission File and Output Predictions

```{python}
ada = ada.fit(S_train, y_train.values.ravel())
```

```{python}
predictions = ada.predict(S_test)
```

```{python}
submission = pd.read_csv("submission_format.csv")
```

```{python}
submission["damage_grade"] = predictions
```

```{python}
submission.to_csv(r'C:\Users\dgags\Desktop\CompetitionSubmission.csv', index = False, header=True)
```

## **Below Not Included in Final Model**
## Side Imports and Testing of Models Not Included in the Final Ensemble
```{python}
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import lightgbm as lbm
from lightgbm.sklearn import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
```

## LightGBM
```{python}
lgbm = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=7,
               min_child_samples=20, min_child_weight=0.1, min_split_gain=0.0,
               n_estimators=1251, n_jobs=-1, num_leaves=80, objective=None,
               random_state=None, reg_alpha=2, reg_lambda=0.0, silent=True,
               subsample=1.0, subsample_for_bin=232000, subsample_freq=0)
```

```{python}
lgbm = lgbm.fit(X_train, y_train.values.ravel())
```

```{python}
in_sample_preds = lgbm.predict(X_test)
```

```{python}
f1_score(y_test, in_sample_preds, average='micro')
```

```{python}
confusion_matrix(y_test, in_sample_preds)
```

```{python}
predictions = cross_val_predict(lgbm, X_train, y_train.values.ravel(), cv=5, n_jobs=-1)
probs = cross_val_predict(lgbm, X_train, y_train.values.ravel(), method='predict_proba', cv=5, n_jobs=-1)
```

```{python}
print("accuracy",metrics.accuracy_score(y_train.values.ravel(), predictions))
print("f1 score macro",metrics.f1_score(y_train.values.ravel(), predictions, average='macro'))
print("f1 score micro",metrics.f1_score(y_train.values.ravel(), predictions, average='micro'))
print("precision score",metrics.precision_score(y_train.values.ravel(), predictions, average='macro'))
print("recall score",metrics.recall_score(y_train.values.ravel(), predictions, average='macro'))
print("hamming_loss",metrics.hamming_loss(y_train.values.ravel(), predictions))
print("classification_report", metrics.classification_report(y_train.values.ravel(), predictions))
print("jaccard_similarity_score", metrics.jaccard_score(y_train.values.ravel(), predictions, average='macro'))
print("log_loss", metrics.log_loss(y_train.values.ravel(), probs))
print("zero_one_loss", metrics.zero_one_loss(y_train.values.ravel(), predictions))
print("AUC&ROC",metrics.roc_auc_score(y_train.values.ravel(), probs, average='macro', multi_class='ovo'))
print("matthews_corrcoef", metrics.matthews_corrcoef(y_train.values.ravel(), predictions))
```

## AdaBoosted Decision Stump
```{python}
ada = AdaBoostClassifier(algorithm='SAMME.R',
                   learning_rate=1.0, n_estimators=500)
```

```{python}
ada = ada.fit(X_train, y_train.values.ravel())
```

```{r}
py$X_test
```

```{python}
X_stump = ada.predict_proba(X_test)
```

```{python}
in_sample_preds = ada.predict(X_test)
```

```{python}
f1_score(y_test, in_sample_preds, average='micro')
```

```{python}
confusion_matrix(y_test, in_sample_preds)
```

```{python}
predictions = cross_val_predict(ada, X_train, y_train.values.ravel(), cv=5, n_jobs=-1)
probs = cross_val_predict(ada, X_train, y_train.values.ravel(), method='predict_proba', cv=5, n_jobs=-1)
```

```{python}
print("accuracy",metrics.accuracy_score(y_train.values.ravel(), predictions))
print("f1 score macro",metrics.f1_score(y_train.values.ravel(), predictions, average='macro'))
print("f1 score micro",metrics.f1_score(y_train.values.ravel(), predictions, average='micro'))
print("precision score",metrics.precision_score(y_train.values.ravel(), predictions, average='macro'))
print("recall score",metrics.recall_score(y_train.values.ravel(), predictions, average='macro'))
print("hamming_loss",metrics.hamming_loss(y_train.values.ravel(), predictions))
print("classification_report", metrics.classification_report(y_train.values.ravel(), predictions))
print("jaccard_similarity_score", metrics.jaccard_score(y_train.values.ravel(), predictions, average='macro'))
print("log_loss", metrics.log_loss(y_train.values.ravel(), probs))
print("zero_one_loss", metrics.zero_one_loss(y_train.values.ravel(), predictions))
print("AUC&ROC",metrics.roc_auc_score(y_train.values.ravel(), probs, average='macro', multi_class='ovo'))
print("matthews_corrcoef", metrics.matthews_corrcoef(y_train.values.ravel(), predictions))
```

## Random Forest
```{python}
rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=70, max_features=3,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=12,
                       min_weight_fraction_leaf=0.0, n_estimators=50,
                       n_jobs=-1, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)
```

```{python}
rf = rf.fit(X_train, y_train.values.ravel())
```

```{python}
in_sample_preds = rf.predict(X_test)
```

```{python}
f1_score(y_test, in_sample_preds, average='micro')
```

```{python}
confusion_matrix(y_test, in_sample_preds)
```

```{python}
predictions = cross_val_predict(rf, X_train, y_train.values.ravel(), cv=5, n_jobs=-1)
probs = cross_val_predict(rf, X_train, y_train.values.ravel(), method='predict_proba', cv=5, n_jobs=-1)
```

```{python}
print("accuracy",metrics.accuracy_score(y_train.values.ravel(), predictions))
print("f1 score macro",metrics.f1_score(y_train.values.ravel(), predictions, average='macro'))
print("f1 score micro",metrics.f1_score(y_train.values.ravel(), predictions, average='micro'))
print("precision score",metrics.precision_score(y_train.values.ravel(), predictions, average='macro'))
print("recall score",metrics.recall_score(y_train.values.ravel(), predictions, average='macro'))
print("hamming_loss",metrics.hamming_loss(y_train.values.ravel(), predictions))
print("classification_report", metrics.classification_report(y_train.values.ravel(), predictions))
print("jaccard_similarity_score", metrics.jaccard_score(y_train.values.ravel(), predictions, average='macro'))
print("log_loss", metrics.log_loss(y_train.values.ravel(), probs))
print("zero_one_loss", metrics.zero_one_loss(y_train.values.ravel(), predictions))
print("AUC&ROC",metrics.roc_auc_score(y_train.values.ravel(), probs, average='macro', multi_class='ovo'))
print("matthews_corrcoef", metrics.matthews_corrcoef(y_train.values.ravel(), predictions))
```

## Support Vector Classifier Radial Basis Function
```{python}
rbf = SVC(C=3, break_ties=False, cache_size=1024,
                                     class_weight=None, coef0=0.0,
                                     decision_function_shape='ovr', degree=3,
                                     gamma='scale', kernel='rbf', max_iter=-1,
                                     probability=True, random_state=None,
                                     shrinking=True, tol=0.001, verbose=False)
```

## Support Vector Classifier Linear
```{python}
lin = LinearSVC(C=10, class_weight=None, dual=True,
                                           fit_intercept=True,
                                           intercept_scaling=1,
                                           loss='squared_hinge', max_iter=1000,
                                           multi_class='ovr', penalty='l2',
                                           random_state=None, tol=0.0001,
                                           verbose=0)
```

## Bagging Ensemble of RBF
```{python}
rbf_bag = BaggingClassifier(base_estimator=rbf,
                  bootstrap=False, bootstrap_features=False, max_features=1.0,
                  max_samples=0.01, n_estimators=20, n_jobs=-1, oob_score=False,
                  random_state=42, verbose=0, warm_start=False)
```

```{python}
rbf_bag = rbf_bag.fit(X_train, y_train.values.ravel())
```

```{python}
in_sample_preds = rbf_bag.predict(X_test)
```

```{python}
f1_score(y_test, in_sample_preds, average='micro')
```

```{python}
confusion_matrix(y_test, in_sample_preds)
```

## Bagging Ensemble of Lin
```{python}
lin_bag = BaggingClassifier(base_estimator=lin,
                  bootstrap=False, bootstrap_features=False, max_features=1.0,
                  max_samples=0.05, n_estimators=10, n_jobs=-1, oob_score=False,
                  random_state=42, verbose=0, warm_start=False)
```

```{python}
lin_bag = lin_bag.fit(X_train, y_train.values.ravel())
```

```{python}
in_sample_preds = lin_bag.predict(X_test)
```

```{python}
f1_score(y_test, in_sample_preds, average='micro')
```

```{python}
confusion_matrix(y_test, in_sample_preds)
```

## Neural Network
```{python}
nn = MLPClassifier(activation='tanh', alpha=0.0001, batch_size=512, beta_1=0.9,
              beta_2=0.999, early_stopping=True, epsilon=1e-08,
              hidden_layer_sizes=(48, 36, 24, 12), learning_rate='adaptive',
              learning_rate_init=0.001, max_fun=15000, max_iter=500,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=None, shuffle=True, solver='adam',
              tol=0.00001, validation_fraction=0.2, verbose=False,
              warm_start=False)
```

```{python}
nn.fit(X_train, y_train.values.ravel())
```

```{python}
in_sample_preds = nn_bag.predict(X_test)
```

```{python}
f1_score(y_test, in_sample_preds, average='micro')
```

```{python}
nn_bag = BaggingClassifier(base_estimator=nn,
                  bootstrap=True, bootstrap_features=False, max_features=1.0,
                  max_samples=1.0, n_estimators=25, n_jobs=-1, oob_score=False,
                  verbose=0, warm_start=False)
```

```{python}
nn_bag_probs = cross_val_predict(nn_bag, X_train, y_train.values.ravel(), method='predict_proba', cv=5, n_jobs=-1)
```

```{python}
nn_bag_probs
```

```{python}
nn_bag_testdf
```

```{python}
nn_bag_testdf.to_csv(r'C:\Users\dgags\Desktop\NN_Bag_Test.csv', index = False, header=True)
```

```{python}
nn_bag = nn_bag.fit(X_train, y_train.values.ravel())
```

```{python}
nn_bag_test = nn_bag.predict_proba(X_test)
```

```{python}
nn_bag = nn_bag.fit(X_train, y_train.values.ravel())
```

```{python}
confusion_matrix(y_test, in_sample_preds)
```

```{python}
predictions = cross_val_predict(nn, X_train, y_train.values.ravel(), cv=5, n_jobs=-1)
probs = cross_val_predict(nn, X_train, y_train.values.ravel(), method='predict_proba', cv=5, n_jobs=-1)
```

```{python}
print("accuracy",metrics.accuracy_score(y_train.values.ravel(), predictions))
print("f1 score macro",metrics.f1_score(y_train.values.ravel(), predictions, average='macro'))
print("f1 score micro",metrics.f1_score(y_train.values.ravel(), predictions, average='micro'))
print("precision score",metrics.precision_score(y_train.values.ravel(), predictions, average='macro'))
print("recall score",metrics.recall_score(y_train.values.ravel(), predictions, average='macro'))
print("hamming_loss",metrics.hamming_loss(y_train.values.ravel(), predictions))
print("classification_report", metrics.classification_report(y_train.values.ravel(), predictions))
print("jaccard_similarity_score", metrics.jaccard_score(y_train.values.ravel(), predictions, average='macro'))
print("log_loss", metrics.log_loss(y_train.values.ravel(), probs))
print("zero_one_loss", metrics.zero_one_loss(y_train.values.ravel(), predictions))
print("AUC&ROC",metrics.roc_auc_score(y_train.values.ravel(), probs, average='macro', multi_class='ovo'))
print("matthews_corrcoef", metrics.matthews_corrcoef(y_train.values.ravel(), predictions))
```

## RF Hypertuning
```{python}
rf_grid = {'max_depth': [65, 70, 75]}
```

```{python}
rf_gridCV = GridSearchCV(estimator =  rf, param_grid=rf_grid, n_jobs=-1, cv=3)
```

```{python}
rf_gridCV = rf_gridCV.fit(X_train, y_train.values.ravel())
```

```{python}
rf_gridCV.best_estimator_
```

## RBF Hypertuning Using 10% of Training Data
```{python}
svc1 = SVC(C=0.001, break_ties=False, cache_size=2000, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, shrinking=True,
    tol=0.001, verbose=False)
    
svc2 = SVC(C=0.01, break_ties=False, cache_size=2000, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, shrinking=True,
    tol=0.001, verbose=False)
svc3 = SVC(C=0.1, break_ties=False, cache_size=2000, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, shrinking=True,
    tol=0.001, verbose=False)
    
svc4 = SVC(C=10, break_ties=False, cache_size=2000, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, shrinking=True,
    tol=0.001, verbose=False)
    
svc5 = SVC(C=100, break_ties=False, cache_size=2000, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, shrinking=True,
    tol=0.001, verbose=False)
    
svc6 = SVC(C=1000, break_ties=False, cache_size=2000, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, shrinking=True,
    tol=0.001, verbose=False)
    
svc7 = SVC(C=1.0, break_ties=False, cache_size=2000, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, shrinking=True,
    tol=0.001, verbose=False)
```

```{python}
svc_grid = {'base_estimator': [svc1, svc2, svc3, svc4, svc5, svc6, svc7]}
```

```{python}
svc_gridCV = GridSearchCV(estimator = rbf_bag, param_grid=svc_grid, n_jobs=-1, cv=3)
```

```{python}
svc_gridCV = svc_gridCV.fit(X_train, y_train.values.ravel())
```

```{python}
svc_gridCV.best_estimator_
```

## Lin Hypertuning Using 10% of Training Data
```{python}
svc1 = LinearSVC(C=0.001, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
    
svc2 = LinearSVC(C=0.01, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
    
svc3 = LinearSVC(C=0.1, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
    
svc4 = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
    
svc5 = LinearSVC(C=10, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
    
svc6 = LinearSVC(C=100, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
    
svc7 = LinearSVC(C=1000, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
```

```{python}
svc_grid = {'base_estimator': [svc1, svc2, svc3, svc4, svc5, svc6, svc7]}
```

```{python}
svc_gridCV = GridSearchCV(estimator = lin_bag, param_grid=svc_grid, n_jobs=-1, cv=3)
```

```{python}
svc_gridCV = svc_gridCV.fit(X_train, y_train.values.ravel())
```

```{python}
svc_gridCV.best_estimator_
```

##Hypertuning LightGBM

```{python}
eval_set = [(X_test, y_test.values.ravel())]
lgbm.fit(X_train, y_train.values.ravel(), early_stopping_rounds=200, eval_metric="logloss", eval_set=eval_set, verbose=True)
```

```{python}
lgbm.best_iteration_
```

```{python}
lgbm_grid = {'min_child_weight': [1.0, 0.1, 0.001, 0.0001]}
```

```{python}
lgbm_grid_search = GridSearchCV(estimator = lgbm, param_grid=lgbm_grid, n_jobs=-1, cv=5)
```

```{python}
lgbm_grid_search.fit(X_train, y_train.values.ravel())
```

```{python}
lgbm_grid_search.best_estimator_
```