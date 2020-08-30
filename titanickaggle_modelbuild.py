# -*- coding: utf-8 -*-
"""TitanicKaggle_ModelBuild.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yyZsIQJzdjaQbRTrlo9sY2lqCMhfFmcT
"""

import pandas as pd
train=pd.read_csv('/content/drive/My Drive/Colab Notebooks/Competitions/Titanic Kaggle/train_clean.csv')

import seaborn as sns 
import matplotlib.pyplot as plt

train.head()

passenger_details=train[['Name','Ticket','PassengerId']]
train.drop(['Name','Ticket','PassengerId'],axis=1,inplace=True)

train.head()

y=train['Survived']
x=train.drop('Survived',axis=1)

x.columns

#Selecting feautres
features=['Age', 'SibSp', 'Parch', 'Fare', 'Family_Size', 'Superior_Cabins',
       'male', '2', '3', 'Q', 'S', 'Miss', 'Mr', 'Mrs', 'Other', 'B', 'C', 'D',
       'E', 'F', 'G', 'T', 'U']

x=train[features]

from sklearn.model_selection import train_test_split
x_train,x_valid,y_train,y_valid=train_test_split(x,y,test_size=0.1,random_state=123)

from sklearn.preprocessing import StandardScaler
mms=StandardScaler()
x_scaled_train=mms.fit_transform(x_train)
x_scaled_valid=mms.transform(x_valid)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import KFold,cross_val_score

kf5=KFold(n_splits=5)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models=[]
models.append(("LogReg",LogisticRegression()))
models.append(("DesicionTree",DecisionTreeClassifier()))
models.append(("SVC",SVC()))
models.append(("kNN",KNeighborsClassifier()))
models.append(("RandomForest",RandomForestClassifier()))

names=[]
result=[]
scoring='accuracy'
for name,model in models:
  model.fit(x_scaled_train,y_train)
  print("Model created:",name, "------>Cross Validation Error:" ,cross_val_score(model,x_scaled_train,y_train,scoring=scoring).mean())
  print("Model created:",name, "------>Validation Error:" ,accuracy_score(y_valid,model.predict(x_scaled_valid)))
  print("-------------------------------------------------------------------")

# LogReg and SVC , KNN looks to do well here 
# Lets try optimising hyperparameters

from sklearn.model_selection import GridSearchCV
import numpy as np

"""# Logistic Regression"""

log_params={'tol':[0.01,0.001,0.0001,0.00001],
            'max_iter':np.linspace(100,500,10),
            'class_weight':['None','balanced']
          }

grid_log=GridSearchCV(estimator=LogisticRegression(n_jobs=-1),
                          param_grid=log_params)

grid_log.fit(x_scaled_train,y_train)

grid_log.best_estimator_

grid_log.best_score_

print(classification_report(y_valid,grid_log.predict(x_scaled_valid)))

sns.heatmap(confusion_matrix(y_valid,grid_log.predict(x_scaled_valid)),annot=True,center=True)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

print(accuracy_score(y_valid,grid_log.predict(x_scaled_valid)))

"""# KNN Model"""

knn_params={'n_neighbors':[5,10,15,40],
            'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
            'weights':['uniform', 'distance'],
            'metric' :['euclidean', 'manhattan', 'minkowski'],
            'leaf_size':[20,30,40,50],
            }

grid_knn=GridSearchCV(KNeighborsClassifier(),knn_params)

grid_knn.fit(x_scaled_train,y_train)

grid_knn.best_estimator_

grid_knn.best_score_

print(classification_report(y_valid,grid_knn.predict(x_scaled_valid)))

print(accuracy_score(y_valid,grid_knn.predict(x_scaled_valid)))

sns.heatmap(confusion_matrix(y_valid,grid_knn.predict(x_scaled_valid)),annot=True,center=True)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

"""# SVC"""

svc_params={#'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            'tol':[0.01,0.001,0.0001],
            'C':[1,10,100]
            }

svc_grid=GridSearchCV(SVC(),param_grid=svc_params,n_jobs=-1)

svc_grid.fit(x_scaled_train,y_train)

svc_grid.best_estimator_

print(classification_report(y_valid,svc_grid.predict(x_scaled_valid)))

print(accuracy_score(y_valid,svc_grid.predict(x_scaled_valid)))

sns.heatmap(confusion_matrix(y_valid,svc_grid.predict(x_scaled_valid)),annot=True,center=True)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

"""# Random Forest"""

rf_params={ 'n_estimators':[50,100,250,500],
           'max_depth':[5,10,20,32],
           'min_samples_split':[2,5,15,25],
           'min_samples_leaf':[1,5,15,20],
           'max_features':['sqrt','log2','auto'],
           'criterion':['gini','entropy']
}

rf_grid=GridSearchCV(RandomForestClassifier(),param_grid=rf_params)

rf_grid.fit(x_scaled_train,y_train)

rf_grid.best_estimator_

rf_grid.best_score_

print(accuracy_score(y_valid,rf_grid.predict(x_scaled_valid)))

print(classification_report(y_valid,rf_grid.predict(x_scaled_valid)))

sns.heatmap(confusion_matrix(y_valid,rf_grid.predict(x_scaled_valid)),annot=True)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

"""# CatBoost"""

pip install catboost

from catboost import CatBoostClassifier
cat=CatBoostClassifier()

cat.fit(x_train,y_train)

sns.heatmap(confusion_matrix(y_valid,cat.predict(x_valid)),annot=True)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

print(accuracy_score(y_valid,cat.predict(x_valid)))

"""# Building final model on test dataset"""

test=pd.read_csv('/content/drive/My Drive/Colab Notebooks/Competitions/Titanic Kaggle/test_clean.csv')

passenger_details_test=test[['Name','Ticket','PassengerId']]
test.drop(['Name','Ticket','PassengerId'],axis=1,inplace=True)

test.columns

x.columns

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

test_scaled=scaler.transform(test)

rf_grid.best_estimator_

rf_final=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=32, max_features='log2',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=15,
                       min_weight_fraction_leaf=0.0, n_estimators=50,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

rf_final.fit(x,y)

predictions=rf_final.predict(test)

result=pd.DataFrame()
result['PassengerId']=passenger_details_test['PassengerId']
result['Survived']=predictions

