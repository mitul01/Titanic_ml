#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator,TransformerMixin

# the custom scaler class 
class CustomScaler(BaseEstimator,TransformerMixin): 
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.array(np.mean(X[self.columns]))
        self.var_ = np.array(np.var(X[self.columns]))
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]
    
class titanic_model():
      
        def __init__(self, model_file, scaler_file):
            # read the 'model' and 'scaler' files which were saved
            with open('model2','rb') as model_file, open('scaler2', 'rb') as scaler_file:
                self.logic= pickle.load(model_file)
                self.scaler2 = pickle.load(scaler_file)
                self.data = None
        
        def load_and_clean_data(self, data_file):
            
            # import the data
            df = pd.read_csv(data_file,delimiter=',')
            # store the data in a new variable for later use
            self.df_copy=df.copy()
            # Clean the data
            #Drop uncessary Data
            df=df.drop(['Name','Cabin','Ticket'],axis=1)
            #Fill NA Values
            df['Age']=df['Age'].fillna(df['Age'].mean())   
            df['Embarked']=df['Embarked'].fillna(value='D')
            df=df.fillna(value=0)
            #PassengerID
            passengerID=df.iloc[:,0]
            df=df.drop('PassengerId',axis=1)
            #Pclass
            Pclass_dummies=pd.get_dummies(df['Pclass'])
            Pclass_dummies=Pclass_dummies.rename(columns={1:'Pclass1',2:'Pclass2',3:'Pclass3'})
            df=df.drop('Pclass',axis=1)
            df=pd.concat([df,Pclass_dummies],axis=1)
            #Age
            df['Sex']=df['Sex'].map({'male':1,'female':0})
            #Embarked
            embarked_dummies=pd.get_dummies(df['Embarked'])
            df=pd.concat([df,embarked_dummies],axis=1)
            df=df.drop('Embarked',axis=1)
            #Save preprocessed data
            self.preprocessed_data = df.copy()
            self.preprocessed_data['PassengerID']=passengerID
            #Scale the data
            self.data = self.scaler2.transform(df)
            return self.data
        
        def add_missing_dummy_columns(self):
            columns=(['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Pclass1', 'Pclass2', 'Pclass3',
                        'C', 'D', 'Q', 'S'])   
            missing_cols = set(columns) - set(self.data.columns)
            for c in missing_cols:
                self.data[c] = 0
            return self.data
        
        def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data ['Prediction'] = self.logic.predict(self.data)
            return self.preprocessed_data
            
    


# In[ ]:




