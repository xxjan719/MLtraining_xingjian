# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 23:24:51 2023

@author: xxjan
"""
from Logisticregression import *
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
def data1():
    X,y = make_classification(n_features=4)
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.1)
    normal = StandardScaler()
    X_tr = normal.fit_transform(X_tr)
    X_te = normal.fit_transform(X_te)
    M,N = np.shape(X_tr)
    obj1 = LogisticRegression(M,N)
    model= obj1.fit(X_tr,y_tr)
    y_pred = obj1.predict(X_te)
    y_train = obj1.predict(X_tr)
    #Let's see the f1-score for training and testing data
    f1_score_tr = F1_score(y_tr,y_train)
    f1_score_te = F1_score(y_te,y_pred)
    print(f1_score_tr)
    print(f1_score_te)

    
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_tr, y_tr)
    y_pred2 = logisticRegr.predict(X_tr)
    f1_score_tr2 = F1_score(y_tr,y_pred2)
    print(f1_score_tr2)
    y_pred3 = logisticRegr.predict(X_te)
    f1_score_tr3 = F1_score(y_te,y_pred3)
    print(f1_score_tr3)

def data2():
    data = pd.read_csv("C:\\Users\\xxjan\\Downloads\\archive\\health care diabetes.csv")
    X = np.array(data)[:,:-1]
    y = np.array(data)[:,-1]
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.1)
    normal = StandardScaler()
    X_tr = normal.fit_transform(X_tr)
    X_te = normal.fit_transform(X_te)
    M,N = np.shape(X_tr)
    obj1 = LogisticRegression(M,N)
    model= obj1.fit(X_tr,y_tr)
    y_pred = obj1.predict(X_te)
    y_train = obj1.predict(X_tr)
    #Let's see the f1-score for training and testing data
    f1_score_tr = F1_score(y_tr,y_train)
    f1_score_te = F1_score(y_te,y_pred)
    print(f1_score_tr)
    print(f1_score_te)

    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_tr, y_tr)
    y_pred2 = logisticRegr.predict(X_tr)
    f1_score_tr2 = F1_score(y_tr,y_pred2)
    print(f1_score_tr2)
    y_pred3 = logisticRegr.predict(X_te)
    f1_score_tr3 = F1_score(y_te,y_pred3)
    print(f1_score_tr3)
