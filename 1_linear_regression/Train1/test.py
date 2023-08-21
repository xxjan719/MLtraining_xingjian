# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 16:51:16 2023

@author: xxjan
"""

from train import *
from sklearn import datasets
import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import StandardScaler
House_price_X, House_price_y = datasets.load_boston(return_X_y=True)
House_price_yy = House_price_y.reshape(np.shape(House_price_y)[0],1)
data = np.concatenate([House_price_X,House_price_yy],axis=1)
mm = StandardScaler()# create Object for StandScaler 
mm_data = mm.fit_transform(data) # normalization for this Boston data
origin_data = mm.inverse_transform(mm_data) # return back to this Boston data



def sklearn_sample(data,model2):
    mm_test = data[-1-50:-1,:]
    mm_train = data[:-2-50,:]
    X_train = mm_train[:,:-1]
    Y_train = mm_train[:,-1]
    X_test = mm_test[:,:-1]
    Y_test = mm_test[:,-1]
    model = model2.fit(X_train,Y_train)
    coef = model.coef_
    intercept = model.intercept_
    print("coef:{},  intercept:{}".format(coef,intercept))
    y_test = model.predict(X_test)
    epsilon = (1/2)*np.mean((y_test-Y_test)**2)
    print(epsilon)


def myself_sample(data):
    mm_test = data[-1-50:-1,:]
    mm_train = data[:-2-50,:]
    X_train = mm_train[:,:-1]
    Y_train = mm_train[:,-1]
    Y_train = Y_train.reshape(Y_train.shape[0],1)
    X_test = mm_test[:,:-1]
    Y_test = mm_test[:,-1]
    M,N = np.shape(X_train)
    model = LinearRegression2(M, N)
    coef,intercept = model.fit(X_train,Y_train)
    print("coef:{},  intercept:{}".format(coef,intercept))
    y_test = model.predict(X_test)
    epsilon = (1/2)*np.mean((y_test-Y_test)**2)
    print(epsilon)







#sklearn_sample(mm_data,LR())
myself_sample(mm_data)
