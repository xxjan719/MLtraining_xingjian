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
from sklearn.metrics import mean_squared_error
House_price_X, House_price_y = datasets.load_boston(return_X_y=True)
House_price_yy = House_price_y.reshape(np.shape(House_price_y)[0],1)
data = np.concatenate([House_price_X,House_price_yy],axis=1)
mm = StandardScaler()# 创建MinMaxScaler 对象
mm_data = mm.fit_transform(data) # 归一化数据
origin_data = mm.inverse_transform(mm_data) # 转换成归一化之前的数据



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
    epsilon = np.mean((y_test-Y_test)**2)
    epsilon2 = mean_squared_error(y_test, Y_test)
    print("epsilon:{},epsilon2:{}".format(epsilon,epsilon2))


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
    Y_test = Y_test.reshape(Y_test.shape[0],1)
    epsilon = np.mean((y_test-Y_test)**2)
    epsilon2 = mean_squared_error(y_test, Y_test)
    print("epsilon:{},epsilon2:{}".format(epsilon,epsilon2))







sklearn_sample(mm_data,LR())
myself_sample(mm_data)

