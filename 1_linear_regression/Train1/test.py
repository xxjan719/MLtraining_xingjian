# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 16:51:16 2023

@author: xxjan
"""

from train_2 import *
from sklearn import datasets
import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import StandardScaler
House_price_X, House_price_y = datasets.load_boston(return_X_y=True)
House_price_yy = House_price_y.reshape(np.shape(House_price_y)[0],1)
data = np.concatenate([House_price_X,House_price_yy],axis=1)
mm = StandardScaler()# 创建MinMaxScaler 对象
mm_data = mm.fit_transform(data) # 归一化数据
origin_data = mm.inverse_transform(mm_data) # 转换成归一化之前的数据
X = mm_data[:,:-1]
Y = mm_data[:,-1]
model = LR().fit(X,Y)
coef = model.coef_
intercept = model.intercept_
print("coef:{},  intercept:{}".format(coef,intercept))

M,N = np.shape(X)
reg = LinearRegression2(M,N)
#reg.fit(X, y)
y = Y.reshape(Y.shape[0],1)
coef,intercept = reg.fit(X, y)
