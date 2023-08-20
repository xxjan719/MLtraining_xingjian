# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 07:46:43 2023

@author: xxjan
"""

import numpy as np
np.random.seed(123)
class LinearRegression2:
    def __init__(self, M, N):
        ''' param:
            M:    the number of data
            N:    the number of features
        '''
        self.M = M
        self.N = N        
        self.coef_ = np.random.normal(0,1,size=(N,1))
        self.intercept_ = np.random.rand(1).reshape(1,1)
        
    def fit(self, X, y):
        '''
            update the coefficient of the linear regression model
            X: (#data, #feature)
            y: (#data, )
        '''
        weight_matrix = np.vstack((self.intercept_,self.coef_))
        #print(weight_matrix.shape)
        X_matrix = np.hstack((np.ones((self.M,1)),X))
        #print(X_matrix.shape)
        
        for i in range(11):
            self.dlossdw = np.matmul(np.matmul(X_matrix.transpose(),X_matrix),weight_matrix)-np.matmul(X_matrix.transpose(),y)            
            weight_matrix -= 0.1*self.dlossdw
        self.coef_ = weight_matrix[1:]
        self.intercept_ =weight_matrix[0] 
        w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_matrix.transpose(),X_matrix)),X_matrix.transpose()),y)
        print(w)
        return self.coef_,self.intercept_
        

    def predict(self, X):
        y_pred =  np.matmul(X,self.coef_)+self.intercept_
        return y_pred
    






