# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:34:52 2023

@author: Jiahui Chen
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import datasets
import sys,os

class LinearRegression2:
    def __init__(self,M,N):
        '''
        M: the number of features
        N: the number of data
        coef: The coefficent
        '''
        self.M = M
        self.N = N
        self.coef_ = np.zeros((N,), dtype= float)
        self.intercept_ = 0.0
        self.scaler = StandardScaler()
    
    def fit(self,X,y,method = 'GM',lr=0.01,epoch=2000,tol=12.00,visual=False):
        #update the coefficient of the linear regression model
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        
        if method == "PM":
            #use the projection matrix
            print(X)
            #create an array of all ones with the size of(self.M,1)
            ones = np.ones((self.M,1))
            #concatenate the array of all ones in the first column of X
            X = np.concatenate([ones,X],axis=1)
            #calculate the matrix multiplication of the transpose of X with X
            XTX = np.matmul(X.T,X)
            #calculate the coefficients using the projection matrix
            coefficients = np.matmul(np.linalg.inv(XTX),np.matmul(X.T,y))
            self.intercept_ = coefficients[0]
            self.coef_ = coefficients[1:]
        
        elif method =="GD":
            #gradient descent method
            i =0
            while np.abs(self.loss_func(X,y))>tol and i<epoch:
                y_hat = np.dot(X,self.coef_)+self.intercept_
                diff =(y_hat-y)/self.M
                self.coef_ -=lr*(np.dot(X.T,diff)+0.0005*self.coef_)
                self.intercept_ -=lr*np.sum(diff)
                if i%20 ==0:
                    print('epoch:%d,current loss:%.5f'%(i,self.loss_func(X,y)))
                i+=1
    def loss_func(self,X,y):
        return 0.5*np.sum((np.dot(X,self.coef_)+self.intercept_-y)**2)/self.M
    
    def predict(self,X):
        #multiply x to the coefficients and return the prediction
        x = self.scaler.transform(X)
        y = np.matmul(X,self.coef_)+self.intercept_
        return y 
        
        
#load the House price dataset1
House_price_X,House_price_y = datasets.load_boston(return_X_y=True)

#split the data into training/testing sets
House_price_X_train = House_price_X[:-50]
House_price_X_test = House_price_X[-50:]  
House_price_y_train = House_price_y[:-50]
House_price_y_test = House_price_y[-50:]       
            
numDataset,numFeatures = House_price_X_train.shape
linear_model = LinearRegression2(numDataset, numFeatures)
linear_model.fit(House_price_X_train, House_price_y_train,'PM')
           
linear_model2 = LinearRegression()
linear_model2.fit(House_price_X_train, House_price_y_train)            
            
#Make predictions using the testing set
House_price_y_pred1 = linear_model.predict(House_price_X_test)
House_price_y_pred2 = linear_model2.predict(House_price_X_test)       
            
print('coefficient myexample:\n',linear_model.coef_)
print('coefficient sklearn:\n',linear_model2.coef_) 

print('intercept myexample:\n',linear_model.intercept_)
print('intercept sklearn:\n',linear_model2.intercept_)

print('myexample:Mean-squared error:%.2f'%mean_squared_error(House_price_y_pred1, House_price_y_test))
print('sklearn:Mean-squared error:%.2f'%mean_squared_error(House_price_y_pred2, House_price_y_test))
                      
            
            
            
            
            
            
            
            
            
            
            
            

