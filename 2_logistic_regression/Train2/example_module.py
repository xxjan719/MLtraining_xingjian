# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:19:20 2023

@author: Jiahun Chen
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys,os

def sigmoid(z):
    return 1./(1+np.exp(-z))

class LogisticRegression:
    def __init__(self,M,N,lr=0.1):
        self.M = M
        self.N = N
        self.lr = lr
        self.W = np.zeros((N,1))
        self.b = 0.
        self.scaler = StandardScaler()
    
    def fit(self,X,y,epoch=2000,visual=False):
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        
        for i in range(epoch):
            y_hat = sigmoid(np.dot(X,self.W)+self.b)
            diff = (y_hat-y)/self.M
            dW = np.dot(X.T,diff)
            db = np.sum(diff)
            self.W -= self.lr*dW
            self.b -=self.lr*db
            print('Epoch:%d,loss=%.2f'%(i,self.loss_func(X,y)))
    
    def predict(self,X_test):
        X_test = self.scaler.transform(X_test)
        y_hat = sigmoid(np.dot(X_test,self.W)+self.b)
        return np.where(y_hat>0.5,1,0)
    
    def loss_func(self,X,y):
        y_hat = sigmoid(np.dot(X,self.W)+self.b)
        return -np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))/self.M