# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 21:08:54 2023

@author: xxjan
"""

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)
#Input data
X = np.array([
    [-2,4],
    [4,1],
    [1,6],
    [2,4],
    [6,2],
    
    ])

Y = np.array([-1,-1,1,1,1])
X_test = np.array([
    [2,2],
    [4,3],
    [5,5],
    [1,3],
    [3,4],
    [-2,5]
    ])
def accuracy(ypred, yexact):
    p = np.array(ypred==yexact, dtype=int)
    return np.sum(p)/float(len(yexact))


def data_fitst_plot(X,X_test,Y):
    for i in range(X_test.shape[0]):
        plt.scatter(X_test[i][0],X_test[i][1],s=100,marker='*',color='orange',linewidth=5)
    
    for val,inp in enumerate(X):
        if Y[val]== 1:
            plt.scatter(inp[0],inp[1],s=100,marker='+',color='blue',linewidth=5)
        else:
            plt.scatter(inp[0],inp[1],s=100,marker='_',color='red',linewidth=5)
    plt.plot([-2,6],[6,1],color='green')
    
class SVM():
    def __init__(self,X,y,lr=0.01,lmd=0.01,epoch=2000):
        self.X = X
        self.y = y
        self.M = X.shape[0]
        self.N = X.shape[1]
        self.P = y.shape[0]
        self.W = np.random.rand(self.N,1)
        self.b = np.zeros((1,1))
        self.lr = lr
        self.lmd = lmd
        self.epoch = epoch
    
    def fit(self):
        print("M:{},N:{},P:{}".format(self.M,self.N,self.P))
        y = self.y.reshape([self.P,1])
        for _ in range(self.epoch):
            # calculate the predicted values
            y_pred = np.dot(self.X,self.W)+self.b
            #print(y_pred)
            # # calculate the derivative
            part1 = np.zeros((self.N,1))
            part2 = np.zeros((1,1))
            for i in range(self.P):
                part2 += (self.lmd*y[i])
                if (self.y[i]*y_pred[i] < 1):
                    part1 += -(self.lmd*self.y[i]*self.X[i]).reshape(self.N,1)
                else:
                    part1 += 0
                
            dW = 2*self.W+part1
            db = part2
            self.W -=self.lr*dW
            self.b -=self.lr*db
     
    def predict(self, X_test):
        ypred = np.dot(X_test, self.W) + self.b
        return np.sign(ypred)
    
    
data_fitst_plot(X,X_test,Y)   
model = SVM(X,Y)
model.fit()
y_pred = model.predict(X)
y_pred_test = model.predict(X_test)
y_test = np.array([-1,1,1,-1,1,-1]).reshape(y_pred_test.shape[0],1)
print(accuracy(y_pred_test, y_test))