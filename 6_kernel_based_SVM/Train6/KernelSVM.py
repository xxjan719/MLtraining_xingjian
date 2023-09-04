# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 14:12:00 2023

@author: xxjan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Util import gen_two_clusters, visualize2d
from Util import gen_spiral
np.random.seed(123)

X_train = np.array([[-2,4],[4,1],[1,6],[2,4],[6,2],])

Y_train = np.array([-1,-1,1,1,1]).reshape(X_train.shape[0],1)
X_test = np.array([[2,2],[4,3],[5,5],[1,3],[3,4],[-2,5]])
y_test = np.array([-1,1,1,-1,1,-1]).reshape(X_test.shape[0],1)

def data_fitst_plot(X,X_test,Y):
    for i in range(X_test.shape[0]):
        plt.scatter(X_test[i][0],X_test[i][1],s=100,marker='*',color='orange',linewidth=5)
    
    for val,inp in enumerate(X):
        if Y[val]== 1:
            plt.scatter(inp[0],inp[1],s=100,marker='+',color='blue',linewidth=5)
        else:
            plt.scatter(inp[0],inp[1],s=100,marker='_',color='red',linewidth=5)
    plt.plot([-2,6],[6,1],color='green')



def accuracy(ypred,yexact):
    p = np.array(ypred==yexact,dtype=int)
    return np.sum(p)/float(len(yexact))

class SVM_kernel():
    def __init__(self,X,y,kernel='rbf',lr=0.01,lmd=0.01,epoch=2000):
        
        self.X = X
        self.y = y.reshape(y.shape[0],1)
        self.M = X.shape[0]
        self.N = X.shape[1]
        self.P = y.shape[0]
        self.alpha = np.random.rand(self.M,1)
        self.b = np.zeros((1,1))
        self.lr = lr
        self.lmd = lmd
        self.epoch = epoch
        self.kernels = { 'linear':self.linear,'rbf':self.rbf}
        self.kernel = self.kernels[kernel]
        self.gamma = 1/self.N
        self.KerMat = self.kernel_matrix(self.X)
    
    def kernel_matrix(self,X):
        KerMat = np.zeros([X.shape[0],X.shape[0]])
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                KerMat[i,j] = self.kernel(X[i],X[j])
        return KerMat
    
    def linear(self, Xi, Xj):
        return np.dot(Xi, Xj)

    def rbf(self, Xi, Xj):
        return np.exp(-self.gamma*np.dot(Xi-Xj, Xi-Xj))

    def fit(self):
         k_mat_diag = np.diag(self.KerMat)
         #print(k_mat_diag.shape)
         for _ in range(self.epoch):
            #self.W -= self.lr * (np.sum(self.W * self.KerMat, axis=1) + self.W * k_mat_diag) * 0.5
            err = 1 - self.y * (self.KerMat.dot(self.alpha) + self.b)
            #print(err)
            part1 = 0
            part2 = 0
            part3 = self.KerMat
            for i in range(err.shape[0]):
                part3[i,:] = self.alpha[i] * part3[i,:]
                part2 += (self.lmd*self.y[i])
                if err[i]>0:
                    part1 += -(self.lmd*self.y[i]*np.sum(self.KerMat,axis=1)).reshape(self.M,1)
                else:
                    part1 += 0
            
            part4 = (1/2)*(np.sum(part3,axis=1)).reshape(self.M,1)
            
            part4 = part4+ (1/2)*np.multiply(self.alpha,k_mat_diag.reshape(self.M,1))
            
            dW = part4+part1
            db = part2
            self.alpha -= self.lr*dW
            self.b -= self.lr*db

    def predict(self, X_test):
         KerMat = self.kernel_matrix(X_test)
         y_pred = np.dot(KerMat, self.alpha) + self.b
         return np.sign(y_pred)

#data_fitst_plot(X_train,X_test,Y_train)   
model = SVM_kernel(X_train,Y_train)
model.fit()
y_pred = model.predict(X_train)
print(accuracy(y_pred, Y_train))