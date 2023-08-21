# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 21:55:02 2023

@author: xxjan
"""
import numpy as np
np.random.seed(12)
def sigmoid(z):
    # write the sigmoid function
    return 1/(1+np.exp(-z))


class LogisticRegression:
    def __init__(self, M, N, lr=0.1):
        # unlike linear regression. we need the weight matrix here
        # StandardScaler(): normalizer, which is important
        # initailize all variables
        self.W = np.random.normal(0,1,size=(N,1))# weight matrix
        self.b = np.random.rand(1).reshape(1,1)# bias
        self.lr = lr
        self.M = M
        self.N = N
    def fit(self, X, y, epoch=5000):
        # normalize first
        # for each epoch, update the weight matrix and bias
        M,N = np.shape(X)
        if len(y.shape) == 1:
            y = y.reshape(y.shape[0],1)
            
        weights = np.concatenate([self.b, self.W],axis=0)
        X = np.c_[np.ones((np.shape(X)[0],1)),X]
        costs = []
        
        for i in range(1,epoch+1):
            H = sigmoid(np.dot(X,weights))        
            cost0 = y.T.dot(np.log(sigmoid(H)))
            cost1 = (1-y).T.dot(np.log(1-sigmoid(H)))
            cost = -((cost1 + cost0))/self.M
            cost = np.squeeze(cost)
            costs.append(cost)
            weights = weights - self.lr * np.dot(X.T, sigmoid(np.dot(X,weights)) - np.reshape(y,(len(y),1)))
            if i % 100 == 0:
                print ('Epoch:{}, The cost is :{}'.format(i, cost))
        
        self.b = weights[0]
        
        self.W = weights[1:]
        
        return self.W, self.b, costs

    def predict(self, X_test):
        X = np.c_[np.ones((np.shape(X_test)[0],1)),X_test]
        weight = np.concatenate([self.b.reshape(self.b.shape[0],1), self.W.reshape(self.W.shape[0],1)],axis=0)
        H = sigmoid(np.dot(X,weight))
        y_pred = []
        for i in H:
            if i>0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
        return y_pred

def F1_score(y,y_hat):
    tp,tn,fp,fn = 0,0,0,0
    for i in range(len(y)):
        if y[i] == 1 and y_hat[i] == 1:
            tp += 1
        elif y[i] == 1 and y_hat[i] == 0:
            fn += 1
        elif y[i] == 0 and y_hat[i] == 1:
            fp += 1
        elif y[i] == 0 and y_hat[i] == 0:
            tn += 1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*precision*recall/(precision+recall)
    return f1_score


