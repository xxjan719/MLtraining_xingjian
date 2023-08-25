# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 15:51:15 2023

@author: xxjan
"""
import numpy as np
from sklearn import datasets
from collections import Counter
#set up the data for the KNN model
X,y = datasets.load_iris(return_X_y= True)

test_size = 25
### Randomly shuffle data
np.random.seed(1)
mask = np.random.permutation(range(len(X)))
X = X[mask]
y = y[mask].reshape(-1,1)

## Assign to training and testing sets
X_train = X[test_size:,:]
y_train = y[test_size:,:]
X_test = X[:test_size,:]
y_test = y[:test_size,:]


def normalize(x):
    for i in range(x.shape[1]):
        x[:, i] = (x[:, i] - x[:, i].min()) / (x[:, i].max() - x[:,
                                                               i].min())  # normalize each column of x seperately. x_new = (x_old - x_old.min())/(x_old.max - x_old.min())

    return x  


X_train = normalize(X_train)
X_test = normalize(X_test)




#Build the KNN model
def Euclidean_distance(X,Y):
    def Euclidean_distance_part(X,Y):
        dist = 0
        for i in range(len(X)):
            dist+= ((X[i]-Y[i])**2)
        return np.sqrt(dist)
    distance_all = []
    for i in range(X.shape[0]):
        dist = Euclidean_distance_part(X[i], Y)
        distance_all.append(dist)
    distance_all = np.squeeze(distance_all)
    distance_all= distance_all.reshape(distance_all.shape[0],1)
    return distance_all
    
    

def knn_predict(X_train,y_train,X_test,k=5):
    prediction = []
    for x in X_test:
        dist = list(np.squeeze(Euclidean_distance(X_train,x)))  
        sort_dist = np.argsort(dist)[:k] 
        label = y_train[sort_dist].squeeze()
        pred = np.bincount(label).argmax() 
        print('pred is: {}'.format(pred))
        prediction.append(pred)
    return prediction

def accuracy(prediction, actual):
    return np.sum(prediction == actual) / len(prediction)

predictions = knn_predict(X_train, y_train, X_test)
predictions = np.array(predictions).reshape(len(predictions),1)
print('Accuracy on test set: {:.2f}'.format(accuracy(predictions, y_test)))





