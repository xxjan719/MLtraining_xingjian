# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 17:41:40 2023

@author: xxjan
"""
import numpy as np
from collections import Counter
from sklearn import datasets
 
class KNN:
    def __init__(self):
        self.train_data = None
        self.train_label = None
 
    def fit(self, train_data, train_label):
        self.train_data = np.array(train_data)
        self.train_label = np.array(train_label)
 
    def predict(self, test_data, k=3):
        test_data = np.array(test_data)
        preds = []
        for x in test_data:
            dists = self.Eulidean_distance(x)
            sorted_idx = np.argsort(dists)
            knearnest_labels = self.train_label[sorted_idx[:k]]
            pred = None          
            knearnest_labels = list(knearnest_labels.squeeze())
            pred = Counter(knearnest_labels).most_common(1)[0][0]
            preds.append(pred)
        return preds
 
    def Eulidean_distance(self, x):
        return np.sqrt(np.sum(np.square(self.train_data-x), axis=1))
 
def accuracy(prediction, actual):
    return np.sum(prediction==actual)/len(prediction)

# Normalize the input
def normalize(x):
    for i in range(x.shape[1]):
        x[:, i] = (x[:, i] - x[:, i].min()) / (x[:, i].max() - x[:,
                                                               i].min())  # normalize each column of x seperately. x_new = (x_old - x_old.min())/(x_old.max - x_old.min())

    return x    


if __name__ == '__main__':
    
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
    
    X_train = normalize(X_train)
    
    X_test = normalize(X_test)
    
    
    knn = KNN()
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test, k=5)
    print('Accuracy on test set: {:.2f}'.format(accuracy(preds, y_test.flatten())))

