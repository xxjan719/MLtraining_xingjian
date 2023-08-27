# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 21:10:36 2023

@author: xxjan
"""

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np


np.random.seed(7)
iris = datasets.load_iris()
X = iris.data
y = iris.target

#Normalize X
scale = StandardScaler()
scale.fit(X)
X = scale.transform(X)

def initialize_random_centroids(K,X):
    #randomly choose K centroid from X
    m,n = np.shape(X)
    centroids = np.empty((K,n))
    for i in range(K):
        centroids[i] = X[np.random.choice(range(m))]
    return centroids

def Euclidean_dist(x,y):
    return np.sqrt(np.sum(np.square(x-y)))

def closets_centroid(x,centroids,K):
    distances = np.empty(K)
    for i in range(K):
        distances[i] = Euclidean_dist(centroids[i], x)
    return np.argmin(distances)

def cluster_make(K,X,centroids):
    m = X.shape[0]
    cluster_index = np.empty(m)
    for i in range(m):
        cluster_index[i] = closets_centroid(
            X[i], centroids, K)
    return cluster_index

def cluster_mean(K,X,cluster_index):
    n = X.shape[1]
    centroids = np.empty((K,n))
    for i in range(K):
        points = X[cluster_index==i]
        centroids[i] = np.mean(points,axis=0)
    return centroids






class kmeans:
    def __init__(self,X,k=3,iters=1000):
        self.X = X
        self.k = k
        self.iters = iters
        self.centers =initialize_random_centroids(k, X) 
        

    def fit(self):
        print('initial centroids:{}'.format(self.centers))
        for _ in range(self.iters):
            clusters = cluster_make(self.k, self.X, self.centers)
            previous_centroids = self.centers
            self.center = cluster_mean(self.k, self.X, clusters)
            diff = previous_centroids - self.centers
            if not diff.any():
                return clusters
        return clusters

def accuracy(prediction, actual):
    return np.sum(prediction == actual) / len(prediction)

    

myKM = kmeans(X, k=3)
y_pred = myKM.fit() 
print(y_pred)
print('Accuracy on test set: {:.2f}'.format(accuracy(y_pred, y)))
    #y_var.append(y_pred)
    
