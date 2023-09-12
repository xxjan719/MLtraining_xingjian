# -*- coding: utf-8 -*-
"""
Created on Mon Sep 9 21:10:45 2023

@author: xxjan
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
np.random.seed(123)

class RegressionTree:
    def __init__(self, max_depth=5, current_depth=0, max_features=None):
        self.left_tree = None
        self.right_tree = None
        self.max_depth = max_depth
        self.current_depth = current_depth

        self.best_feature_id = 0
        self.best_gain = 0.
        self.best_split_value = 0.
        self.var = 0.
        self.label = None

        self.X = None
        self.y = None
        self.N = 0
        self.M = 0

        self.max_features = max_features

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.N = X.shape[1]
        self.M = X.shape[0]
        if self.max_features is None or self.max_features > self.N:
            self.max_features = self.N
        if self.current_depth <= self.max_depth:
            self.var = np.var(self.y)
            self.best_feature_id, self.best_gain, self.best_split_value = self.find_best_split()
            if self.best_gain > 0.:
                self.split_trees()

    def split_trees(self):
        self.left_tree = RegressionTree(self.max_depth, self.current_depth + 1)
        self.right_tree = RegressionTree(self.max_depth, self.current_depth + 1)
        best_feature_values = self.X[:, self.best_feature_id]
        left_indices = np.where(best_feature_values < self.best_split_value)
        right_indices = np.where(best_feature_values >= self.best_split_value)
        
        left_tree_X = self.X[left_indices]
        left_tree_y = self.y[left_indices]
        right_tree_X = self.X[right_indices]
        right_tree_y = self.y[right_indices]

        self.left_tree.fit(left_tree_X, left_tree_y)
        self.right_tree.fit(right_tree_X, right_tree_y)

    def find_best_split(self):
        best_feature_id = None
        best_gain = 0.
        best_split_value = None
        n_features = np.random.choice(self.N, self.max_features, replace=False)
        for feature_id in n_features:
            current_gain, current_split_value = self.find_best_split_one_feature(feature_id)
            if best_gain < current_gain:
                best_feature_id = feature_id
                best_gain = current_gain
                best_split_value = current_split_value

        return best_feature_id, best_gain, best_split_value

    def find_best_split_one_feature(self, feature_id):
        feature_values = self.X[:, feature_id]
        unique_feature_values = np.unique(feature_values)
        best_gain = 0.
        best_split_value = None
        if len(unique_feature_values) == 1:
            return best_gain, best_split_value
        for fea_val in unique_feature_values:
            left_indices = np.where(feature_values < fea_val)
            right_indices = np.where(feature_values >= fea_val)

            left_tree_y = self.y[left_indices]
            right_tree_y = self.y[right_indices]

            left_var = np.var(left_tree_y)
            right_var = np.var(right_tree_y)

            left_n = len(left_tree_y)
            right_n = len(right_tree_y)

            current_gain = self.var - (left_n / self.M * left_var + right_n / self.M * right_var)
            if best_gain < current_gain:
                best_gain = current_gain
                best_split_value = fea_val

        return best_gain, best_split_value

    def predict(self, X_test):
        n_test = X_test.shape[0]
        y_pred = np.empty(n_test)
        for i in range(n_test):
            y_pred[i] = self.tree_propagation(X_test[i])
        return y_pred

    def tree_propagation(self, feature):
        if self.left_tree is None:
            return self.predict_label()
        if feature[self.best_feature_id] < self.best_split_value:
            child_tree = self.left_tree
        else:
            child_tree = self.right_tree
        return child_tree.tree_propagation(feature)

    def predict_label(self):
        if self.label is not None:
            return self.label
        self.label = np.mean(self.y)
        return self.label


class RandomForest:
    def __init__(self, n_trees=10, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            # Bootstrap sampling
            bootstrap_indices = np.random.choice(np.arange(X.shape[0]), size=X.shape[0], replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            tree = RegressionTree(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_predictions, axis=0)

class GradientBoostedDecisionTreeRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5, max_features=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        self.max_depth = max_depth
        self.max_features = max_features

    def fit(self, X, y):
        # Initial prediction as the mean of the target
        y_pred = np.full(y.shape, np.mean(y))
        for _ in range(self.n_estimators):
            # Compute the residuals
            residuals = y - y_pred
            tree = RegressionTree(max_depth=self.max_depth, max_features=self.max_features)
            tree.fit(X, residuals)
            self.trees.append(tree)
            # Update predictions
            y_pred += self.learning_rate * np.array([tree.tree_propagation(x) for x in X])

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * np.array([tree.tree_propagation(x) for x in X])
        return y_pred




if __name__ == '__main__': # check why we need to have this
   # generate regression data
   n_samples = 1000
   x1 = np.random.rand(n_samples)*10-5 #Values between -5 to 5
   x1 = x1.reshape(x1.shape[0],1)
   x2 = np.random.rand(n_samples)*10-5 #Values bewteen -5 to 5
   x2 = x2.reshape(x2.shape[0],1)
   noise = np.random.rand(n_samples)*2 # Gaussian noise

   y = x1**2+x2**3+noise

   X = np.concatenate([x1,x2],axis=1)

   X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2,
   shuffle=True, random_state=1) 
   
   #RT = RegressionTree(max_depth=25)
   RF = RandomForest(max_depth=25)
   #RT.fit(X_train, y_train)
   RF.fit(X_train,y_train)
   y_pred = RF.predict(X_test)
   y_test_mean = np.mean(y_test,axis=1)
