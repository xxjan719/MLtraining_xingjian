# -*- coding: utf-8 -*-
"""
Created on Mon Sep 9 23:38:41 2023

@author: xxjan
"""
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
X_train = pd.read_csv('../Data/Digits_X_train.csv').values
y_train = pd.read_csv('../Data/Digits_y_train.csv').values
X_test  = pd.read_csv('../Data/Digits_X_test.csv').values
y_test  = pd.read_csv('../Data/Digits_y_test.csv').values
np.random.seed(123)
# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode the target
encoder = OneHotEncoder(sparse=False)
y_onehot_train = encoder.fit_transform(y_train)
y_onehot_test = encoder.fit_transform(y_test)

def accuracy(ypred, yexact):
    p = np.array(ypred==yexact, dtype=int)
    return np.sum(p)/float(len(yexact))

def tahn(z):
    return (np.exp(z)-np.exp(-z)) / (np.exp(z) + np.exp(-z))


def softmax(z):
    exps = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)



def tahn_derivative(z):
    return 1-(tahn(z))**2

def compute_loss(y_true, y_pred):
    return -(np.sum(y_true * np.log(y_pred)))

input_size = X_train.shape[1]
hidden_size = 64
output_size = y_onehot_train.shape[1]

# Initialize weights
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

learning_rate = 0.01
epochs = 10000

losses = []

for epoch in range(epochs):
    # Forward propagation
    z1 = np.dot(X_train, W1) + b1
    a1 = tahn(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    
    loss = compute_loss(y_onehot_train, a2)
    losses.append(loss)
    
    # Backward propagation
    dz2 = a2 - y_onehot_train
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0)
    
    dz1 = np.dot(dz2, W2.T) * tahn_derivative(a1)
    dW1 = np.dot(X_train.T, dz1)
    db1 = np.sum(dz1, axis=0)
    
    # Update weights
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Test the model
z1 = np.dot(X_test, W1) + b1
a1 = tahn(z1)
z2 = np.dot(a1, W2) + b2
a2 = softmax(z2)

predictions = np.argmax(a2, axis=1)
accuracy1= accuracy(y_test.flatten(), predictions)
print(f"Accuracy: {accuracy1 * 100:.2f}%")






