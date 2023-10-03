# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 18:01:12 2023

@author: xxjan
"""

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = datasets.load_iris()
data = iris.data
target = iris.target
target = np.eye(3)[target]  # One-hot encode

# Scale features
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# RNN parameters
input_size = 4  # 4 features for iris dataset
hidden_size = 5
output_size = 3  # 3 classes
sequence_length = 1  # For simplicity, each data point is a sequence of length 1
learning_rate = 0.01

# Weights and biases
Wxh = np.random.randn(input_size, hidden_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(hidden_size, output_size) * 0.01
bh = np.zeros((1, hidden_size))
by = np.zeros((1, output_size))

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

# Train RNN
for epoch in range(1000):
    loss = 0
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    
    for i, (x, y) in enumerate(zip(X_train, y_train)):
        x = np.reshape(x, (1, input_size))
        
        h_prev = np.zeros((1, hidden_size))
        
        # Forward pass
        h = np.tanh(np.dot(x, Wxh) + np.dot(h_prev, Whh) + bh)
        y_pred = softmax(np.dot(h, Why) + by)
        
        # Loss
        loss += -np.sum(y * np.log(y_pred))
        
        # Backward pass
        dy = y_pred - y
        dWhy += np.dot(h.T, dy)
        dby += dy
        dh = np.dot(dy, Why.T) * (1 - h * h)
        dWxh += np.dot(x.T, dh)
        dbh += dh
        
    # Parameter update using vanilla SGD
    Wxh -= learning_rate * dWxh
    Whh -= learning_rate * dWhh
    Why -= learning_rate * dWhy
    bh -= learning_rate * dbh
    by -= learning_rate * dby

    if epoch % 100 == 0:
        print('Epoch:', epoch, 'Loss:', loss / len(X_train))

# Test accuracy
correct = 0
for i, (x, y) in enumerate(zip(X_test, y_test)):
    x = np.reshape(x, (1, input_size))
    h = np.tanh(np.dot(x, Wxh) + bh)
    y_pred = softmax(np.dot(h, Why) + by)
    if np.argmax(y_pred) == np.argmax(y):
        correct += 1
print('Test Accuracy:', correct / len(X_test))