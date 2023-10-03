# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 23:22:58 2023

@author: xxjan
"""

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(hidden_size, output_size) * 0.01
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward_pass(self, x):
        h = np.tanh(np.dot(x, self.Wxh) + np.dot(np.zeros((1, self.Wxh.shape[1])), self.Whh) + self.bh)
        y_pred = self.softmax(np.dot(h, self.Why) + self.by)
        return y_pred, h

    def train(self, X_train, y_train, epochs=1000):
        for epoch in range(epochs):
            loss = 0
            dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
            dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)

            for x, y in zip(X_train, y_train):
                x = np.reshape(x, (1, self.Wxh.shape[0]))

                y_pred, h = self.forward_pass(x)

                loss += -np.sum(y * np.log(y_pred))

                dy = y_pred - y
                dWhy += np.dot(h.T, dy)
                dby += dy
                dh = np.dot(dy, self.Why.T) * (1 - h * h)
                dWxh += np.dot(x.T, dh)
                dbh += dh

            self.Wxh -= self.learning_rate * dWxh
            self.Whh -= self.learning_rate * dWhh
            self.Why -= self.learning_rate * dWhy
            self.bh -= self.learning_rate * dbh
            self.by -= self.learning_rate * dby

            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Loss: {loss / len(X_train)}')

    def predict(self, X):
        y_preds = []
        for x in X:
            x = np.reshape(x, (1, self.Wxh.shape[0]))
            y_pred, _ = self.forward_pass(x)
            y_preds.append(np.argmax(y_pred))
        return np.array(y_preds)

    def accuracy(self, X, y_true):
        y_preds = self.predict(X)
        y_true = np.argmax(y_true, axis=1)
        return np.mean(y_preds == y_true)


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

# Create and train RNN
rnn = SimpleRNN(input_size=4, hidden_size=5, output_size=3, learning_rate=0.01)
rnn.train(X_train, y_train, epochs=1000)

# Evaluate
print('Test Accuracy:', rnn.accuracy(X_test, y_test))