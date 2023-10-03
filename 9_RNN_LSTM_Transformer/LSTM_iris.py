# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 00:32:18 2023

@author: xxjan
"""
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
np.random.seed(123)
# 1. Preprocess the dataset
iris = datasets.load_iris()
X = iris.data.reshape(150, 4, 1)  # treat features as a sequence
y = iris.target.reshape(-1, 1)

# One-hot encode the labels
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# 2. Modify the LSTM for classification
input_size = 1
hidden_size = 64
output_size = 3  # 3 classes in Iris dataset

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return sigmoid(y) * (1 - sigmoid(y))

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1.0 - np.tanh(y)**2



class LSTM:
    def __init__(self, input_size, hidden_size):
        # Initialize weights and biases for input, forget, output, and block gates
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wg = np.random.randn(hidden_size, input_size + hidden_size)
        
        self.bi = np.zeros((hidden_size, 1))
        self.bf = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bg = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
        # Concatenate h_prev and x
        concat = np.vstack((h_prev, x))

        # Compute values using LSTM equations
        i = sigmoid(np.dot(self.Wi, concat) + self.bi)
        f = sigmoid(np.dot(self.Wf, concat) + self.bf)
        o = sigmoid(np.dot(self.Wo, concat) + self.bo)
        g = tanh(np.dot(self.Wg, concat) + self.bg)

        c_next = f * c_prev + i * g
        h_next = o * tanh(c_next)

        return h_next, c_next, (i, f, o, g, c_next)

    def backward(self, dh_next, dc_next, cache, concat):
        i, f, o, g, c_next = cache

        # Gradients of output w.r.t various gates
        do = tanh(c_next) * dh_next
        dc = (o * (1 - tanh(c_next)**2) * dh_next + dc_next)
        dg = i * dc
        di = g * dc
        df = c_next * dc

        # Compute gradient of the activation functions
        di_input = dsigmoid(i) * di
        df_input = dsigmoid(f) * df
        do_input = dsigmoid(o) * do
        dg_input = dtanh(g) * dg

        # Gradients of weights and biases
        self.dWi = np.dot(di_input, concat.T)
        self.dWf = np.dot(df_input, concat.T)
        self.dWo = np.dot(do_input, concat.T)
        self.dWg = np.dot(dg_input, concat.T)
        
        self.dbi = np.sum(di_input, axis=1, keepdims=True)
        self.dbf = np.sum(df_input, axis=1, keepdims=True)
        self.dbo = np.sum(do_input, axis=1, keepdims=True)
        self.dbg = np.sum(dg_input, axis=1, keepdims=True)

        # Gradient w.r.t to h_prev and x
        dconcat = np.dot(self.Wi.T, di_input) + np.dot(self.Wf.T, df_input) + np.dot(self.Wo.T, do_input) + np.dot(self.Wg.T, dg_input)
        dh_prev = dconcat[:dh_next.shape[0], :]
        dx = dconcat[dh_next.shape[0]:, :]

        return dx, dh_prev



# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0, keepdims=True)


# Modified LSTM for classification
class ClassificationLSTM(LSTM):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size)
        self.Wy = np.random.randn(output_size, hidden_size)
        self.by = np.zeros((output_size, 1))

    def forward(self, x, h_prev, c_prev):
        h_next, c_next, cache = super().forward(x, h_prev, c_prev)
        y_pred = softmax(np.dot(self.Wy, h_next) + self.by)
        return y_pred, h_next, c_next, cache
    
    def backward(self, dy, cache, concat):
        dh = np.dot(self.Wy.T, dy)
        dx, dh_prev = super().backward(dh, np.zeros_like(dh), cache, concat)
        return dx, dh_prev

learning_rate = 0.01
n_epochs = 500
lstm = ClassificationLSTM(input_size, hidden_size, output_size)
# Training the LSTM

for epoch in range(n_epochs):
    total_loss = 0
    for i in range(X_train.shape[0]):
        h_prev = np.zeros((hidden_size, 1))
        c_prev = np.zeros((hidden_size, 1))

        caches = []
        concat_inputs = []

        # Forward pass for each feature in sequence
        for j in range(4):
            y_pred, h_prev, c_prev, cache = lstm.forward(X_train[i][j], h_prev, c_prev)
            caches.append(cache)
            concat_input = np.vstack((h_prev, X_train[i][j]))
            concat_inputs.append(concat_input)

        # Calculate the loss using categorical cross-entropy
        loss = -np.sum(y_train[i] * np.log(y_pred))
        total_loss += loss

        # Backward pass: We use the simplified gradient for softmax and cross-entropy
        dy = y_pred - y_train[i].reshape(3, 1)
        lstm.Wy -= learning_rate * np.dot(dy, h_prev.T)
        lstm.by -= learning_rate * dy

        # Now, backward prop through LSTM
        _, _ = lstm.backward(dy, caches[-1], concat_inputs[-1])  # We only backpropagate through the last step
    # Accuracy
    if epoch % 10 == 0:
        correct_predictions = 0
        for i in range(X_test.shape[0]):
            h_prev = np.zeros((hidden_size, 1))
            c_prev = np.zeros((hidden_size, 1))

            for j in range(4):
                y_pred, h_prev, c_prev, cache = lstm.forward(X_test[i][j], h_prev, c_prev)

            if np.argmax(y_pred) == np.argmax(y_test[i]):
                correct_predictions += 1

        accuracy = correct_predictions / X_test.shape[0]
        avg_loss = total_loss / X_train.shape[0]
        print(f"Epoch {epoch+1}/{n_epochs}, Accuracy: {accuracy*100:.2f}%")