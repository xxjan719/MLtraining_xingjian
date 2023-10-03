# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 22:42:18 2023

@author: xxjan
"""
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1.0 - y**2

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

# Generate a sine wave dataset
num_points = 200
x = np.linspace(0, 10, num_points)
y = np.sin(x)

# Hyperparameters
learning_rate = 0.01
n_epochs = 10000
hidden_size = 64
input_size = 1
output_size = 1

lstm = LSTM(input_size, hidden_size)

# Training loop
for epoch in range(n_epochs):
    h_prev = np.zeros((hidden_size, 1))
    c_prev = np.zeros((hidden_size, 1))
    total_loss = 0

    for i in range(num_points - 1):
        # Prepare the input and target data
        input_val = y[i].reshape((1, 1))
        target_val = y[i+1].reshape((1, 1))

        # Forward pass
        h, c, cache = lstm.forward(input_val, h_prev, c_prev)

        # Compute the loss
        loss = np.mean((h - target_val)**2)  # mean squared error
        total_loss += loss

        # Backward pass
        dh = h - target_val
        dc = np.zeros_like(c)
        dx, dh_prev = lstm.backward(dh, dc, cache, np.vstack((h_prev, input_val)))

        # Update weights and biases
        lstm.Wi -= learning_rate * lstm.dWi
        lstm.Wf -= learning_rate * lstm.dWf
        lstm.Wo -= learning_rate * lstm.dWo
        lstm.Wg -= learning_rate * lstm.dWg

        lstm.bi -= learning_rate * lstm.dbi
        lstm.bf -= learning_rate * lstm.dbf
        lstm.bo -= learning_rate * lstm.dbo
        lstm.bg -= learning_rate * lstm.dbg

    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss / num_points}")

# Test the LSTM
test_points = np.linspace(10, 15, num_points)
actual_sin = np.sin(test_points)
predicted_sin = []

h_prev = np.zeros((hidden_size, 1))
c_prev = np.zeros((hidden_size, 1))

for i in range(num_points - 1):
    input_val = actual_sin[i].reshape((1, 1))
    h, c, _ = lstm.forward(input_val, h_prev, c_prev)
    predicted_sin.append(h[0][0])
    h_prev = h
    c_prev = c

plt.plot(test_points, actual_sin, label='Actual')
plt.plot(test_points[:-1], predicted_sin, label='Predicted')
plt.legend()
plt.show()




























# for epoch in range(1000):
#     total_loss = 0
#     correct_predictions = 0

#     h_prev = np.zeros((1, 5))
#     c_prev = np.zeros((1, 5))

#     for x_data, y_true in zip(data, target):
#         x_seq = x_data.reshape(-1, 1)
#         y_pred_seq, h_prev, c_prev = lstm.forward(x_seq, h_prev, c_prev)
        
#         y_pred = y_pred_seq[-1]
        
#         total_loss += cross_entropy(y_pred, y_true)
#         if np.argmax(y_pred) == np.argmax(y_true):
#             correct_predictions += 1
#         y_seq = y_true
#         dy_seq = y_pred_seq - y_seq
#         dWi, dWf, dWo, dWc, dbi, dbf, dbo, dbc, dWy, dby = lstm.backward(dy_seq, y_seq, h_prev, c_prev, x_seq, h_prev, c_prev)

    
#     lstm.Wi -= learning_rate * dWi
#     lstm.Wf -= learning_rate * dWf
#     lstm.Wc -= learning_rate * dWc
#     lstm.Wo -= learning_rate * dWo
#     lstm.bi -= learning_rate * dbi
#     lstm.bf -= learning_rate * dbf
#     lstm.bc -= learning_rate * dbc
#     lstm.bo -= learning_rate * dbo
#     lstm.Wy -= learning_rate * dWy
#     lstm.by -= learning_rate * dby
#     avg_loss = total_loss / len(data)
#     accuracy = correct_predictions / len(data)

#     if epoch % 50 == 0:
#         print(f"Epoch: {epoch}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# print("Training completed.")