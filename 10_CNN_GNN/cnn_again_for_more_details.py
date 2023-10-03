# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 09:45:02 2023

@author: xxjan
"""

import numpy as np
import torch
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
np.random.seed(123)

# 1. Preprocess the dataset
digits = datasets.load_digits()
X = digits.data.reshape(digits.data.shape[0], -1, 1)  # treat features as a sequence
y = digits.target.reshape(-1, 1)

# One-hot encode the labels
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
#y_train_inver = encoder.

def sigmoid(x):
    z = 1/(1 + np.exp(-x))
    return z


def sigmoid_derivative(x):
    z = 1/(1 + np.exp(-x))
    return z * (1 - z)





def relu_conv(x):
    if len(x.shape)!=4:
        raise ValueError("The dimension is not correct to activation function")
    (N,C,H,W) = x.shape 
    y = np.zeros((N,C,H,W))
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    y[n,c,h,w] = np.max(x[n,c,h,w],0)
    return y

def flatten(x):
    return x.reshape(x.shape[0], -1)


def conv2d(x,kernel,bias,padding,stride):
    if x.shape[2]==1:
        x = x.reshape(x.shape[0],int(np.sqrt(x.shape[1])),int(np.sqrt(x.shape[1])))
    #print(x[0])
    out_channel,kernelsize,_ = kernel.shape  
    if len(x.shape)==3:
        N,H,W = x.shape
        C = out_channel
        #print("N：{}, C:{}, H:{}, W:{}".format(N,C,H,W))
    elif len(x.shape) == 4:
        (N,C,H,W) = x.shape
        #print("N：{}, C:{}, H:{}, W:{}".format(N,C,H,W))
    
    
    H_out = (H-kernelsize+2*padding)//stride+1
    W_out = (W-kernelsize+2*padding)//stride+1
    #print("H_out:{},W_out：{}".format(H_out,W_out))
    output_x = np.zeros((N,out_channel,H_out,W_out))
    for n in range(N):
        for c in range(C):
            #print("this is the update", c)
            for h in range(H_out):
                for w in range(W_out):
                    a = np.mat(x[n,h*stride:h*stride+kernelsize,w*stride:w*stride+kernelsize])
                    b = np.mat(kernel[c,:,:])
                    output_x[n,c,h,w] = np.sum(np.dot(a,b))+bias[c]
            #print(output_x[n,c,:,:])                 
    return output_x

def max_pool(x,pool_size):
    N, channels, H, W = x.shape
    output = np.zeros((N, channels,H//pool_size, W//pool_size))
    
    label = []
    for n in range(N):
        for c in range(channels):
            for i in range(0, H, pool_size):
                for j in range(0, W, pool_size):
                    #print(np.argmax(x[n,c, i:i+pool_size, j:j+pool_size]))
                    label.append(np.argmax(x[n,c, i:i+pool_size, j:j+pool_size]))
                    output[n,c, i//pool_size, j//pool_size] = np.max(x[n,c, i:i+pool_size, j:j+pool_size], axis=(0,1))
    label = np.array(label)
    
    label = label.reshape(N,int((H//pool_size)**2))
    #print(label)
    return output,label



def unpool(x, pool_size,label):
    N,C,H,W = x.shape
    output = np.zeros((N, C, H * pool_size, W * pool_size))
    #print("===============unpool process start =========================")
    #print("N：{}, C:{}, H:{}, W:{}".format(N,C,H* pool_size,W* pool_size))
    def match_function(matrix,k,x):
        if matrix.shape[0]!=2:
            raise ValueError("We can not solve this problem right now")
        if k ==0:
            matrix[0,0] = x
        elif k==1:
            matrix[0,1] = x
        elif k==2:
            matrix[1,0] = x
        elif k==3:
            matrix[1,1] = x
        return matrix
    
    
    for n in range(N):
        index_ = label[n]
        label_matrix = index_.reshape(H,W)
        #print(label_matrix.shape)
        for c in range(C):
            for i in range(0,int(H*pool_size),pool_size):
                for j in range(0,int(W*pool_size),pool_size):
                    ind = (label_matrix[i//pool_size,j//pool_size])
                    mat = output[n,c, i:i+pool_size, j:j+pool_size]
                    output[n,c, i:i+pool_size, j:j+pool_size] = match_function(mat,ind,x[n,c,i//pool_size,j//pool_size])
                              
    #print("===============unpool process finished=======================")
    return output
     

def conv_backward(x,grad_output,kernel_size,padding,stride):
    if x.shape[2]==1:
        x = x.reshape(x.shape[0],int(np.sqrt(x.shape[1])),int(np.sqrt(x.shape[1])))
    #print(x[0])
    if len(x.shape)==3:
        N,H,W = x.shape
        C = 1
    #print("============conv backward process start =====================")
    H_out = (H - kernel_size + 2*padding) // stride + 1
    W_out = (W - kernel_size + 2*padding) // stride + 1
    #print("H_out shape is {}, W_out shape is {}".format(H_out,W_out))
    grad_weight = np.zeros((C, kernel_size, kernel_size))
    grad_bias = np.zeros((C))
    # update weights and bias, grad_input
    for n in range(N):
         for c in range(C):
             for i in range(H_out):
                 for j in range(W_out):
                    
                      grad_weight[c, :, :] += grad_output[n, c, i, j] * \
                              np.multiply(np.maximum(x[n, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size],0),x[n, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size])
                      #print(grad_bias[c])
                      grad_bias[c] += grad_output[n, c, i, j]
                      
    #                 grad_input[n, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size].data += \
    #                      grad_output[n, c, i, j] * self.weight[c, :, :, :]
    #print("============conv backward process finish ====================")
    return grad_weight,grad_bias


class CNN_again():
    def __init__(self,input_data,output_channels,kernelsize,stride=1,padding=0):
        
        self.input_data = input_data
        self.output_channels = output_channels
        self.kernelsize = kernelsize
        self.padding = padding
        self.stride = stride
        self.fc_units = 10
        self.lr = 1.0
        self.weights = {
            "conv1": #np.ones((self.output_channels, self.kernelsize, self.kernelsize)),
                np.random.randn(self.output_channels, self.kernelsize, self.kernelsize),
            "fc": np.random.randn(self.kernelsize * self.kernelsize * self.output_channels, self.fc_units),  # Adjust the weight matrix size
        }
        
        #print('kernel size is',self.weights['conv1'].shape)
        
        self.bias = {
            "conv1": np.zeros(self.output_channels),
            "fc": np.zeros(self.fc_units),
        }
        #print('bias size is',self.bias['conv1'].shape)
        
    def forward(self,x):
        self.input = x
        self.conv_out = conv2d(x, self.weights["conv1"], self.bias["conv1"],self.padding,self.stride)

        self.relu_out = sigmoid(self.conv_out)
        #print(np.argmax(self.relu_out[0,0,:,:]))
        self.pool_out,self.label = max_pool(self.relu_out, 2)
        self.flat_out = flatten(self.pool_out)
        
        
        self.output = np.dot(self.flat_out, self.weights["fc"]) + self.bias["fc"]
        
        self.output2 = sigmoid(self.output)
        #print("===============forward process is finished ============")
        return self.output2
    
    def mse_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    
    def mse_loss_grad(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.size
    

    
    def backward(self,x,y_true):
        loss_grad = self.mse_loss_grad(self.output2, y_true)
        
        a = np.multiply(sigmoid_derivative(self.flat_out),self.flat_out)
        
        # Gradient for FC layer
        fc_weight_grad = np.dot(a.T, loss_grad)
        
        fc_bias_grad = np.sum(loss_grad, axis=0)
        
        # Gradient after flattening
    
        b = np.multiply(sigmoid_derivative(self.weights["fc"]),self.weights["fc"])
        flat_grad = np.dot(loss_grad, b.T)
        #print(flat_grad.shape)        
        pool_grad = flat_grad.reshape(self.pool_out.shape)
        #print(pool_grad.shape)
        
        # Unpool the gradient
        #[label[i]] = x[n,c, i//pool_size, j//pool_size]
        unpool_grad = unpool(pool_grad, 2,self.label)
        print(unpool_grad.shape)
        
        
        
        
        # Update weights and biases
        self.weights["fc"] -= self.lr * fc_weight_grad
        self.bias["fc"] -= self.lr * fc_bias_grad
        # Calculate gradients for convolutional layer
        #print(x.shape)
        grad_weight,grad_bias = conv_backward(x,unpool_grad,self.kernelsize,self.padding,self.stride)

        # Update convolutional weights
        self.weights["conv1"] -= self.lr * grad_weight
        self.bias["conv1"] -= self.lr * grad_bias
        
    def train(self, X, y, num_epochs):
         for epoch in range(num_epochs):
             losses = []
             
             y_pred = self.forward(X)
             loss = self.mse_loss(y_pred, y)
             print("loss is",loss)
             losses.append(loss)
             self.backward(X, y)
             print(f"Epoch {epoch}, Accuracy: {self.accuracy(X, y)}")

    def accuracy(self, X, y):
         predictions = self.forward(X)
         labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
         ypred = np.zeros(X.shape[0], dtype=int)
         yy = np.zeros(X.shape[0], dtype=int)
         for i in range(X.shape[0]):
           ypred[i] = labels[np.argmax(predictions[i, :])]
           yy[i] = labels[np.argmax(y[i, :])]
         correct = np.sum(ypred == yy)
         return correct 


model = CNN_again(X_train,1,3)
model.train(X_train,y_train,100)

accuracy = model.accuracy(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
# #    def __init__(self,x,kernel,bias):
# kernel = np.ones((1,3,3))
# bias = np.zeros(1)
# padding = 0
# stride = 1
# a = conv2d(X_train,kernel,bias,padding,stride)    
# b = max_pool(a,2)
# oo = b.reshape(b.shape[0],-1)
# print(oo.shape)
