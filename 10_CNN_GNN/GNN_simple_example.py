# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:30:58 2023

@author: xxjan
"""
import numpy as np
#Define the graph structure for adjacency matrix
A = np.array([
    [0,1,0],
    [1,0,1],
    [0,1,0]
    ])

X = np.array(
    [1,2],
    [2,3],
    [3,1]
    )

class graphneuralnetwork:
    def __init__(self,T,D):
        sigma = 0.4
        self.T = T
        self.D = D
        self.W = sigma * np.random.randn(D,D)
        self.A = sigma * np.random.randn(D)
        self.b = 0

        self.dLdW = np.zeros((D,D))
        self.dLdA = np.zeros((D))
        self.dLdb = 0
    def aggreation1(self,X,adj):
        a = np.dot(adj,X)
        return a
    
    def aggreation2(self,W,a):
        x = np.dot(W,np.tranpose(a))
        x = np.transpose(x)
        return x
    
    def relu(self,inp):
        out = np.maximum(0,inp)
        return out
    
    def readout(self,X):
        hG = np.sum(X,axis=0)
        return hG
    
    def s(self,hG,A,b):
        s = np.dot(hG,A)+b
        return s

    def sigmoid(self,s):
        p = 1/(1+np.exp(-s))
        return p

    def output(self,p):
        out = np.where((p>0.5),1,0)
        return out

    def forward(self, nnodes, adj, W = None, A = None, b = None):
        """
        forward method to calculate forward propagation of the nets
        Args :
            nnodes  : number of nodes in the batch
            adj     : adjacency matrix
            W       : parameter matrix W
            A       : parameter vector A
            b       : bias b
        Return : 
            slist       : vector of predictor value 
            output list : vector of predicted class`
        """
        slist = []
        outputlist = []

        X = []
       
        # feature vector definition
        feat =  np.zeros(self.D)
        feat[0] = 1
        

        self.tempnnodes = nnodes

        self.tempadj = adj

        if np.any(W == None) :
            W = self.W
        
        if np.any(A == None) :
            A = self.A
        
        if b == None :
            b = self.b

        for i in range(adj.shape[0]):
            X.append(np.tile(feat,[nnodes[i],1]))
            for j in range(self.T):
                a = self.aggregation1(X[i],adj[i])
                x = self.aggregation2(W,a)
                out = self.relu(x)
                X[i] = out
            hG = self.readout(X[i])
            s = self.s(hG,A,b)
            p = self.sigmoid(s)
            output = self.output(p)
            slist.append(s)
            outputlist.append(int(output))

        
        return slist,outputlist
    
    def loss(self,s,y):
        """
        loss function
        Args :
            s   : vector of predictor values
            y   : vector of true class labels
        Return :
            losslist : vector of loss values
        """
        losslist = []
        for i in range (len(s)):
            if np.exp(s[i]) > np.finfo(type(np.exp(s[i]))).max:
                loss = y[i]*np.log(1+np.exp(-s[i])) + (1-y[i]) * s[i] #avoid overflow
            else :
                loss = y[i]*np.log(1+np.exp(-s[i])) + (1-y[i]) * np.log(1+np.exp(s[i]))
            losslist.append(loss)

        return losslist
            
    def updateweight(self,W,A,b):
        """
        update weight function
        Args :
            W: parameter matrix W
            A: parameter vector A
            b: bias b
        """

        self.W = W
        self.A = A
        self.b = b

    def backward(self,output,y_true,adj):
       # Compute the gradient of the loss w.r.t. the network's output (assumes binary cross-entropy loss)
       grad_output = output - y_true
    
       # Backprop through the second linear layer (with weights A and bias b)
       dLdA = np.dot(self.hG.T, grad_output)
       dLdb = np.sum(grad_output, axis=0)
    
       # Backprop through the readout (sum)
       grad_hG = np.dot(grad_output, self.A.T)
    
       # Backprop through the ReLU activation
       grad_relu = grad_hG * (self.relu_out > 0)
    
       # Backprop through the first linear layer (with weights W)
       dLdW = np.dot(X.T, np.dot(adj, grad_relu))
    
       # Update gradients
       self.dLdW = dLdW
       self.dLdA = dLdA
       self.dLdb = dLdb





    
        
        
        
        