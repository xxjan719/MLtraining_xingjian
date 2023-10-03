# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 12:12:43 2023

@author: xxjan
"""
import math
import torch
import sys
from torch.nn.parameter import Parameter

class Conv2d:
    def __init__(self,in_channels,out_channels,kernel_size,lr=0.0001,stride=1,padding=0,bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lr = lr
        
        self.weight = Parameter(torch.zeros((out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)) # requires_grad=True
        self.bias = Parameter(torch.zeros(out_channels,requires_grad=True)) if bias else None
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_channels*self.kernel_size*self.kernel_size)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x_in):
        N, C, H, W = x_in.shape
        print("N：{}, C:{}, H:{}, W:{}".format(N,C,H,W))
        H_out = (H - self.kernel_size + 2*self.padding) // self.stride + 1 # // the floor division
        W_out = (W - self.kernel_size + 2*self.padding) // self.stride + 1
        x_out = torch.zeros((N, self.out_channels, H_out, W_out))
        for n in range(N):
            for c in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        x_out[n, c, i, j] = torch.sum(x_in[n, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] \
                                * self.weight[c, :, :, :]) + self.bias[c]
        return x_out

    def backward(self, x, grad_output):

        N, C, H, W = x.shape
        H_out = (H - self.kernel_size + 2*self.padding) // self.stride + 1
        W_out = (W - self.kernel_size + 2*self.padding) // self.stride + 1
        grad_input = torch.zeros((N, C, H, W))
        # update weights and bias, grad_input
        for n in range(N):
            for c in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        # this is right. But adding/substructing to self.weight[c, :, :, :] is illegal 
                        # RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation.
                        self.weight[c, :, :, :].data -= self.lr * grad_output[n, c, i, j] * \
                                x[n, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                        self.bias[c].data -= self.lr * grad_output[n, c, i, j]
                        grad_input[n, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size].data += \
                                grad_output[n, c, i, j] * self.weight[c, :, :, :]
        return grad_input

def test_Conv2d_backward():
    # Set up inputs and known outputs
    input_tensor = torch.tensor([[[[1., 2., 3., 4.],
                                   [5., 6., 7., 8.],
                                   [9., 10., 11., 12.],
                                   [13., 14., 15., 16.]]]], requires_grad=True)

    known_grad_output = torch.tensor([[[[1., 2.],
                                        [3., 4.]],
                                       [[-1., -2.],
                                        [-3., -4.]]]])

    # Initialize the Conv2d layer
    conv2d_layer = Conv2d(1, 2, 3)
    conv2d_layer.weight.data = torch.ones((2, 1, 3, 3))
    conv2d_layer.bias.data = torch.zeros(2)

    # Forward pass
    output_tensor = conv2d_layer.forward(input_tensor)

    # Clear gradients before backward pass
    input_tensor.grad = None
    conv2d_layer.weight.grad = None
    conv2d_layer.bias.grad = None

    # Backward pass
    grad_input = conv2d_layer.backward(input_tensor, known_grad_output)
    print(grad_input)
    sys.exit()

    #Check the computed gradients
    assert torch.allclose(input_tensor.grad, torch.tensor([[[[1., 3., 5., 3.],
                                                            [3., 6., 10., 6.],
                                                            [5., 10., 14., 8.],
                                                            [2., 4., 6., 3.]]]]))
    assert torch.allclose(conv2d_layer.weight.grad, torch.tensor([[[[28., 34., 40.],
                                                                    [52., 58., 64.],
                                                                    [76., 82., 88.]]],
                                                                  [[-28., -34., -40.],
                                                                   [-52., -58., -64.],
                                                                   [-76., -82., -88.]]]))
    assert torch.allclose(conv2d_layer.bias.grad, torch.tensor([10., -10.]))


if __name__ == '__main__':
    #from utils import load_data
    #trainloader, testloader = load_data(batch_size = 1)
    ## get the first data
    #dataiter = iter(trainloader)
    #images, labels = next(dataiter)

    #m = Conv2d(1, 2, 3)
    #m.forward(images)

    test_Conv2d_backward()
   