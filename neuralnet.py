import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import random

class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x<=0)
        out = torch.tensor(x)
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        out = 1 / (1 + torch.exp(-x))
        self.out = out
        
        return out
    
    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        
        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.x = torch.tensor(x)
        
        out = torch.matmul(self.x, self.W) + self.b
        
        return out
    
    def backward(self, dout):
        dx = torch.mm(dout, self.W.transpose(0,1))
        self.dW = torch.mm(self.x.transpose(0,1), dout)
        self.db = torch.sum(dout, dim = 0)
        
        return dx

def softmax(x):
    if x.dim() == 2:
        x = x.transpose(0,1)
        x = x - torch.max(x)
        y = torch.exp(x) / torch.sum(torch.exp(x), dim=0)
        return y.transpose(0,1) 
    
    x = x - torch.max(x) 
    return torch.exp(x) / torch.sum(torch.exp(x))

def cross_entropy_error(y, t):
    batch_size = y.shape[0]
    return - (torch.sum(torch.log(y + 1e-7)) / batch_size)

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        
        return dx

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * torch.randn(input_size, hidden_size)
        self.params['b1'] = torch.zeros(hidden_size)
        self.params['W2'] = weight_init_std * torch.randn(hidden_size, output_size)
        self.params['b2'] = torch.zeros(output_size)
        
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x
    
    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = torch.argmax(y, dim=1)
        t = torch.tensor(t, dtype=torch.double)
        if t.dim() != 1:
            t = torch.argmax(t, dim=1)
        
        accuracy = (torch.sum(y == t).to(dtype=torch.float) / float(x.shape[0])).item() * 100
        return accuracy
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        
        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        
        return grads

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
#print(x_train.shape) #(60000, 784)
#print(t_train.shape) #(60000,)
#print(x_test.shape) #(10000, 784)
#print(t_test.shape) #(10000,)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    zipped = list(zip(x_train, t_train))
    test_zipped = list(zip(x_test, t_test))

    random.shuffle(zipped)
    random.shuffle(test_zipped)

    batch = zipped[:batch_size]
    test_batch = test_zipped[:batch_size]

    x_batch = []
    t_batch = []
    for x, t in batch:
        x_batch.append(x)
        t_batch.append(t)
        
    x_batch = torch.tensor(x_batch, dtype=torch.float)
    t_batch = torch.tensor(t_batch, dtype=torch.float)

    test_x_batch = []
    test_t_batch = []
    for x, t in test_batch:
        test_x_batch.append(x)
        test_t_batch.append(t)
        
    test_x_batch = torch.tensor(test_x_batch, dtype=torch.float)
    test_t_batch = torch.tensor(test_t_batch, dtype=torch.float)
        
    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_batch, t_batch)
        test_acc = network.accuracy(test_x_batch, test_t_batch)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("Train acc: %.2f %%, Test acc: %.2f %%" %(train_acc, test_acc))