# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 22:02:12 2018
deep learning neural network
@author: joye
tips : don't use rank one array
"""

import numpy as np
from dnn_utils import sigmoid, relu, tanh_activate, sigmoid_backward, relu_backward, tanh_backward
#import matplotlib.pyplot as plt

class NeuralNet:
    def __init__(self, optimizer):
        'initialize deep neural network'
        self.optimizer = optimizer
        self.parameters = {}
        self._grads      = {}
        self.caches     = []
        self.activation = ''
        
    def feed(self, train_x, train_y):
        """
        input training set
        the training set's format is like this
        if one train_x element has 2 components, such as x1, x2
        x1_1 x2_1 x3_1 ...
        x1_2 x2_2 x3_2 ...
        the first number means training data's index, the second number means which component in one training data
        can use pandas pkgs to get this format 
        """
        self.train_x = train_x
        self.train_y = train_y
        #self.train_y = self.train_y.reshape(len(self.train_y), 1) #avoid rank one array
        
    def set_activation(self, activation):
        self.activation = activation
    
    def initialize_layers(self, layers_dim):
        """
        layers_dim format is each hidden layer's unit number 
        and output layer's unit number, for example, the format is
        [2,4,1] which means 2 hidden layers and 1 output layer,
        the first hidden layer has 2 units, the second hidden layer has
        4 units, and output layer has 1 output units.
        """
        layers_dim.insert(0, self.train_x.shape[0])
        L = len(layers_dim)
        for l in range(1, L):
            self.parameters['W'+str(l)] = np.random.randn(layers_dim[l], layers_dim[l-1]) * 0.01
            self.parameters['b'+str(l)] = np.zeros((layers_dim[l],1))
            assert(self.parameters['W' + str(l)].shape == (layers_dim[l], layers_dim[l-1]))
            assert(self.parameters['b' + str(l)].shape == (layers_dim[l], 1))
    
    def __linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache
    
    def __linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = 1./m * np.dot(dZ, A_prev.T)
        db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db
    
    def __linear_activation_forward(self, A_prev, W, b, activation):
        """
        A_prev -- activations from previous layer or input data(size of previous layer, number of examples)
        W -- weight matrix (size of current layer, previous layer)
        b -- bias vector (size of current layer, 1)
        
        returns:
            A -- the output of activation function
            cache -- a directory contain linear cache and activation cache, used in back propagation 
        """
        Z, linear_cache = self.__linear_forward(A_prev, W, b)
        if activation == "sigmoid":
            A,activation_cache = sigmoid(Z)
        elif activation == "relu":
            A,activation_cache = relu(Z)
        elif activation == "tanh":
            A, activation_cache = tanh_activate(Z)
        assert(A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)
        return A, cache
    
    def __linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
        elif activation == "tanh":
            dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = self.__linear_backward(dZ, linear_cache)
        return dA_prev, dW, db
        
    def __forward(self, X, activation):
        """
        forward propogation
        activation -- the activation to be used in the hidden layer, as a string: "sigmoid" or "relu" or "tanh"
        """
        A = X
        L = len(self.parameters) // 2
        for l in range(1, L):
            A_prev = A
            A, cache = self.__linear_activation_forward(A_prev, self.parameters['W'+str(l)], self.parameters['b'+str(l)], activation)
            self.caches.append(cache)
        AL, cache = self.__linear_activation_forward(A, self.parameters['W'+str(L)], self.parameters['b'+str(L)], "sigmoid")
        self.caches.append(cache)
        assert(AL.shape == (1, self.train_x.shape[1]))        
        return AL
    
    def __backward(self, AL, activation):
        L = len(self.caches)
        self.train_y = self.train_y.reshape(AL.shape)
        dAL = - (np.divide(self.train_y, AL) - np.divide(1 - self.train_y, 1 - AL))
        current_cache = self.caches[L-1]
        self._grads["dA" + str(L)], self._grads["dW" + str(L)], self._grads["db" + str(L)] = self.__linear_activation_backward(dAL, current_cache, "sigmoid")
        for l in reversed(range(L-1)):
            current_cache = self.caches[l]
            dA_prev_temp, dW_temp, db_temp = self.__linear_activation_backward(self._grads["dA"+str(l+2)], current_cache, activation)
            self._grads["dA" + str(l + 1)] = dA_prev_temp
            self._grads["dW" + str(l + 1)] = dW_temp
            self._grads["db" + str(l + 1)] = db_temp
    
    def __update_parameters(self, learning_rate):
        L = len(self.parameters) // 2
        for l in range(L):
            self.parameters["W"+str(l+1)] = self.parameters["W"+str(l+1)] - learning_rate * self._grads["dW" + str(l+1)]
            self.parameters["b"+str(l+1)] = self.parameters["b"+str(l+1)] - learning_rate * self._grads["db" + str(l+1)]

        
    def __compute_cost(self, AL):
        m = self.train_y.shape[1]
        cost = -1./m*np.sum(self.train_y * np.log(AL) + (1-self.train_y)*np.log(1-AL))
        cost = np.squeeze(cost)
        return cost
        
    def train(self, learning_rate, num_iteration):
        for i in range(0, num_iteration):
            AL = self.__forward(self.train_x,self.activation)
            cost = self.__compute_cost(AL)
            self.__backward(AL, self.activation)
            self.__update_parameters(learning_rate)
            
            self.caches.clear()  #after each iteration clear cache data
            #if i % 100 == 0:
            print("cost after iteration {} : {}".format(i, cost))
    
    def predict(self, test_x):
       # m = test_x.shape[1]
        AL = self.__forward(test_x, self.activation)
        predict_result = AL > 0.5
        return predict_result
        #print("Accuracy: " + str(np.sum(predict_result == test_y)/m))        