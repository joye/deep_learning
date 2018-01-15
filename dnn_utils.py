# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 21:31:48 2018

@author: joye
"""

import numpy as np

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def tanh_activate(Z):
    A = np.tanh(Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z<=0] = 0
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA*s*(1-s)
    return dZ

def tanh_backward(dA, cache):
    Z = cache
    dZ = dA*(1-np.power(Z, 2))
    return dZ