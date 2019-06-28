# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import h5py


def sigmoid(z):
    
    s = 1/(1 + np.exp(-z))

    
    return s


def initialize_with_zeros(dim):
    
    
    w = np.zeros((dim,1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b


def propagate(w, b, X, Y):
    
    
    m = X.shape[1]
    
    A = sigmoid(np.dot(w.T,X)+b)                        # compute activation
    cost = (1/m)*np.sum(-Y*np.log(A)-((1-Y)*np.log(1-A)))      # compute cost
 
    dw = (1/m)*np.dot(X,(A-Y).T)
    db = (1/m)*np.sum(A-Y)


    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    
    costs = []
    
    for i in range(num_iterations):
        
    
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        

        w = w - learning_rate*dw
        b = b - learning_rate*db

        

        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs



def predict(w, b, X):
    
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T,X)+b)
    
    for i in range(A.shape[1]):
        
 
        if A[0,i] <= 0.5:
            Y_prediction[0,i]=0
        else: 
            Y_prediction[0,i]=1
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction