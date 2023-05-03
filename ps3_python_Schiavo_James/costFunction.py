import numpy as np
import math
from sigmoid import sigmoid

def costFunction(theta,X_train,y_train):
    m = X_train.shape[0]
    sum = 0
    h = sigmoid(np.matmul(X_train,theta))
    c = (-y_train*np.log10(h) - (1 - y_train)*np.log10(1 - h))
    sum = sum + np.sum(c)
    cost = sum/m
    return cost

X_train = np.array([[1,0,1],[1,0,3],[1,2,0],[1,2,1]])
y_train = np.array([[0],[1],[0],[1]])
theta = np.array([[0],[0],[0]])
cost = costFunction(theta,X_train,y_train)
