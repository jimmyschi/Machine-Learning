from cmath import inf
import numpy as np
import math

from predict import predict

def nnCost(Theta1,Theta2,X,y,K,lamb):
    y_k = np.zeros((y.shape[0],3))
    J = 0
    for i in range(y_k.shape[0]):
        if y[i] == 1:
            y_k[i] = [1,0,0]
        elif y[i] == 2:
            y_k[i] = [0,1,0]
        elif y[i] == 3:
            y_k[i] = [0,0,1]
    #print("y_k: " + str(y_k))
    first_term = 0
    p,h_x = predict(Theta1,Theta2,X)
    for m in range(X.shape[0]):
        for k in range(K):
            if h_x[m][k] != 1:
                first_term += (y_k[m][k])*math.log(h_x[m][k]) + (1-y_k[m][k])*math.log(1-h_x[m][k])

    #print("first_term: " + str(first_term))
    reg_term = 0
    for m in range(8):
        for n in range(5):
           reg_term += Theta1[m][n]**2
    for m in range(3):
        for n in range(9):
            reg_term += Theta2[m][n]**2
    #print("reg_term: " + str(reg_term))
    J = (-first_term/X.shape[0]) + (lamb*reg_term)/(2*X.shape[0])
    return J