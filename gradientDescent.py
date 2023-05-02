from operator import itemgetter
from re import X
import numpy as np
from matplotlib import pyplot as plt
from computeCost import computeCost


def gradientDescent(X_train,y_train,alpha,iters):
    theta = np.random.rand(X_train.shape[1],1)
    #print(theta.shape) 
    cost = np.zeros((iters,1))
    for it in range(iters):
        sum = 0
        theta_T = np.transpose(theta)
        #print(theta_T.shape) 
        for i in range(X_train.shape[0]):
            for j in range(X_train.shape[1]):
                #h = np.matmul(theta_T,X_train[:][j])
                #print(X_train[i][:])
                h = np.matmul(X_train[i][:],theta)       
                #print(h)
                dist = (h -y_train[i])*X_train[i][j]
                sum = sum + dist
                theta[j] = theta[j] - (alpha*sum)/X_train.shape[0]
                cost[it] = computeCost(np.transpose(X_train[:][i]),y_train[i],theta)  
    #print("cost: " + str(cost[it]))
    return theta, cost

#X_train = np.array([[1,1,1],[1,2,2],[1,3,3],[1,4,4]])
#y_train = np.array([[2],[4],[6],[8]])
#theta, cost = gradientDescent(X_train,y_train,.001,15)
#print("theta: " + str(theta))
#print("cost: " + str(cost))

