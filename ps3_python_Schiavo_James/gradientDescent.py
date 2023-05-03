import numpy as np
from sigmoid import sigmoid

def gradientDescent(theta,X_train,y_train):
    gradient = np.zeros((theta.shape[0],1))
    theta_t_x = np.matmul(X_train,theta)
    #print("!!!!!!!!!theta_t_x: ", theta_t_x.shape)
    h  = sigmoid(theta_t_x)
    #print("!!!!!!!!!h: ", h.shape)
    gradient =  np.matmul((h - y_train), X_train )
    gradient = np.mean(gradient, axis = 0)
    #print("gradient: " , gradient.shape)
    return gradient 

#X_train = np.array([[1,0,1],[1,0,3],[1,2,0],[1,2,1]])
#y_train = np.array([[0],[1],[0],[1]])
#theta = np.array([[0],[0],[0]])
#gradient = gradientDescent(theta,X_train,y_train)
#print("gradient: " + str(gradient))